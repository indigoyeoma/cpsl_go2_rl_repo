from itertools import product
import json
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.nn.parallel import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather

from .model import CnnMlpNetwork
from .ghn_modules import MLP_GHN
# No need for complex Graph and GraphBatch imports

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class hyperActor(nn.Module):

    def __init__(self, 
                act_dim, 
                obs_dim, 
                architecture_config_path: str,
                meta_batch_size = 1,
                device = "cpu",
                multi_gpu = False,
                architecture_sampling_mode = "uniform",
                ):
        super().__init__()

        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.architecture_config_path = architecture_config_path
        self.meta_batch_size = meta_batch_size
        self.multi_gpu = multi_gpu
        self.std_mode = 'multi'  # Only use multi std mode
        self.architecture_sampling_mode = architecture_sampling_mode

        
        # initialize all devices for parallelization on multiple GPUs
        self._initialize_devices(device)

        # load architecture configurations from JSON
        self._load_architecture_configs()

        # initialize all list of CNN+MLP architectures
        self._initialize_cnn_mlp_architectures()

        # initialize architecture sampling data
        self._initialize_architecture_sampling_data()

        # initialize the original MLP_GHN
        self._initialize_ghn()

        # initialize standard deviation vectors
        self._initialize_std()
        

    def _initialize_std(self):
        ''' Initialize multi standard deviation vectors for each architecture '''
        # Initialize with small negative values for stability (similar to PPO)
        init_log_std = 0.0  # exp(-0.5) ≈ 0.6 std
        self.log_std = nn.ParameterList([
            nn.Parameter(torch.full((1, np.prod(self.act_dim)), init_log_std), requires_grad=False)
            for index in self.list_of_arc_indices
        ])  


    def _initialize_architecture_sampling_data(self):
        ''' Initialize architecture sampling data based on selected mode '''
        self.sampled_indices = None
        
        if self.architecture_sampling_mode == "uniform":
            # Uniform sampling - no special initialization needed
            self.arch_sampling_probs = None
            # print(f"🎯 Using uniform architecture sampling")
            
        elif self.architecture_sampling_mode == "biased":
            # Biased sampling based on architecture complexity (number of layers)
            self.arch_sampling_probs = []
            
            # Get number of layers for each architecture (CNN + MLP layers)
            arch_complexities = []
            for config in self.architecture_configs:
                num_cnn_layers = len(config['cnn_config'])
                num_mlp_layers = len(config['mlp_config'])
                total_layers = num_cnn_layers + num_mlp_layers
                arch_complexities.append(total_layers)
            
            # Calculate biased probabilities - favor architectures with unique complexity
            num_unique_complexities = len(set(arch_complexities))
            for i in self.list_of_arc_indices:
                complexity = arch_complexities[i]
                num_archs_with_same_complexity = arch_complexities.count(complexity)
                # Equal probability within each complexity group, normalized by number of groups
                prob = 1.0 / num_archs_with_same_complexity
                self.arch_sampling_probs.append(prob)
            
            # Normalize probabilities to sum to 1
            self.arch_sampling_probs = np.array(self.arch_sampling_probs)
            self.arch_sampling_probs = self.arch_sampling_probs / np.sum(self.arch_sampling_probs)
            
            # print(f"🎯 Using biased architecture sampling with {num_unique_complexities} complexity groups")
            # print(f"   Architecture complexities: {arch_complexities}")
            # print(f"   Sampling probabilities: {self.arch_sampling_probs}")
        
        else:
            raise ValueError(f"Unsupported architecture_sampling_mode: {self.architecture_sampling_mode}. Use 'uniform' or 'biased'.")


    def _load_architecture_configs(self):
        """Load architecture configurations from JSON file"""
        with open(self.architecture_config_path, 'r') as f:
            self.arch_data = json.load(f)
        
        self.metadata = self.arch_data['metadata']
        self.architecture_configs = self.arch_data['architectures']
        self.teacher_config = self.arch_data.get('teacher_model', None)
        
        # print(f"📁 Loaded {len(self.architecture_configs)} architectures from {self.architecture_config_path}")

    def _initialize_cnn_mlp_architectures(self):
        """Initialize CNN+MLP architectures from loaded configurations"""
        
        # Sort architectures by complexity (parameter count)
        self.architecture_configs.sort(key=lambda x: self._get_cnn_mlp_params(x))
        
        # Create list of architecture indices
        self.list_of_arc_indices = np.arange(len(self.architecture_configs))
        
        # Get state dimension from metadata
        self.state_dim = self.metadata['state_dim']
        
        # Create all CNN+MLP models with parallel state processing
        self.all_models = []
        for config in self.architecture_configs:
            model = CnnMlpNetwork(
                cnn_config=config['cnn_config'],
                cnn_mlp_config=config['cnn_mlp_config'],
                state_mlp_config=config['state_mlp_config'],  # State branch config
                mlp_config=config['mlp_config'],
                state_dim=self.state_dim,
                input_channels=self.metadata['input_channels'],
                output_dim=self.act_dim,
                input_size=self.metadata['input_size']
            )
            self.all_models.append(model)
        
        # Initialize architecture descriptors for GHN-2
        self._initialize_arch_descriptors()
        
        # Initialize detailed shape indices for GHN (like original HyperPPO)
        self._initialize_shape_inds()
        
        # Shuffle architecture indices
        np.random.shuffle(self.list_of_arc_indices)
        
        # print(f"🏗️  Initialized {len(self.all_models)} CNN+MLP models")


    def _initialize_arch_descriptors(self):
        """Initialize architecture descriptors for GHN-2"""
        
        self.arch_descriptors = []
        for config in self.architecture_configs:
            descriptor = torch.tensor(config['arch_descriptor'], dtype=torch.float32, device=self.device)
            self.arch_descriptors.append(descriptor)
        
        # Convert to tensor
        self.arch_descriptors = torch.stack(self.arch_descriptors)
        self.arch_max_len = self.metadata['arch_descriptor_config']['total_length']
        
        # print(f"📋 Initialized architecture descriptors: {self.arch_descriptors.shape}")

    def _initialize_shape_inds(self):
        """Initialize detailed shape indices for CNN+MLP architectures (like original HyperPPO)"""
        
        # Create string-to-index mapping for our CNN+MLP components
        component_vocab = {}
        component_index = 0
        
        # Build vocabulary from all architecture configs
        for config in self.architecture_configs:
            # CNN layers: "32ch3ker", "64ch3ker", etc.
            for layer in config['cnn_config']:
                desc = f"{layer['channels']}ch{layer['kernel']}ker"
                if desc not in component_vocab:
                    component_vocab[desc] = component_index
                    component_index += 1
            
            # CNN-MLP connector layers: "256mlp", "512mlp", etc.
            if 'cnn_mlp_config' in config and config['cnn_mlp_config']:
                for units in config['cnn_mlp_config']:
                    desc = f"{units}mlp"
                    if desc not in component_vocab:
                        component_vocab[desc] = component_index
                        component_index += 1
            
            # MLP layers: "128mlp", "256mlp", etc.
            for units in config['mlp_config']:
                desc = f"{units}mlp"
                if desc not in component_vocab:
                    component_vocab[desc] = component_index
                    component_index += 1
        
        # Output layer
        output_desc = f"{self.act_dim}out"
        component_vocab[output_desc] = component_index
        
        # print(f"📝 Built component vocabulary: {len(component_vocab)} unique components")
        self.component_vocab = component_vocab
        
        # Build detailed shape indices for each architecture (like original HyperPPO)
        self.list_of_shape_inds = []
        
        for config in self.architecture_configs:
            shape_ind = [torch.tensor(0.0).to(self.device)]  # Start token
            
            # CNN layers
            for layer in config['cnn_config']:
                desc = f"{layer['channels']}ch{layer['kernel']}ker"
                idx = self.component_vocab[desc]
                shape_ind.append(torch.tensor(float(idx)).to(self.device))
                shape_ind.append(torch.tensor(float(idx)).to(self.device))  # Double like original
            
            # CNN-MLP connector
            if 'cnn_mlp_config' in config and config['cnn_mlp_config']:
                for units in config['cnn_mlp_config']:
                    desc = f"{units}mlp"
                    idx = self.component_vocab[desc]
                    shape_ind.append(torch.tensor(float(idx)).to(self.device))
                    shape_ind.append(torch.tensor(float(idx)).to(self.device))
            
            # MLP layers
            for units in config['mlp_config']:
                desc = f"{units}mlp"
                idx = self.component_vocab[desc]
                shape_ind.append(torch.tensor(float(idx)).to(self.device))
                shape_ind.append(torch.tensor(float(idx)).to(self.device))
            
            # Output layer
            output_idx = self.component_vocab[output_desc]
            shape_ind.append(torch.tensor(float(output_idx)).to(self.device))
            shape_ind.append(torch.tensor(float(output_idx)).to(self.device))
            
            # Stack and reshape like original HyperPPO
            shape_ind = torch.stack(shape_ind).view(-1, 1)
            self.list_of_shape_inds.append(shape_ind)
        
        # Get lengths and max length like original HyperPPO
        self.list_of_shape_inds_lenths = [x.squeeze().numel() for x in self.list_of_shape_inds]
        self.shape_inds_max_len = max(self.list_of_shape_inds_lenths)
        
        # Pad with -1 to max length like original HyperPPO  
        for i in range(len(self.list_of_shape_inds)):
            num_pad = self.shape_inds_max_len - self.list_of_shape_inds[i].shape[0]
            if num_pad > 0:
                pad_tensor = torch.tensor(-1.0).to(self.device).repeat(num_pad, 1)
                self.list_of_shape_inds[i] = torch.cat([self.list_of_shape_inds[i], pad_tensor], 0)
        
        # Stack and reshape like original HyperPPO
        self.list_of_shape_inds = torch.stack(self.list_of_shape_inds)
        self.list_of_shape_inds = self.list_of_shape_inds.reshape(len(self.list_of_shape_inds), self.shape_inds_max_len)
        
        # print(f"📋 Initialized detailed shape indices: {self.list_of_shape_inds.shape}")


    def _initialize_devices(self, device):
        ''' Initialize device for single GPU usage '''
        self.device = device



    def _initialize_ghn(self):
        """Initialize simple MLP_GHN for weight generation (HyperPPO style)"""
        
        # Use architecture config's GHN settings if available
        if 'ghn_config' in self.metadata and 'ghn_max_shape' in self.metadata['ghn_config']:
            max_shape = tuple(self.metadata['ghn_config']['ghn_max_shape'])
        else:
            # Fallback to default HyperPPO shape (fixed 3x3 kernels)
            max_shape = (256, 256, 3, 3)
        
        # Calculate max observation size from architectures (approximate)
        max_obs_size = self.metadata['input_channels'] * self.metadata['input_size'] * self.metadata['input_size']
        
        self.ghn = MLP_GHN(
            max_shape=max_shape,
            num_classes=self.act_dim,
            num_observations=max_obs_size,
            hypernet='gatedgnn',
            decoder='conv',
            weight_norm=True,
            ve=False,
            layernorm=True,
            hid=32,
            device=self.device,
            debug_level=0
        ).to(self.device)
        
        # print(f"🧠 Initialized Simple MLP_GHN for CNN+MLP networks on device: {self.device}")



    def _get_cnn_mlp_params(self, config):
        """Get the number of parameters in a CNN+MLP architecture"""
        total_params = 0
        
        # CNN parameters
        prev_channels = self.metadata['input_channels']
        for layer in config['cnn_config']:
            channels = layer['channels']
            kernel = layer['kernel']
            # Conv params: (in_channels * kernel^2 + 1) * out_channels  
            conv_params = (prev_channels * kernel * kernel + 1) * channels
            total_params += conv_params
            prev_channels = channels
        
        # CNN MLP parameters (if exists)
        if 'cnn_mlp_config' in config and config['cnn_mlp_config']:
            # This is estimated since CNN output size is dynamic
            # We use a conservative estimate based on input size
            input_size = self.metadata['input_size']
            for layer in config['cnn_config']:
                input_size = (input_size + 2 * layer['padding'] - layer['kernel']) // layer['stride'] + 1
            
            cnn_output_size = prev_channels * input_size * input_size
            cnn_mlp_out = config['cnn_mlp_config'][0]
            total_params += (cnn_output_size + 1) * cnn_mlp_out
        
        # MLP parameters
        prev_size = config['mlp_config'][0]  # First MLP dimension
        for i in range(1, len(config['mlp_config'])):
            current_size = config['mlp_config'][i]
            total_params += (prev_size + 1) * current_size
            prev_size = current_size
        
        # Final layer to action space
        total_params += (prev_size + 1) * self.act_dim
        
        return total_params            

    def sample_arc_indices(self):
        ''' Sample architecture indices for the current meta batch based on sampling mode '''
        if self.architecture_sampling_mode == "uniform":
            # Uniform sampling without replacement
            self.sampled_indices = np.random.choice(self.list_of_arc_indices, self.meta_batch_size, replace=False)
        elif self.architecture_sampling_mode == "biased":
            # Biased sampling with probabilities, without replacement
            self.sampled_indices = np.random.choice(
                self.list_of_arc_indices, 
                self.meta_batch_size, 
                replace=False,
                p=self.arch_sampling_probs
            )
        else:
            raise ValueError(f"Unsupported architecture_sampling_mode: {self.architecture_sampling_mode}")



    def set_graph(self, indices_vector):
        ''' Set the graph to be used by the GHN-2. Generate weights for specified architectures. '''

        # Clear gradients of previous log_std parameters
        if self.sampled_indices is not None:
            for i in self.sampled_indices:
                self.log_std[i].requires_grad = False
                self.log_std[i].grad = None
        
        self.sampled_indices = indices_vector
        self.current_model = [self.all_models[i] for i in self.sampled_indices]
        self.current_configs = [self.architecture_configs[i] for i in self.sampled_indices]
        self.current_arch_descriptors = self.arch_descriptors[self.sampled_indices]
        
        # Create detailed shape indices like original HyperPPO
        self.current_shape_inds_vec = [self.list_of_shape_inds[index] for index in self.sampled_indices]
        self.list_of_sampled_shape_inds = [self.current_shape_inds_vec[k][:self.list_of_shape_inds_lenths[index]] for k,index in enumerate(self.sampled_indices)]
        
        # Simple weight generation without complex graph construction
        self.ghn(self.current_model, shape_ind=self.sampled_shape_inds, return_embeddings=False)
        
        # Enable gradients for current architecture std parameters
        for i in self.sampled_indices:
            self.log_std[i].requires_grad = True


    def change_graph(self, repeat_sample=False):
        ''' Generate weights using GHN-2 with uniform sampling '''
        if not repeat_sample or self.sampled_indices is None:
            # Clear gradients of previous std parameters
            if self.sampled_indices is not None:
                for i in self.sampled_indices:
                    self.log_std[i].requires_grad = False
                    self.log_std[i].grad = None
            
            # Uniformly sample new architecture indices
            self.sample_arc_indices()
            
            # Get current models and configs
            self.current_model = [self.all_models[i] for i in self.sampled_indices]
            self.current_configs = [self.architecture_configs[i] for i in self.sampled_indices]
            self.current_arch_descriptors = self.arch_descriptors[self.sampled_indices]
            
            # Create detailed shape indices like original HyperPPO
            self.current_shape_inds_vec = [self.list_of_shape_inds[index] for index in self.sampled_indices]
            self.list_of_sampled_shape_inds = [self.current_shape_inds_vec[k][:self.list_of_shape_inds_lenths[index]] for k,index in enumerate(self.sampled_indices)]
            
            # Enable gradients for current architecture std parameters
            for i in self.sampled_indices:
                self.log_std[i].requires_grad = True 
        else:
            # Create shape indices for repeat sampling case too
            self.current_shape_inds_vec = [self.list_of_shape_inds[index] for index in self.sampled_indices]
            self.list_of_sampled_shape_inds = [self.current_shape_inds_vec[k][:self.list_of_shape_inds_lenths[index]] for k,index in enumerate(self.sampled_indices)]

        # Generate weights using detailed shape indices (like original HyperPPO)
        # Single GPU weight generation with detailed shape indices
        self.sampled_shape_inds = torch.cat(self.list_of_sampled_shape_inds).view(-1,1)
        _, embeddings = self.ghn(self.current_model, shape_ind=self.sampled_shape_inds, return_embeddings=True)


    def forward(self, obs, state_obs=None, track=True):
        ''' Do a forward pass through the current CNN+MLP models with parallel state processing. 
            obs: image observations [batch_size, channels, height, width]
            state_obs: state observations [batch_size, state_dim] (optional for parallel processing)
            track: if True, we track the architecture descriptors and indices
        '''
        batch_per_net = int(obs.shape[0] // len(self.current_model))

        if track:
            # Track architecture descriptors per state
            self.arch_descriptors_per_state = torch.cat([
                self.current_arch_descriptors[i].repeat(batch_per_net, 1) 
                for i in range(len(self.current_model))
            ])
            self.sampled_indices_per_state_dim = torch.cat([
                torch.tensor([self.sampled_indices[i]]).repeat(batch_per_net) 
                for i in range(len(self.current_model))
            ])
            
        # Single GPU forward pass with parallel state processing
        if state_obs is not None:
            # Pass both RGB and state to each model
            x = torch.cat(parallel_apply(self.current_model, 
                [(obs[i*batch_per_net:(i+1)*batch_per_net],
                  state_obs[i*batch_per_net:(i+1)*batch_per_net])
                 for i in range(len(self.current_model))]))
        else:
            # RGB only (backward compatibility)
            x = torch.cat(parallel_apply(self.current_model, 
                [obs[i*batch_per_net:(i+1)*batch_per_net] 
                 for i in range(len(self.current_model))]))
        
        mu = x
        action_logstd = self.get_logstd(obs, mu, batch_per_net)

        return mu, action_logstd    

    def get_logstd(self, state, mu, batch_per_net):
        ''' Get multi log_std for current architectures '''
        # Debug: Check log_std parameters before expanding
        for i, idx in enumerate(self.sampled_indices):
            if torch.any(torch.isnan(self.log_std[idx])):
                # print(f"⚠️ NaN in log_std[{idx}]: {torch.sum(torch.isnan(self.log_std[idx]))} NaN values")
                # print(f"⚠️ log_std[{idx}] values: {self.log_std[idx]}")
                # Reset to safe value if NaN detected
                self.log_std[idx].data = torch.full_like(self.log_std[idx], -0.5)
                # print(f"⚠️ Reset log_std[{idx}] to -0.5")
        
        log_std_expanded = torch.cat([self.log_std[i].expand(batch_per_net, self.act_dim) for i in self.sampled_indices])
        # # Clamp log_std to prevent extreme values and NaN - COMMENTED OUT to match original HyperPPO
        # log_std_clamped = torch.clamp(log_std_expanded, min=-20, max=2)
        # return log_std_clamped
        return log_std_expanded  # Return raw log_std like original HyperPPO



    ############################################################### forward helper functions, mostly only for debugging purposes ######################################################
    def sample(self, obs, state_obs=None, epsilon=1e-6):
        mu, log_std = self.forward(obs, state_obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(obs.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)
        
        return action, log_prob, torch.tanh(mu)
    

    def get_action(self, obs, state_obs=None):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(obs, state_obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(obs.device)
        action = torch.tanh(e)
        return action.detach().cpu()
    
    def get_det_action(self, obs, state_obs=None):
        mu, log_std = self.forward(obs, state_obs)
        return torch.tanh(mu).detach().cpu()


    def get_logprob(self, obs, actions, state_obs=None, epsilon=1e-6):
        mu, log_std = self.forward(obs, state_obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(1, keepdim=True)
        return log_prob