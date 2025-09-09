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

from src.model import CnnMlpNetwork
from src.ghn_modules import MLP_GHN
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
                ):
        super().__init__()

        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.architecture_config_path = architecture_config_path
        self.meta_batch_size = meta_batch_size
        self.multi_gpu = multi_gpu
        self.std_mode = 'multi'  # Only use multi std mode

        
        # initialize all devices for parallelization on multiple GPUs
        self._initialize_devices(device)

        # load architecture configurations from JSON
        self._load_architecture_configs()

        # initialize all list of CNN+MLP architectures
        self._initialize_cnn_mlp_architectures()

        # initialize uniform sampling data
        self._initialize_uniform_sampling()

        # initialize the original MLP_GHN
        self._initialize_ghn()

        # initialize standard deviation vectors
        self._initialize_std()
        
        # Sample initial architectures and generate initial weights
        self._sample_initial_architectures()

    def _initialize_std(self):
        ''' Initialize multi standard deviation vectors for each architecture '''
        # Initialize with small negative values for stability (similar to PPO)
        init_log_std = 0.0  # exp(-0.5) ≈ 0.6 std
        self.log_std = nn.ParameterList([
            nn.Parameter(torch.full((1, np.prod(self.act_dim)), init_log_std), requires_grad=False)
            for index in self.list_of_arc_indices
        ])  


    def _initialize_uniform_sampling(self):
        ''' Initialize uniform sampling data '''
        self.sampled_indices = None
        
        # Sample initial architectures after initialization is complete

    def _sample_initial_architectures(self):
        ''' Sample initial architectures and generate initial weights '''
        # Sample initial architecture indices
        self.sample_arc_indices()
        
        # Get current models and configs
        self.current_model = [self.all_models[i] for i in self.sampled_indices]
        self.current_configs = [self.architecture_configs[i] for i in self.sampled_indices]
        self.current_arch_descriptors = self.arch_descriptors[self.sampled_indices]
        
        # Create detailed shape indices like original HyperPPO
        self.current_shape_inds_vec = [self.list_of_shape_inds[index] for index in self.sampled_indices]
        self.list_of_sampled_shape_inds = [self.current_shape_inds_vec[k][:self.list_of_shape_inds_lenths[index]] for k,index in enumerate(self.sampled_indices)]
        
        # Generate initial weights using detailed shape indices
        self.sampled_shape_inds = torch.cat(self.list_of_sampled_shape_inds).view(-1,1)
        _, embeddings = self.ghn(self.current_model, shape_ind=self.sampled_shape_inds, return_embeddings=True)
        
        # Enable gradients for current architecture std parameters
        for i in self.sampled_indices:
            self.log_std[i].requires_grad = True
        
        print(f"Sampled initial architectures: {self.sampled_indices}")





    def _load_architecture_configs(self):
        """Load architecture configurations from JSON file"""
        with open(self.architecture_config_path, 'r') as f:
            self.arch_data = json.load(f)
        
        self.metadata = self.arch_data['metadata']
        self.architecture_configs = self.arch_data['architectures']
        self.teacher_config = self.arch_data.get('teacher_model', None)
        
        print(f"Loaded {len(self.architecture_configs)} architectures from {self.architecture_config_path}")

    def _initialize_cnn_mlp_architectures(self):
        """Initialize CNN+MLP architectures from loaded configurations"""
        
        # Sort architectures by complexity (parameter count)
        self.architecture_configs.sort(key=lambda x: self._get_cnn_mlp_params(x))
        
        # Create list of architecture indices
        self.list_of_arc_indices = np.arange(len(self.architecture_configs))
        
        # Create all CNN+MLP models
        self.all_models = []
        for config in self.architecture_configs:
            model = CnnMlpNetwork(
                cnn_config=config['cnn_config'],
                cnn_mlp_config=config.get('cnn_mlp_config', [256]),
                mlp_config=config['mlp_config'],
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
        
        print(f"Initialized {len(self.all_models)} CNN+MLP models")


    def _initialize_arch_descriptors(self):
        """Initialize architecture descriptors for GHN-2"""
        
        self.arch_descriptors = []
        for config in self.architecture_configs:
            descriptor = torch.tensor(config['arch_descriptor'], dtype=torch.float32, device=self.device)
            self.arch_descriptors.append(descriptor)
        
        # Convert to tensor
        self.arch_descriptors = torch.stack(self.arch_descriptors)
        self.arch_max_len = self.metadata['arch_descriptor_config']['total_length']
        
        print(f"Initialized architecture descriptors: {self.arch_descriptors.shape}")

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
        
        print(f"Built component vocabulary: {len(component_vocab)} unique components")
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
        
        print(f"Initialized detailed shape indices: {self.list_of_shape_inds.shape}")


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
        
        print(f"Initialized Simple MLP_GHN for CNN+MLP networks on device: {self.device}")



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
        ''' Uniformly sample architecture indices for the current meta batch '''
        self.sampled_indices = np.random.choice(self.list_of_arc_indices, self.meta_batch_size, replace=False)



    def set_graph(self, indices_vector):
        ''' 
        Explicitly set specific architecture indices and generate weights.
        Used for deterministic architecture selection (not random sampling).
        '''

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
        ''' 
        Generate weights using GHN-2 with uniform sampling
        
        Architecture resampling timing (corrected to match original HyperPPO):
        - repeat_sample=False: Sample NEW architectures (called AFTER each iteration completes)
        - repeat_sample=True: Regenerate weights for SAME architectures (called BEFORE each minibatch)
        '''
        if not repeat_sample:
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

        # Generate weights using detailed shape indices (like original HyperPPO)
        # Single GPU weight generation with detailed shape indices
        self.sampled_shape_inds = torch.cat(self.list_of_sampled_shape_inds).view(-1,1)
        _, embeddings = self.ghn(self.current_model, shape_ind=self.sampled_shape_inds, return_embeddings=True)


    def forward(self, state, track=True):
        ''' Do a forward pass through the current CNN+MLP models. 
            state: image observations [batch_size, channels, height, width]
            track: if True, we track the architecture descriptors and indices
        '''
        batch_per_net = int(state.shape[0] // len(self.current_model))

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
            
        # Single GPU forward pass
        x = torch.cat(parallel_apply(self.current_model, 
            [state[i*batch_per_net:(i+1)*batch_per_net] 
             for i in range(len(self.current_model))]))
        
        mu = x
        action_logstd = self.get_logstd(state, mu, batch_per_net)

        return mu, action_logstd    

    def get_logstd(self, state, mu, batch_per_net):
        ''' Get multi log_std for current architectures '''
        # Debug: Check log_std parameters before expanding
        for i, idx in enumerate(self.sampled_indices):
            if torch.any(torch.isnan(self.log_std[idx])):
                print(f"Warning: NaN in log_std[{idx}]: {torch.sum(torch.isnan(self.log_std[idx]))} NaN values")
                print(f"log_std[{idx}] values: {self.log_std[idx]}")
                # Reset to safe value if NaN detected
                self.log_std[idx].data = torch.full_like(self.log_std[idx], -0.5)
                print(f"Reset log_std[{idx}] to -0.5")
        
        log_std_expanded = torch.cat([self.log_std[i].expand(batch_per_net, self.act_dim) for i in self.sampled_indices])
        # # Clamp log_std to prevent extreme values and NaN - COMMENTED OUT to match original HyperPPO
        # log_std_clamped = torch.clamp(log_std_expanded, min=-20, max=2)
        # return log_std_clamped
        return log_std_expanded  # Return raw log_std like original HyperPPO



    ############################################################### forward helper functions, mostly only for debugging purposes ######################################################
    def sample(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)
        
        return action, log_prob, torch.tanh(mu)
    

    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        return action.detach().cpu()
    
    def get_det_action(self, state):
        mu, log_std = self.forward(state)
        return torch.tanh(mu).detach().cpu()


    def get_logprob(self,obs, actions, epsilon=1e-6):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(1, keepdim=True)
        return log_prob