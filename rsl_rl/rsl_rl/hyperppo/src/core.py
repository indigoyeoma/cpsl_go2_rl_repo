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

from rsl_rl.hyperppo.src.model import CnnMlpNetwork
from rsl_rl.hyperppo.src.ghn_modules import MLP_GHN

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
        self.std_mode = 'multi'
        self.architecture_sampling_mode = architecture_sampling_mode

        self._initialize_devices(device)

        self._load_architecture_configs()

        self._initialize_cnn_mlp_architectures()

        self._initialize_architecture_sampling_data()

        self._initialize_ghn()

        self._initialize_std()
        
        # Initialize current model tracking attributes
        self.current_model = None
        self.current_configs = None 
        self.current_arch_descriptors = None
        self.sampled_indices = None
        self.list_of_sampled_shape_inds = []

    def _initialize_std(self):
        ''' Initialize multi standard deviation vectors for each architecture '''
        init_log_std = 0.0  
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
            
        elif self.architecture_sampling_mode == "biased":
            self.arch_sampling_probs = []
            
            arch_complexities = []
            for config in self.architecture_configs:
                num_cnn_layers = len(config['cnn_config'])
                num_mlp_layers = len(config['mlp_config'])
                total_layers = num_cnn_layers + num_mlp_layers
                arch_complexities.append(total_layers)
            
            num_unique_complexities = len(set(arch_complexities))
            for i in self.list_of_arc_indices:
                complexity = arch_complexities[i]
                num_archs_with_same_complexity = arch_complexities.count(complexity)
                prob = 1.0 / num_archs_with_same_complexity
                self.arch_sampling_probs.append(prob)
            
            self.arch_sampling_probs = np.array(self.arch_sampling_probs)
            self.arch_sampling_probs = self.arch_sampling_probs / np.sum(self.arch_sampling_probs)
            
        
        else:
            raise ValueError(f"Unsupported architecture_sampling_mode: {self.architecture_sampling_mode}. Use 'uniform' or 'biased'.")


    def _load_architecture_configs(self):
        """Load architecture configurations from JSON file"""
        with open(self.architecture_config_path, 'r') as f:
            self.arch_data = json.load(f)
        
        self.metadata = self.arch_data['metadata']
        self.architecture_configs = self.arch_data['architectures']
        self.teacher_config = self.arch_data.get('teacher_model', None)
       
    def _initialize_cnn_mlp_architectures(self):
        """Initialize CNN+MLP architectures from loaded configurations"""
        
        self.architecture_configs.sort(key=lambda x: self._get_cnn_mlp_params(x))
        self.list_of_arc_indices = np.arange(len(self.architecture_configs))
        self.state_dim = self.metadata['state_dim']
        self.all_models = []
        for config in self.architecture_configs:
            model = CnnMlpNetwork(
                cnn_config=config['cnn_config'],
                cnn_mlp_config=config['cnn_mlp_config'],
                state_mlp_config=config['state_mlp_config'], 
                mlp_config=config['mlp_config'],
                state_dim=self.state_dim,
                input_channels=self.metadata['input_channels'],
                output_dim=self.act_dim,
                input_size=self.metadata['input_size']
            )
            self.all_models.append(model)
        
        self._initialize_arch_descriptors()
        self._initialize_shape_inds()
        np.random.shuffle(self.list_of_arc_indices)
        

    def _initialize_arch_descriptors(self):
        """Initialize architecture descriptors for GHN-2"""
        
        self.arch_descriptors = []
        for config in self.architecture_configs:
            descriptor = torch.tensor(config['arch_descriptor'], dtype=torch.float32, device=self.device)
            self.arch_descriptors.append(descriptor)
        
        self.arch_descriptors = torch.stack(self.arch_descriptors)
        self.arch_max_len = self.metadata['arch_descriptor_config']['total_length']
        
    def _initialize_shape_inds(self):
        """Initialize detailed shape indices for CNN+MLP architectures (like original HyperPPO)"""
        
        component_vocab = {}
        component_index = 0
        
        for config in self.architecture_configs:
            for layer in config['cnn_config']:
                desc = f"{layer['channels']}ch{layer['kernel']}ker"
                if desc not in component_vocab:
                    component_vocab[desc] = component_index
                    component_index += 1
            
            if 'cnn_mlp_config' in config and config['cnn_mlp_config']:
                for units in config['cnn_mlp_config']:
                    desc = f"{units}mlp"
                    if desc not in component_vocab:
                        component_vocab[desc] = component_index
                        component_index += 1
            
            for units in config['mlp_config']:
                desc = f"{units}mlp"
                if desc not in component_vocab:
                    component_vocab[desc] = component_index
                    component_index += 1
        output_desc = f"{self.act_dim}out"
        component_vocab[output_desc] = component_index
        
        self.component_vocab = component_vocab
        
        self.list_of_shape_inds = []
        
        for config in self.architecture_configs:
            shape_ind = [torch.tensor(0.0).to(self.device)]  
            
            for layer in config['cnn_config']:
                desc = f"{layer['channels']}ch{layer['kernel']}ker"
                idx = self.component_vocab[desc]
                shape_ind.append(torch.tensor(float(idx)).to(self.device))
                shape_ind.append(torch.tensor(float(idx)).to(self.device))  # Double like original
            
            if 'cnn_mlp_config' in config and config['cnn_mlp_config']:
                for units in config['cnn_mlp_config']:
                    desc = f"{units}mlp"
                    idx = self.component_vocab[desc]
                    shape_ind.append(torch.tensor(float(idx)).to(self.device))
                    shape_ind.append(torch.tensor(float(idx)).to(self.device))
            
            for units in config['mlp_config']:
                desc = f"{units}mlp"
                idx = self.component_vocab[desc]
                shape_ind.append(torch.tensor(float(idx)).to(self.device))
                shape_ind.append(torch.tensor(float(idx)).to(self.device))
            
            output_idx = self.component_vocab[output_desc]
            shape_ind.append(torch.tensor(float(output_idx)).to(self.device))
            shape_ind.append(torch.tensor(float(output_idx)).to(self.device))
            
            shape_ind = torch.stack(shape_ind).view(-1, 1)
            self.list_of_shape_inds.append(shape_ind)
        
        self.list_of_shape_inds_lenths = [x.squeeze().numel() for x in self.list_of_shape_inds]
        self.shape_inds_max_len = max(self.list_of_shape_inds_lenths)
        
        for i in range(len(self.list_of_shape_inds)):
            num_pad = self.shape_inds_max_len - self.list_of_shape_inds[i].shape[0]
            if num_pad > 0:
                pad_tensor = torch.tensor(-1.0).to(self.device).repeat(num_pad, 1)
                self.list_of_shape_inds[i] = torch.cat([self.list_of_shape_inds[i], pad_tensor], 0)
        
        self.list_of_shape_inds = torch.stack(self.list_of_shape_inds)
        self.list_of_shape_inds = self.list_of_shape_inds.reshape(len(self.list_of_shape_inds), self.shape_inds_max_len)
        

    def _initialize_devices(self, device):
        ''' Initialize device for single GPU usage '''
        self.device = device


    def _initialize_ghn(self):
        """Initialize simple MLP_GHN for weight generation (HyperPPO style)"""
        if 'ghn_config' in self.metadata and 'ghn_max_shape' in self.metadata['ghn_config']:
            max_shape = tuple(self.metadata['ghn_config']['ghn_max_shape'])
        else:
            raise ValueError("Required 'ghn_max_shape' configuration missing from metadata")
        
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
        


    def _get_cnn_mlp_params(self, config):
        """Get the number of parameters in a CNN+MLP architecture"""
        total_params = 0
        prev_channels = self.metadata['input_channels']
        for layer in config['cnn_config']:
            channels = layer['channels']
            kernel = layer['kernel']
            conv_params = (prev_channels * kernel * kernel + 1) * channels
            total_params += conv_params
            prev_channels = channels
        
        if 'cnn_mlp_config' in config and config['cnn_mlp_config']:
            input_size = self.metadata['input_size']
            for layer in config['cnn_config']:
                input_size = (input_size + 2 * layer['padding'] - layer['kernel']) // layer['stride'] + 1
            
            cnn_output_size = prev_channels * input_size * input_size
            cnn_mlp_out = config['cnn_mlp_config'][0]
            total_params += (cnn_output_size + 1) * cnn_mlp_out
        
        prev_size = config['mlp_config'][0]  # First MLP dimension
        for i in range(1, len(config['mlp_config'])):
            current_size = config['mlp_config'][i]
            total_params += (prev_size + 1) * current_size
            prev_size = current_size
        
        total_params += (prev_size + 1) * self.act_dim
        
        return total_params            

    def sample_arc_indices(self):
        ''' Sample architecture indices for the current meta batch based on sampling mode '''
        if self.architecture_sampling_mode == "uniform":
            self.sampled_indices = np.random.choice(self.list_of_arc_indices, self.meta_batch_size, replace=False)
        elif self.architecture_sampling_mode == "biased":
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

        if self.sampled_indices is not None:
            for i in self.sampled_indices:
                self.log_std[i].requires_grad = False
                self.log_std[i].grad = None
        
        self.sampled_indices = indices_vector
        self.current_model = [self.all_models[i] for i in self.sampled_indices]
        self.current_configs = [self.architecture_configs[i] for i in self.sampled_indices]
        self.current_arch_descriptors = self.arch_descriptors[self.sampled_indices]
        
        self.current_shape_inds_vec = [self.list_of_shape_inds[index] for index in self.sampled_indices]
        self.list_of_sampled_shape_inds = [self.current_shape_inds_vec[k][:self.list_of_shape_inds_lenths[index]] for k,index in enumerate(self.sampled_indices)]
        

        self.sampled_shape_inds = torch.cat(self.list_of_sampled_shape_inds).view(-1,1)
        self.ghn(self.current_model, shape_ind=self.sampled_shape_inds, return_embeddings=False)
        
        for i in self.sampled_indices:
            self.log_std[i].requires_grad = True


    def set_specific_architecture(self, architecture_id):
        ''' Set a specific single architecture for evaluation '''
        if architecture_id < 0 or architecture_id >= len(self.architecture_configs):
            raise ValueError(f"Architecture ID {architecture_id} is out of range. Available: 0-{len(self.architecture_configs)-1}")
        self.set_graph([architecture_id])

    def change_graph(self, repeat_sample=False):
        ''' Generate weights using GHN-2 with uniform sampling '''
        if not repeat_sample:
            if self.sampled_indices is not None:
                for i in self.sampled_indices:
                    self.log_std[i].requires_grad = False
                    self.log_std[i].grad = None
            
            self.sample_arc_indices()
            
            self.current_model = [self.all_models[i] for i in self.sampled_indices]
            self.current_configs = [self.architecture_configs[i] for i in self.sampled_indices]
            self.current_arch_descriptors = self.arch_descriptors[self.sampled_indices]
            
            self.current_shape_inds_vec = [self.list_of_shape_inds[index] for index in self.sampled_indices]
            self.list_of_sampled_shape_inds = [self.current_shape_inds_vec[k][:self.list_of_shape_inds_lenths[index]] for k,index in enumerate(self.sampled_indices)]
        
            for i in self.sampled_indices:
                self.log_std[i].requires_grad = True 

        self.sampled_shape_inds = torch.cat(self.list_of_sampled_shape_inds).view(-1,1)
        _, embeddings = self.ghn(self.current_model, shape_ind=self.sampled_shape_inds, return_embeddings=True)


    def forward(self, obs, state_obs=None, track=True):
        ''' Do a forward pass through the current CNN+MLP models with parallel state processing. 
            obs: image observations [batch_size, channels, height, width]
            state_obs: state observations [batch_size, state_dim] (optional for parallel processing)
            track: if True, we track the architecture descriptors and indices
        '''
        # Ensure architectures are sampled before forward pass
        if self.current_model is None:
            # Initialize with first architecture
            self.change_graph(repeat_sample=False)
        
        batch_per_net = int(obs.shape[0] // len(self.current_model))

        if track:
            self.arch_descriptors_per_state = torch.cat([
                self.current_arch_descriptors[i].repeat(batch_per_net, 1) 
                for i in range(len(self.current_model))
            ])
            self.sampled_indices_per_state_dim = torch.cat([
                torch.tensor([self.sampled_indices[i]]).repeat(batch_per_net) 
                for i in range(len(self.current_model))
            ])
            
        if state_obs is not None:
            x = torch.cat(parallel_apply(self.current_model, 
                [(obs[i*batch_per_net:(i+1)*batch_per_net],
                  state_obs[i*batch_per_net:(i+1)*batch_per_net])
                 for i in range(len(self.current_model))]))
        else:
            x = torch.cat(parallel_apply(self.current_model, 
                [obs[i*batch_per_net:(i+1)*batch_per_net] 
                 for i in range(len(self.current_model))]))
        
        mu = x
        action_logstd = self.get_logstd(obs, mu, batch_per_net)

        return mu, action_logstd    

    def get_logstd(self, state, mu, batch_per_net):
        ''' Get multi log_std for current architectures '''
        for i, idx in enumerate(self.sampled_indices):
            if torch.any(torch.isnan(self.log_std[idx])):
                self.log_std[idx].data = torch.full_like(self.log_std[idx], -0.5)
        
        log_std_expanded = torch.cat([self.log_std[i].expand(batch_per_net, self.act_dim) for i in self.sampled_indices])
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
        return action.detach()  # Keep on GPU for training efficiency
    
    def get_det_action(self, obs, state_obs=None):
        mu, log_std = self.forward(obs, state_obs)
        return torch.tanh(mu).detach()  # Keep on GPU for training efficiency


    def get_logprob(self, obs, actions, state_obs=None, epsilon=1e-6):
        mu, log_std = self.forward(obs, state_obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(1, keepdim=True)
        return log_prob