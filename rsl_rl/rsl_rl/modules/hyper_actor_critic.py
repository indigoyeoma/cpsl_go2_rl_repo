import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import json
import os
from typing import Optional, Tuple

# Import from existing HyperPPO modules
from rsl_rl.hyperppo.src.ghn_modules import MLP_GHN, get_activation
from rsl_rl.hyperppo.src.model import CnnMlpNetwork
from rsl_rl.hyperppo.src.core import hyperActor

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer weights with orthogonal initialization"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ArchConditionedCritic(nn.Module):
    """Architecture-conditioned critic network with Depth + State + Architecture inputs (asymmetric design)"""
    
    def __init__(self, state_dim, arch_descriptor_dim, device, input_size=84):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.input_size = input_size
        
        # Process depth observations with CNN (privileged info for critic)
        depth_features_dim = 256
        self.depth_encoder = nn.Sequential(
            # Input: [batch, 1, 84, 84] - single channel depth
            layer_init(nn.Conv2d(1, 32, 8, stride=4, padding=0)),  # → [batch, 32, 20, 20]
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2, padding=1)), # → [batch, 64, 10, 10] 
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, 4, stride=2, padding=1)), # → [batch, 128, 5, 5]
            nn.ReLU(),
            layer_init(nn.Conv2d(128, 256, 3, stride=1, padding=1)), # → [batch, 256, 5, 5]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),                           # → [batch, 256, 2, 2]
            nn.Flatten(),                                           # → [batch, 1024]
            layer_init(nn.Linear(1024, depth_features_dim))         # → [batch, 256]
        )
        
        # Process state information (privileged info for critic)
        state_features_dim = 128
        self.state_encoder = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, state_features_dim))
        )
        
        # Process architecture descriptor
        arch_embedding_dim = 128
        self.arch_embedding = nn.Sequential(
            layer_init(nn.Linear(arch_descriptor_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, arch_embedding_dim))
        )
        
        # Combine all features: Depth + State + Architecture
        combined_dim = depth_features_dim + state_features_dim + arch_embedding_dim  # 256 + 128 + 128 = 512
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(combined_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0)
        )
        
    def forward(self, obs_dict, arch_descriptors):
        """
        Forward pass with depth + state + architecture
        
        Args:
            obs_dict: Dict with 'depth' and 'state' keys
            arch_descriptors: Architecture descriptors [batch, arch_descriptor_dim]
        """
        # Extract depth and state from observation dictionary
        depth_obs = obs_dict["depth"]  # [batch, 1, 84, 84]
        state_obs = obs_dict["state"]  # [batch, state_dim]
        
        # Process all inputs separately
        depth_features = self.depth_encoder(depth_obs)  # [batch, 256]
        state_features = self.state_encoder(state_obs)  # [batch, 128] 
        arch_embedding = self.arch_embedding(arch_descriptors)  # [batch, 128]
        
        # Combine all privileged information
        combined = torch.cat([depth_features, state_features, arch_embedding], dim=1)  # [batch, 512]
        return self.critic(combined).squeeze(-1)


class HyperActorWrapper(nn.Module):
    """
    HyperPPO Actor that generates weights for different CNN+MLP architectures
    Based on the loaded hyper modules for complete GHN implementation
    """
    def __init__(self, 
                 act_dim, 
                 obs_dim, 
                 architecture_config_path: Optional[str] = None,
                 meta_batch_size: int = 2,
                 device: str = "cuda",
                 multi_gpu: bool = False,
                 input_channels: int = 1,  # For depth images
                 input_size: int = 64,     # 64x64 depth images
                 use_teacher_distillation: bool = True,
                 teacher_update_freq: int = 100,
                 distillation_weight: float = 0.1,
                 **kwargs):
        """
        HyperPPO Actor for GO2 robot with depth camera
        
        Args:
            act_dim: Action dimension (12 for GO2 joints)
            obs_dim: Observation dimension (48 base + 4096 depth = 4144)
            architecture_config_path: Path to architecture JSON config
            meta_batch_size: Number of architectures to sample per iteration
            device: Device for computation
            multi_gpu: Whether to use multiple GPUs
            input_channels: Number of input channels (1 for depth)
            input_size: Input image size (64 for 64x64 depth)
            use_teacher_distillation: Whether to use teacher distillation
            teacher_update_freq: How often to update teacher network
            distillation_weight: Weight for distillation loss
        """
        if kwargs:
            # Unexpected arguments will be ignored
            pass
        
        super(HyperActor, self).__init__()
        
        self.act_dim = act_dim
        self.obs_dim = obs_dim  # Total obs including depth
        self.base_obs_dim = 48  # Base proprioceptive observations
        self.depth_obs_dim = obs_dim - self.base_obs_dim  # Depth observations (4096)
        self.architecture_config_path = architecture_config_path
        self.meta_batch_size = meta_batch_size
        self.multi_gpu = multi_gpu
        self.input_channels = input_channels
        self.input_size = input_size
        self.std_mode = 'multi'  # Use multi std mode for different architectures
        
        # Teacher distillation parameters
        self.use_teacher_distillation = use_teacher_distillation
        self.teacher_update_freq = teacher_update_freq
        self.distillation_weight = distillation_weight
        self.update_count = 0
        
        # Initialize all devices for parallelization
        self._initialize_devices(device)
        
        # Load architecture configurations from JSON
        self._load_architecture_configs()
        
        # Initialize all CNN+MLP architectures
        self._initialize_cnn_mlp_architectures()
        
        # Initialize data required for architecture sampling
        self._initialize_architecture_sampling_data()
        
        # Initialize the GHN
        self._initialize_ghn()
        
        # Initialize standard deviation vectors for each architecture
        self._initialize_std()
        
        # Initialize teacher network for distillation
        if self.use_teacher_distillation:
            self._initialize_teacher_network()
        
        # Current sampling state
        self.sampled_indices = None
        self.current_models = None
        self.list_of_sampled_shape_inds = []
        
        # HyperActor initialized
    
    def _initialize_devices(self, device):
        """Initialize device configuration"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.multi_gpu = self.multi_gpu and torch.cuda.device_count() > 1
        # Multi-GPU configuration set if available
    
    def _load_architecture_configs(self):
        """Load architecture configurations from JSON file"""
        if self.architecture_config_path and os.path.exists(self.architecture_config_path):
            with open(self.architecture_config_path, 'r') as f:
                config_data = json.load(f)
            self.architecture_configs = config_data.get('architectures', [])
            self.metadata = config_data.get('metadata', {})
            # Loaded architecture configurations from file
        else:
            # Default GO2 architectures for depth + proprioception
            self.architecture_configs = self._get_default_go2_architectures()
            self.metadata = self._get_default_metadata()
            # Using default GO2 architectures
    
    def _get_default_go2_architectures(self):
        """Default architecture configurations for GO2 with depth camera"""
        return [
            {
                "id": 0,
                "name": "lightweight_cnn",
                "cnn_config": [
                    {"channels": 16, "kernel": 5, "stride": 4, "padding": 2},
                    {"channels": 32, "kernel": 3, "stride": 2, "padding": 1}
                ],
                "cnn_mlp_config": [128],
                "mlp_config": [128, self.act_dim * 2],  # mu + logstd
                "arch_descriptor": [16,5,32,3,128,128,self.act_dim*2,0,0,0,0,0,0,0,0,0]
            },
            {
                "id": 1,
                "name": "standard_cnn",
                "cnn_config": [
                    {"channels": 32, "kernel": 5, "stride": 4, "padding": 2},
                    {"channels": 64, "kernel": 3, "stride": 2, "padding": 1}
                ],
                "cnn_mlp_config": [256],
                "mlp_config": [256, self.act_dim * 2],
                "arch_descriptor": [32,5,64,3,256,256,self.act_dim*2,0,0,0,0,0,0,0,0,0]
            },
            {
                "id": 2,
                "name": "deep_cnn",
                "cnn_config": [
                    {"channels": 32, "kernel": 3, "stride": 2, "padding": 1},
                    {"channels": 64, "kernel": 3, "stride": 2, "padding": 1},
                    {"channels": 128, "kernel": 3, "stride": 2, "padding": 1}
                ],
                "cnn_mlp_config": [512],
                "mlp_config": [512, 256, self.act_dim * 2],
                "arch_descriptor": [32,3,64,3,128,3,512,512,256,self.act_dim*2,0,0,0,0,0,0]
            },
            {
                "id": 3,
                "name": "wide_cnn",
                "cnn_config": [
                    {"channels": 64, "kernel": 5, "stride": 2, "padding": 2},
                    {"channels": 128, "kernel": 3, "stride": 2, "padding": 1}
                ],
                "cnn_mlp_config": [256],
                "mlp_config": [256, 128, self.act_dim * 2],
                "arch_descriptor": [64,5,128,3,256,256,128,self.act_dim*2,0,0,0,0,0,0,0,0]
            },
            {
                "id": 4,
                "name": "compact_cnn",
                "cnn_config": [
                    {"channels": 24, "kernel": 5, "stride": 2, "padding": 2},
                    {"channels": 48, "kernel": 3, "stride": 2, "padding": 1}
                ],
                "cnn_mlp_config": [128],
                "mlp_config": [128, 64, self.act_dim * 2],
                "arch_descriptor": [24,5,48,3,128,128,64,self.act_dim*2,0,0,0,0,0,0,0,0]
            }
        ]
    
    def _get_default_metadata(self):
        """Default metadata for GO2 architectures"""
        return {
            "input_channels": self.input_channels,
            "input_size": self.input_size,
            "output_dim": self.act_dim * 2,  # mu + logstd
            "ghn_config": {
                "ghn_max_shape": [512, 512, 5, 5]
            },
            "arch_descriptor_config": {
                "total_length": 16,
                "arch_descriptor_dim": 16,
                "description": "Format: [ch1,k1,ch2,k2,ch3,k3,cnn_mlp,mlp1,mlp2,mlp3,...]"
            }
        }
    
    def _initialize_cnn_mlp_architectures(self):
        """Initialize all CNN+MLP architecture models"""
        self.all_models = []
        self.list_of_arcs = []
        self.list_of_arc_indices = []
        
        for i, arch_config in enumerate(self.architecture_configs):
            # Create CNN+MLP network using the loaded model
            model = CnnMlpNetwork(
                cnn_config=arch_config['cnn_config'],
                cnn_mlp_config=arch_config['cnn_mlp_config'],
                mlp_config=arch_config['mlp_config'],
                input_channels=self.input_channels,
                output_dim=self.act_dim * 2,  # mu + logstd for continuous actions
                input_size=self.input_size
            ).to(self.device)
            
            self.all_models.append(model)
            self.list_of_arcs.append(arch_config['arch_descriptor'])
            self.list_of_arc_indices.append(i)
        
        # CNN+MLP architectures created
    
    def _initialize_architecture_sampling_data(self):
        """Initialize architecture sampling probabilities and data"""
        # Use uniform sampling for simplicity
        self.arch_sampling_probs = np.ones(len(self.list_of_arc_indices)) / len(self.list_of_arc_indices)
        self.sampled_indices = None
        # Using uniform sampling across architectures
    
    def _initialize_ghn(self):
        """Initialize the Graph HyperNetwork"""
        max_shape = self.metadata.get('ghn_config', {}).get('ghn_max_shape', [512, 512, 5, 5])
        num_classes = self.act_dim * 2  # mu + logstd
        
        self.ghn = MLP_GHN(
            max_shape=max_shape,
            num_classes=num_classes,
            num_observations=self.obs_dim,
            hypernet='mlp',  # Use MLP hypernet for simplicity
            decoder='mlp',   # Use MLP decoder
            device=self.device
        ).to(self.device)
        
        # GHN initialized
    
    def _initialize_std(self):
        """Initialize standard deviation parameters for each architecture"""
        if self.std_mode == 'multi':
            self.log_std = nn.ParameterList([
                nn.Parameter(torch.zeros(1, self.act_dim), requires_grad=True)
                for _ in self.list_of_arc_indices
            ])
        else:
            self.log_std = nn.Parameter(torch.zeros(1, self.act_dim))
        
        # Initialized std parameters for each architecture
    
    def _initialize_teacher_network(self):
        """Initialize teacher network for distillation"""
        # Teacher network is a fixed CNN+MLP architecture that provides stable guidance
        # Use the first (lightweight) architecture as teacher
        teacher_config = self.architecture_configs[0]
        
        self.teacher_network = CnnMlpNetwork(
            cnn_config=teacher_config['cnn_config'],
            cnn_mlp_config=teacher_config['cnn_mlp_config'],
            mlp_config=teacher_config['mlp_config'],
            input_channels=self.input_channels,
            output_dim=self.act_dim * 2,  # mu + logstd for continuous actions
            input_size=self.input_size
        ).to(self.device)
        
        # Initialize teacher with same weights as the first model
        self.teacher_network.load_state_dict(self.all_models[0].state_dict())
        
        # Teacher std parameters (separate from student)
        self.teacher_log_std = nn.Parameter(torch.zeros(1, self.act_dim), requires_grad=True)
        
        # Teacher network initialized
    
    def change_graph(self, repeat_sample=False):
        """
        CORE GHN TRAINING MECHANISM (following original HyperPPO):
        
        repeat_sample=False: Sample new architectures (AFTER training epoch completes)
        repeat_sample=True: Generate fresh weights for same architectures (BEFORE each minibatch)
        
        This is the KEY to HyperPPO meta-learning:
        - New architectures sampled after each epoch ends
        - Same architectures get fresh weights before each minibatch
        - This trains the GHN to generate good weights for any architecture
        """
        if not repeat_sample:
            # STEP 1: Sample new architectures (called after epoch completes)
            if self.std_mode == 'multi' and self.sampled_indices is not None:
                # Disable gradients for previous architectures' std params
                for i in self.sampled_indices:
                    if i < len(self.log_std):
                        self.log_std[i].requires_grad = False
                        if self.log_std[i].grad is not None:
                            self.log_std[i].grad = None
            
            # Sample new architectures
            self.sample_arc_indices()
            
            # Get the sampled architecture models
            self.current_models = [self.all_models[i] for i in self.sampled_indices]
            
            # Prepare architecture descriptors for GHN
            self._prepare_shape_descriptors()
            
            # Enable gradients for new architectures' std params
            if self.std_mode == 'multi':
                for i in self.sampled_indices:
                    if i < len(self.log_std):
                        self.log_std[i].requires_grad = True
        
        # STEP 2: Generate weights using GHN (happens every call)
        if self.list_of_sampled_shape_inds and self.current_models:
            shape_inds_tensor = torch.cat(self.list_of_sampled_shape_inds).view(-1, 1)
            
            # Generate weights for current models using GHN
            # This is the critical weight generation step
            self.current_models, embeddings = self.ghn(
                self.current_models,
                shape_ind=shape_inds_tensor,
                return_embeddings=True
            )
    
    def sample_arc_indices(self):
        """Sample architecture indices for current meta-batch"""
        # Sample without replacement to ensure diversity
        self.sampled_indices = np.random.choice(
            self.list_of_arc_indices,
            size=min(self.meta_batch_size, len(self.list_of_arc_indices)),
            replace=False,
            p=self.arch_sampling_probs
        )
    
    def _prepare_shape_descriptors(self):
        """Convert architecture configs to GHN input format"""
        self.list_of_sampled_shape_inds = []
        
        for idx in self.sampled_indices:
            arch_descriptor = self.list_of_arcs[idx]
            
            # Convert to tensors
            shape_descriptor = torch.tensor(
                arch_descriptor, 
                dtype=torch.float32, 
                device=self.device
            ).view(-1, 1)
            
            self.list_of_sampled_shape_inds.append(shape_descriptor)
    
    def forward(self, observations, track=True):
        """
        Forward pass through sampled architectures
        
        Args:
            observations: Input observations [batch_size, obs_dim]
            track: Whether this is for tracking (training) or inference
            
        Returns:
            mu: Mean actions [batch_size, act_dim]
            log_std: Log standard deviation [batch_size, act_dim]
        """
        # Initialize architectures on first forward pass (original HyperPPO pattern)
        if self.current_models is None:
            self.change_graph(repeat_sample=False)
        
        batch_size = observations.shape[0]
        
        # Extract depth images from observations
        # observations = [base_obs(48) + depth_flat(4096)]
        base_obs = observations[:, :self.base_obs_dim]  # [batch_size, 48]
        depth_flat = observations[:, self.base_obs_dim:]  # [batch_size, 4096]
        depth_images = depth_flat.view(batch_size, 1, 64, 64)  # [batch_size, 1, 64, 64]
        
        # Forward through all sampled architectures
        all_outputs = []
        for model_idx, model in enumerate(self.current_models):
            # Each model outputs [mu, log_std] concatenated
            output = model(depth_images)  # [batch_size, act_dim * 2]
            all_outputs.append(output)
        
        # Average across architectures
        mean_output = torch.stack(all_outputs).mean(dim=0)  # [batch_size, act_dim * 2]
        
        # Split into mu and log_std
        mu = mean_output[:, :self.act_dim]  # [batch_size, act_dim]
        log_std_output = mean_output[:, self.act_dim:]  # [batch_size, act_dim]
        
        # Use architecture-specific log_std if in multi mode
        if self.std_mode == 'multi' and self.sampled_indices is not None:
            # Average log_std across sampled architectures
            sampled_log_stds = [self.log_std[idx] for idx in self.sampled_indices]
            log_std = torch.stack(sampled_log_stds).mean(dim=0)  # [1, act_dim]
            log_std = log_std.expand(batch_size, -1)  # [batch_size, act_dim]
        else:
            log_std = self.log_std.expand(batch_size, -1)  # [batch_size, act_dim]
        
        return mu, log_std
    
    def teacher_forward(self, observations) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through teacher network
        
        Args:
            observations: Input observations [batch_size, obs_dim]
            
        Returns:
            teacher_mu: Teacher mean actions [batch_size, act_dim]
            teacher_log_std: Teacher log standard deviation [batch_size, act_dim]
        """
        if not self.use_teacher_distillation:
            raise ValueError("Teacher distillation is not enabled")
        
        batch_size = observations.shape[0]
        
        # Extract depth images from observations (same as student)
        depth_flat = observations[:, self.base_obs_dim:]  # [batch_size, 4096]
        depth_images = depth_flat.view(batch_size, 1, 64, 64)  # [batch_size, 1, 64, 64]
        
        # Forward through teacher network
        teacher_output = self.teacher_network(depth_images)  # [batch_size, act_dim * 2]
        
        # Split into mu and log_std
        teacher_mu = teacher_output[:, :self.act_dim]  # [batch_size, act_dim]
        teacher_log_std = self.teacher_log_std.expand(batch_size, -1)  # [batch_size, act_dim]
        
        return teacher_mu, teacher_log_std
    
    def compute_distillation_loss(self, observations) -> torch.Tensor:
        """
        Compute KL(teacher||student) distillation loss
        
        Args:
            observations: Input observations [batch_size, obs_dim]
            
        Returns:
            kl_loss: KL divergence loss KL(teacher||student)
        """
        if not self.use_teacher_distillation:
            return torch.tensor(0.0, device=self.device)
        
        with torch.no_grad():
            # Get teacher distribution (no grad needed)
            teacher_mu, teacher_log_std = self.teacher_forward(observations)
            teacher_std = torch.exp(teacher_log_std)
        
        # Get student distribution
        student_mu, student_log_std = self.forward(observations, track=False)
        student_std = torch.exp(student_log_std)
        
        # Create distributions
        teacher_dist = Normal(teacher_mu.detach(), teacher_std.detach())
        student_dist = Normal(student_mu, student_std)
        
        # Compute KL(teacher||student) = KL(p||q) where p=teacher, q=student
        kl_div = torch.distributions.kl_divergence(teacher_dist, student_dist)
        
        # Sum over action dimensions and average over batch
        kl_loss = kl_div.sum(dim=-1).mean()
        
        return kl_loss
    
    def update_teacher_network(self):
        """Update teacher network periodically with current best performing model"""
        if not self.use_teacher_distillation:
            return
        
        self.update_count += 1
        if self.update_count % self.teacher_update_freq == 0:
            # Update teacher with the first model (could be improved with best performer selection)
            if self.current_models and len(self.current_models) > 0:
                self.teacher_network.load_state_dict(self.current_models[0].state_dict())
                # Teacher network updated
    
    def get_training_metrics(self):
        """Get GHN training metrics for logging (PPO style)"""
        metrics = {}
        
        if self.sampled_indices is not None:
            # Architecture sampling info
            sampled_names = [self.architecture_configs[i]['name'] for i in self.sampled_indices]
            metrics['sampled_architectures'] = sampled_names
            metrics['num_sampled'] = len(self.sampled_indices)
            
            # GHN gradient flow monitoring
            ghn_params_with_grad = [param for param in self.ghn.parameters() if param.grad is not None]
            if ghn_params_with_grad:
                with torch.no_grad():
                    ghn_grad_norm = torch.norm(torch.cat([
                        param.grad.flatten() for param in ghn_params_with_grad
                    ]))
                    metrics['ghn_grad_norm'] = ghn_grad_norm.item()
            else:
                metrics['ghn_grad_norm'] = 0.0
            
            # Teacher distillation info
            if self.use_teacher_distillation:
                metrics['teacher_update_count'] = self.update_count
                metrics['teacher_update_freq'] = self.teacher_update_freq
                metrics['distillation_weight'] = self.distillation_weight
        
        return metrics


class CustomActorCritic(nn.Module):
    """Custom Actor-Critic implementation with flexible architecture"""
    is_recurrent = False
    
    def __init__(self, 
                 num_actor_obs,
                 num_critic_obs, 
                 num_actions,
                 actor_hidden_dims=[512, 256, 128],
                 critic_hidden_dims=[512, 256, 128],
                 activation='elu',
                 init_noise_std=1.0,
                 use_batch_norm=False,
                 dropout_rate=0.0,
                 **kwargs):
        """
        Custom Actor-Critic with additional features
        
        Args:
            num_actor_obs: Number of actor observations
            num_critic_obs: Number of critic observations  
            num_actions: Number of actions
            actor_hidden_dims: List of hidden layer dimensions for actor
            critic_hidden_dims: List of hidden layer dimensions for critic
            activation: Activation function name
            init_noise_std: Initial action noise standard deviation
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate (0.0 = no dropout)
        """
        if kwargs:
            # Unexpected arguments will be ignored
            pass
        
        super(CustomActorCritic, self).__init__()
        
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
        activation_fn = get_activation(activation)
        
        # Build Actor Network
        self.actor = self._build_mlp(
            input_dim=num_actor_obs,
            hidden_dims=actor_hidden_dims,
            output_dim=num_actions,
            activation_fn=activation_fn,
            final_activation=None  # No activation on final layer
        )
        
        # Build Critic Network  
        self.critic = self._build_mlp(
            input_dim=num_critic_obs,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
            activation_fn=activation_fn,
            final_activation=None  # No activation on final layer
        )
        
        # Custom Actor-Critic networks initialized
        
        # Action noise parameter
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        
        # Disable args validation for speedup
        Normal.set_default_validate_args = False
        
    def _build_mlp(self, input_dim, hidden_dims, output_dim, activation_fn, final_activation=None):
        """Build MLP with optional batch norm and dropout"""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(activation_fn)
        if self.dropout_rate > 0:
            layers.append(nn.Dropout(self.dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(activation_fn)
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        if final_activation is not None:
            layers.append(final_activation)
            
        return nn.Sequential(*layers)
    
    def reset(self, dones=None):
        """Reset function - required by interface"""
        pass
    
    def forward(self):
        """Forward pass - not implemented, use act() and evaluate()"""
        raise NotImplementedError
    
    @property
    def action_mean(self):
        """Get mean of action distribution"""
        return self.distribution.mean
    
    @property
    def action_std(self):
        """Get standard deviation of action distribution"""
        return self.distribution.stddev
    
    @property
    def entropy(self):
        """Get entropy of action distribution"""
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, observations):
        """Update action distribution based on observations"""
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0. + self.std)
    
    def act(self, observations, **kwargs):
        """Sample action from policy distribution"""
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        """Get log probability of given actions"""
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations):
        """Get deterministic action for inference (no noise)"""
        return self.actor(observations)
    
    def evaluate(self, critic_observations, **kwargs):
        """Evaluate value function"""
        return self.critic(critic_observations)


class HyperPPOActorCritic(nn.Module):
    """
    Complete HyperPPO Actor-Critic implementation for GO2 robot
    Integrates HyperActor with traditional critic
    """
    is_recurrent = False
    
    def __init__(self,
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 architecture_config_path: Optional[str] = None,
                 meta_batch_size: int = 2,
                 critic_hidden_dims=[512, 256, 128],
                 activation='elu',
                 init_noise_std=1.0,
                 device='cuda',
                 use_teacher_distillation=True,
                 teacher_update_freq=100,
                 distillation_weight=0.1,
                 **kwargs):
        """
        HyperPPO Actor-Critic for GO2 with depth processing
        
        Args:
            num_actor_obs: Total actor observations (48 + 4096 = 4144)
            num_critic_obs: Total critic observations (same as actor)
            num_actions: Number of actions (12 for GO2 joints)
            architecture_config_path: Path to architecture JSON
            meta_batch_size: Number of architectures per iteration
            critic_hidden_dims: Critic network dimensions
            activation: Activation function
            init_noise_std: Initial action noise
            device: Computation device
            use_teacher_distillation: Whether to use teacher distillation
            teacher_update_freq: How often to update teacher network
            distillation_weight: Weight for distillation loss
        """
        if kwargs:
            # Unexpected arguments will be ignored
            pass
        
        super(HyperPPOActorCritic, self).__init__()
        
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.distillation_weight = distillation_weight
        
        # Initialize hyperActor from existing HyperPPO implementation
        self.hyper_actor = hyperActor(
            act_dim=num_actions,
            obs_dim=num_actor_obs,
            architecture_config_path=architecture_config_path,
            meta_batch_size=meta_batch_size,
            device=device,
            multi_gpu=False,
            architecture_sampling_mode="uniform"
        )
        
        # Architecture-conditioned critic with depth + state + architecture inputs
        # Get architecture descriptor dimension from hyper_actor
        arch_descriptor_dim = 16  # Fixed descriptor length from generate_architectures.py
        privileged_state_dim = 48  # Privileged observations (no depth)
        
        self.critic = ArchConditionedCritic(
            state_dim=privileged_state_dim,
            arch_descriptor_dim=arch_descriptor_dim,
            device=self.device,
            input_size=84  # Depth image size
        )
        
        self.distribution = None
        Normal.set_default_validate_args = False
        
        # HyperPPO Actor-Critic initialized
    
    def _preprocess_depth(self, observations):
        """
        Preprocess observations to extract depth images
        
        Args:
            observations: Combined observations [batch, 48 + 84*84]
                         = [state(48) + flattened_depth(7056)]
        
        Returns:
            dict with 'depth' and 'state' keys
        """
        batch_size = observations.shape[0]
        
        # Split observations: state (48) + flattened depth (84*84)
        state_obs = observations[:, :48]  # [batch, 48]
        depth_flat = observations[:, 48:]  # [batch, 7056]
        
        # Reshape depth to image format [batch, 1, 84, 84]
        depth_images = depth_flat.view(batch_size, 1, 84, 84)
        
        return {
            'depth': depth_images,  # [batch, 1, 84, 84]
            'state': state_obs      # [batch, 48]
        }
    
    def _prepare_critic_obs(self, critic_observations):
        """Prepare observations for architecture-conditioned critic"""
        return self._preprocess_depth(critic_observations)
    
    def get_current_arch_descriptors(self):
        """Get current architecture descriptors from hyper_actor"""
        if hasattr(self.hyper_actor, 'arch_descriptors_per_state'):
            return self.hyper_actor.arch_descriptors_per_state
        else:
            # Fallback: get descriptors from sampled architectures
            if hasattr(self.hyper_actor, 'list_of_sampled_shape_inds') and self.hyper_actor.list_of_sampled_shape_inds:
                # Use first architecture descriptor as fallback
                desc = self.hyper_actor.list_of_sampled_shape_inds[0].flatten()
                # Pad or truncate to fixed length 16
                if len(desc) < 16:
                    desc = torch.cat([desc, torch.zeros(16 - len(desc), device=desc.device)])
                else:
                    desc = desc[:16]
                return desc.unsqueeze(0)  # [1, 16]
            else:
                # Emergency fallback: zeros
                return torch.zeros(1, 16, device=self.device)
    
    def reset(self, dones=None):
        """Reset function - required by interface"""
        pass
    
    def forward(self):
        """Forward pass - not implemented, use act() and evaluate()"""
        raise NotImplementedError
    
    @property
    def action_mean(self):
        """Get mean of action distribution"""
        return self.distribution.mean if self.distribution else None
    
    @property
    def action_std(self):
        """Get standard deviation of action distribution"""
        return self.distribution.stddev if self.distribution else None
    
    @property
    def entropy(self):
        """Get entropy of action distribution"""
        return self.distribution.entropy().sum(dim=-1) if self.distribution else None
    
    def update_distribution(self, observations):
        """Update action distribution using HyperActor"""
        mu, log_std = self.hyper_actor(observations)
        std = torch.exp(log_std)
        self.distribution = Normal(mu, std)
    
    def act(self, observations, **kwargs):
        """Sample action from policy distribution"""
        # Extract arch_descriptors from kwargs if provided (for minibatch training)
        arch_descriptors = kwargs.get('arch_descriptors', None)
        if arch_descriptors is not None:
            # Store for tracking during minibatch updates
            self.current_arch_descriptors = arch_descriptors
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        """Get log probability of given actions"""
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations):
        """Get deterministic action for inference (no noise)"""
        mu, log_std = self.hyper_actor(observations)
        return mu
    
    def evaluate(self, critic_observations, **kwargs):
        """Evaluate value function using architecture-conditioned critic"""
        # Extract arch_descriptors from kwargs if provided (for minibatch training)
        arch_descriptors = kwargs.get('arch_descriptors', None)
        if arch_descriptors is None:
            # Get current architecture descriptors as fallback
            arch_descriptors = self.get_current_arch_descriptors()
        
        # Expand arch descriptors to match batch size
        batch_size = critic_observations.shape[0]
        if arch_descriptors.shape[0] == 1 and batch_size > 1:
            arch_descriptors = arch_descriptors.expand(batch_size, -1)
        
        # Prepare observations dict for critic
        obs_dict = self._prepare_critic_obs(critic_observations)
        
        # Use architecture-conditioned critic
        return self.critic(obs_dict, arch_descriptors)
    
    def resample_architectures(self):
        """Resample architectures for new epoch (call AFTER epoch completes)"""
        self.hyper_actor.change_graph(repeat_sample=False)
    
    def regenerate_weights(self):
        """Regenerate weights for minibatch training (call BEFORE each minibatch)"""
        self.hyper_actor.change_graph(repeat_sample=True)
    
    def get_training_metrics(self):
        """Get GHN training metrics (PPO style)"""
        return self.hyper_actor.get_training_metrics()
    
    def get_current_arch_descriptors(self):
        """Get current architecture descriptors for tracking"""
        if hasattr(self, 'current_arch_descriptors') and self.current_arch_descriptors is not None:
            return self.current_arch_descriptors.clone().detach()
        # Try to get from hyper_actor if available
        if hasattr(self.hyper_actor, 'current_arch_descriptors') and self.hyper_actor.current_arch_descriptors is not None:
            return self.hyper_actor.current_arch_descriptors.clone().detach()
        # Fallback: return zero descriptors if no architecture is set
        return torch.zeros(1, 16, device=self.device)
    
    def compute_distillation_loss(self, observations) -> torch.Tensor:
        """
        Compute teacher distillation loss KL(teacher||student)
        
        Args:
            observations: Input observations [batch_size, obs_dim]
            
        Returns:
            distillation_loss: Weighted KL divergence loss
        """
        kl_loss = self.hyper_actor.compute_distillation_loss(observations)
        return self.distillation_weight * kl_loss
    
    def update_teacher_network(self):
        """Update teacher network in HyperActor"""
        self.hyper_actor.update_teacher_network()
    
    def get_teacher_action_distribution(self, observations):
        """
        Get teacher action distribution for comparison
        
        Args:
            observations: Input observations [batch_size, obs_dim]
            
        Returns:
            teacher_dist: Teacher action distribution
        """
        if not self.hyper_actor.use_teacher_distillation:
            return None
        
        teacher_mu, teacher_log_std = self.hyper_actor.teacher_forward(observations)
        teacher_std = torch.exp(teacher_log_std)
        return Normal(teacher_mu, teacher_std)
    
    def get_distillation_loss(self, observations):
        """Get distillation loss for PPO update"""
        distillation_loss = self.compute_distillation_loss(observations)
        # Update teacher network periodically
        self.update_teacher_network()
        return distillation_loss


class CustomVisualActorCritic(nn.Module):
    """Custom Visual Actor-Critic for depth image processing"""
    is_recurrent = False
    
    def __init__(self,
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 depth_image_shape=(64, 64),
                 depth_latent_dim=32,
                 actor_hidden_dims=[512, 256, 128],
                 critic_hidden_dims=[512, 256, 128],
                 activation='elu',
                 init_noise_std=1.0,
                 use_batch_norm=False,
                 dropout_rate=0.0,
                 **kwargs):
        """
        Custom Visual Actor-Critic with depth processing
        
        Args:
            num_actor_obs: Total actor observations (base_obs + depth_flat)
            num_critic_obs: Total critic observations (base_obs + depth_flat)
            num_actions: Number of actions
            depth_image_shape: Shape of depth images (H, W)
            depth_latent_dim: Latent dimension after depth encoding
            actor_hidden_dims: Actor MLP hidden dimensions
            critic_hidden_dims: Critic MLP hidden dimensions
            activation: Activation function
            init_noise_std: Initial action noise
            use_batch_norm: Use batch normalization
            dropout_rate: Dropout rate
        """
        if kwargs:
            # Unexpected arguments will be ignored
            pass
        
        super(CustomVisualActorCritic, self).__init__()
        
        self.depth_image_size = depth_image_shape[0] * depth_image_shape[1]  # 4096
        self.base_actor_obs_size = num_actor_obs - self.depth_image_size  # 48
        self.base_critic_obs_size = num_critic_obs - self.depth_image_size  # 48
        self.depth_latent_dim = depth_latent_dim
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
        activation_fn = get_activation(activation)
        
        # Custom depth encoder (simple CNN)
        self.depth_encoder = self._build_depth_encoder(depth_image_shape, depth_latent_dim, activation_fn)
        
        # Actor: base_obs + depth_features -> actions
        actor_input_dim = self.base_actor_obs_size + depth_latent_dim
        self.actor = self._build_mlp(
            input_dim=actor_input_dim,
            hidden_dims=actor_hidden_dims,
            output_dim=num_actions,
            activation_fn=activation_fn
        )
        
        # Critic: base_obs + depth_features -> value
        critic_input_dim = self.base_critic_obs_size + depth_latent_dim
        self.critic = self._build_mlp(
            input_dim=critic_input_dim,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
            activation_fn=activation_fn
        )
        
        # Custom Visual Actor-Critic with depth encoder initialized
        
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False
    
    def _build_depth_encoder(self, depth_shape, latent_dim, activation_fn):
        """Build custom depth encoder CNN"""
        h, w = depth_shape
        
        # Simple CNN for depth processing
        encoder = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),  # 64x64 -> 15x15
            activation_fn,
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),  # 15x15 -> 6x6
            activation_fn,
            # Third conv block
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),  # 6x6 -> 4x4
            activation_fn,
            # Flatten and FC
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            activation_fn,
            nn.Linear(128, latent_dim)
        )
        
        return encoder
    
    def _build_mlp(self, input_dim, hidden_dims, output_dim, activation_fn):
        """Build MLP with optional batch norm and dropout"""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(activation_fn)
        if self.dropout_rate > 0:
            layers.append(nn.Dropout(self.dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(activation_fn)
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
            
        return nn.Sequential(*layers)
    
    def _extract_depth_from_obs(self, observations):
        """Extract base obs and depth from observations"""
        base_obs = observations[:, :self.base_actor_obs_size]  # [B, 48]
        depth_flat = observations[:, self.base_actor_obs_size:]  # [B, 4096]
        depth_images = depth_flat.view(-1, 1, 64, 64)  # [B, 1, 64, 64] for CNN
        return base_obs, depth_images
    
    def _encode_depth(self, depth_images):
        """Encode depth images to latent features"""
        return self.depth_encoder(depth_images)
    
    def reset(self, dones=None):
        pass
    
    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean
    
    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, observations):
        """Update action distribution with visual processing"""
        base_obs, depth_images = self._extract_depth_from_obs(observations)
        depth_features = self._encode_depth(depth_images)
        actor_input = torch.cat([base_obs, depth_features], dim=1)
        
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean * 0. + self.std)
    
    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations):
        base_obs, depth_images = self._extract_depth_from_obs(observations)
        depth_features = self._encode_depth(depth_images)
        actor_input = torch.cat([base_obs, depth_features], dim=1)
        return self.actor(actor_input)
    
    def evaluate(self, critic_observations, **kwargs):
        base_obs, depth_images = self._extract_depth_from_obs(critic_observations)
        depth_features = self._encode_depth(depth_images)
        critic_input = torch.cat([base_obs, depth_features], dim=1)
        return self.critic(critic_input)
    
    def get_current_arch_descriptors(self):
        """Get current architecture descriptors for tracking"""
        if hasattr(self, 'current_arch_descriptors') and self.current_arch_descriptors is not None:
            return self.current_arch_descriptors.clone().detach()
        # Fallback: return zero descriptors if no architecture is set
        return torch.zeros(1, 16, device=self.device)
    
    def get_hidden_states(self):
        """Get hidden states - not implemented for non-recurrent networks"""
        return None
    
    def regenerate_weights(self):
        """Regenerate weights using GHN (for HyperPPO minibatch isolation)"""
        if hasattr(self, 'change_graph'):
            self.change_graph(repeat_sample=True)
    
    def resample_architectures(self):
        """Resample architectures for next rollout (for HyperPPO)"""
        if hasattr(self, 'change_graph'):
            self.change_graph(repeat_sample=False)