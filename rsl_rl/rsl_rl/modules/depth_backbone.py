import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torchvision


class DepthAugmentation(nn.Module):
    """Depth image augmentation for sim-to-real transfer.

    Simulates real depth sensor artifacts:
    - Random pixel dropout (sensor failures, reflective surfaces)
    - Gaussian noise (sensor noise)
    - Random rectangular cutouts (occlusions, failed regions)
    - Edge noise (stereo matching failures at depth discontinuities)
    - Depth scaling jitter (calibration errors)
    - Random horizontal flip (data augmentation)

    All augmentations are stochastic and only applied during training.
    """
    def __init__(
        self,
        dropout_prob: float = 0.1,        # Probability of dropping each pixel
        noise_std: float = 0.05,           # Gaussian noise std (in normalized depth units)
        cutout_prob: float = 0.3,          # Probability of applying cutout
        cutout_size_range: tuple = (5, 20),  # Min/max cutout size (pixels)
        num_cutouts: int = 3,              # Number of cutouts when applied
        edge_noise_prob: float = 0.5,      # Probability of edge noise
        edge_noise_std: float = 0.1,       # Edge noise intensity
        scale_jitter: float = 0.05,        # Depth scale jitter (+/- percentage)
        flip_prob: float = 0.0,            # Horizontal flip probability (0 for locomotion)
        hole_value: float = 0.0,           # Value for invalid/dropped pixels
    ):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.noise_std = noise_std
        self.cutout_prob = cutout_prob
        self.cutout_size_range = cutout_size_range
        self.num_cutouts = num_cutouts
        self.edge_noise_prob = edge_noise_prob
        self.edge_noise_std = edge_noise_std
        self.scale_jitter = scale_jitter
        self.flip_prob = flip_prob
        self.hole_value = hole_value

        # Sobel kernels for edge detection
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: [batch_size, H, W] or [batch_size, num_frames, H, W] depth images
        Returns:
            Augmented depth images with same shape
        """
        if not self.training:
            return depth

        # Handle both single frame and stacked frames
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)  # [B, 1, H, W]
            squeeze_output = True
        else:
            squeeze_output = False

        B, N, H, W = depth.shape
        device = depth.device

        # Work on flattened batch for efficiency
        depth = depth.view(B * N, 1, H, W)

        # 1. Depth scale jitter (calibration error simulation)
        if self.scale_jitter > 0:
            scale = 1.0 + (torch.rand(B * N, 1, 1, 1, device=device) * 2 - 1) * self.scale_jitter
            depth = depth * scale

        # 2. Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(depth) * self.noise_std
            depth = depth + noise

        # 3. Random pixel dropout (simulates sensor failures)
        if self.dropout_prob > 0:
            dropout_mask = torch.rand(B * N, 1, H, W, device=device) > self.dropout_prob
            depth = torch.where(dropout_mask, depth, torch.full_like(depth, self.hole_value))

        # 4. Edge noise (stereo matching failures at depth discontinuities)
        if self.edge_noise_prob > 0 and torch.rand(1).item() < self.edge_noise_prob:
            # Compute edges using Sobel
            edges_x = F.conv2d(depth, self.sobel_x, padding=1)
            edges_y = F.conv2d(depth, self.sobel_y, padding=1)
            edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)

            # Normalize edges and apply noise proportionally
            edges_norm = edges / (edges.max() + 1e-6)
            edge_noise = torch.randn_like(depth) * self.edge_noise_std * edges_norm
            depth = depth + edge_noise

        # 5. Random rectangular cutouts (occlusions, failed regions)
        if self.cutout_prob > 0 and torch.rand(1).item() < self.cutout_prob:
            for _ in range(self.num_cutouts):
                # Random cutout size
                cut_h = torch.randint(self.cutout_size_range[0], self.cutout_size_range[1] + 1, (1,)).item()
                cut_w = torch.randint(self.cutout_size_range[0], self.cutout_size_range[1] + 1, (1,)).item()

                # Random position
                top = torch.randint(0, max(1, H - cut_h), (1,)).item()
                left = torch.randint(0, max(1, W - cut_w), (1,)).item()

                # Apply cutout
                depth[:, :, top:top+cut_h, left:left+cut_w] = self.hole_value

        # 6. Horizontal flip (data augmentation - careful with locomotion)
        if self.flip_prob > 0 and torch.rand(1).item() < self.flip_prob:
            depth = torch.flip(depth, dims=[-1])

        # Reshape back
        depth = depth.view(B, N, H, W)
        if squeeze_output:
            depth = depth.squeeze(1)

        return depth

    def __repr__(self):
        return (f"DepthAugmentation(dropout={self.dropout_prob}, noise={self.noise_std}, "
                f"cutout={self.cutout_prob}, edge_noise={self.edge_noise_prob})")


class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone, env_cfg, latent_dim=32) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        # Store for forward pass (optional validation)
        self.latent_dim = latent_dim
        
        if env_cfg == None:
            self.combination_mlp = nn.Sequential(
                                    nn.Linear(latent_dim + 53, 128),
                                    activation,
                                    nn.Linear(128, latent_dim)
                                )
        else:
            self.combination_mlp = nn.Sequential(
                                        nn.Linear(latent_dim + env_cfg.env.n_proprio, 128),
                                        activation,
                                        nn.Linear(128, latent_dim)
                                    )
        self.rnn = nn.GRU(input_size=latent_dim, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(
                                nn.Linear(512, latent_dim+2),
                                last_activation
                            )
        self.hidden_states = None

    def forward(self, depth_image, proprioception):
        depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        # depth_latent = self.base_backbone(depth_image)
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        
        return depth_latent

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()

class SimpleDepthEncoder(nn.Module):
    """Identical to RecurrentDepthBackbone but without GRU (no temporal processing).

    Flow: depth → base_backbone → [32] → cat proprio → [85] → combination_mlp → [32] → output_mlp → [34]
    (RecurrentDepthBackbone has GRU between combination_mlp and output_mlp)
    """
    def __init__(self, base_backbone, env_cfg, latent_dim=32) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        self.latent_dim = latent_dim

        # Same as RecurrentDepthBackbone
        if env_cfg == None:
            self.combination_mlp = nn.Sequential(
                                    nn.Linear(latent_dim + 53, 128),
                                    activation,
                                    nn.Linear(128, latent_dim)
                                )
        else:
            self.combination_mlp = nn.Sequential(
                                        nn.Linear(latent_dim + env_cfg.env.n_proprio, 128),
                                        activation,
                                        nn.Linear(128, latent_dim)
                                    )

        # NO GRU here (RecurrentDepthBackbone has: self.rnn = nn.GRU(input_size=32, hidden_size=512))

        # Output: latent_dim → latent_dim+2 (instead of 512 → 34 in RecurrentDepthBackbone)
        self.output_mlp = nn.Sequential(
                                nn.Linear(latent_dim, latent_dim + 2),
                                last_activation
                            )

    def forward(self, depth_image, proprioception):
        """
        Args:
            depth_image: [batch_size, H, W] single depth frame
            proprioception: [batch_size, n_proprio] robot state
        Returns:
            [batch_size, 34] = 32 depth latent + 2 yaw estimate
        """
        # Same as RecurrentDepthBackbone (no unsqueeze - base_backbone handles it)
        depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))

        # NO GRU (RecurrentDepthBackbone does: depth_latent, self.hidden_states = self.rnn(...))

        # Output layer (same structure, different input size)
        depth_latent = self.output_mlp(depth_latent)

        return depth_latent

    def detach_hidden_states(self):
        """No-op: SimpleDepthEncoder has no hidden states (no GRU)."""
        pass


class StackDepthEncoder(nn.Module):
    def __init__(self, base_backbone, env_cfg, latent_dim=32) -> None:
        super().__init__()
        activation = nn.ELU()
        self.base_backbone = base_backbone
        self.combination_mlp = nn.Sequential(
                                    nn.Linear(latent_dim + env_cfg.env.n_proprio, 128),
                                    activation,
                                    nn.Linear(128, latent_dim)
                                )

        # COMMENTED OUT: Temporal encoding with Conv1D (for buffer_len > 1)
        # self.conv1d = nn.Sequential(nn.Conv1d(in_channels=env_cfg.depth.buffer_len, out_channels=16, kernel_size=4, stride=2),  # (30 - 4) / 2 + 1 = 14,
        #                             activation,
        #                             nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2), # 14-2+1 = 13,
        #                             activation)
        # self.mlp = nn.Sequential(nn.Linear(16*14, 32),
        #                          activation)

    def forward(self, depth_image, proprioception):
        # depth_image shape: [batch_size, num, 58, 87]
        depth_latent = self.base_backbone(None, depth_image.flatten(0, 1), None)  # [batch_size * num, 32]
        depth_latent = depth_latent.reshape(depth_image.shape[0], depth_image.shape[1], -1)  # [batch_size, num, 32]

        # COMMENTED OUT: Temporal encoding (no longer needed with buffer_len=1)
        # depth_latent = self.conv1d(depth_latent)
        # depth_latent = self.mlp(depth_latent.flatten(1, 2))

        # Single frame: just squeeze the temporal dimension and return [batch_size, 32]
        depth_latent = depth_latent.squeeze(1)  # Remove temporal dimension since buffer_len=1
        return depth_latent

    
class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # [64, 25, 39]
            activation,
            nn.Flatten(),
            # [64 * 25 * 39 = 62400]
            nn.Linear(62400, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)

        return latent


class DepthOnlyFCBackbone128x96(nn.Module):
    """Depth encoder backbone for 96x128 input resolution.

    Input: [batch, 96, 128] depth image
    Output: [batch, scandots_output_dim] latent vector (typically 32)
    """
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 96, 128]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 92, 124]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 46, 62]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # [64, 44, 60]
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [64, 22, 30]
            nn.Flatten(),
            nn.Linear(64 * 22 * 30, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)

        return latent


class DepthOnlyFCBackbone48x64(nn.Module):
    """Depth encoder backbone for 48x64 input resolution (matched to parkour repo).

    Input: [batch, 48, 64] depth image
    Output: [batch, scandots_output_dim] latent vector (typically 32)
    """
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 48, 64]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 44, 60]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 22, 30]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # [64, 20, 28]
            activation,
            nn.Flatten(),
            nn.Linear(64 * 20 * 28, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)

        return latent