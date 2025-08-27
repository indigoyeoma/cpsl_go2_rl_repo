# CPSL GO2 Reinforcement Learning

**GO2 quadruped robot training and deployment framework for CPSL research**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Isaac Gym](https://img.shields.io/badge/Isaac%20Gym-Preview%204-green.svg)](https://developer.nvidia.com/isaac-gym)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository provides a complete pipeline for training GO2 quadruped robots in simulation and deploying policies to real hardware. The framework focuses on robust locomotion with visual navigation capabilities using Isaac Gym (no MuJoCo dependency).

### Key Features
- **Visual RL Training** with depth camera integration  
- **Wall Avoidance** and obstacle navigation
- **Stable Locomotion** with proper gait patterns
- **GPU-Accelerated** simulation with 100+ parallel environments
- **Real Robot Deployment** ready configurations

## Repository Structure

```
legged_gym/
├── envs/go2/           # GO2-specific configurations
│   ├── go2_config.py   # Robot parameters, rewards, terrain settings
│   ├── go2_env.py      # Environment with depth camera (from Helpful Doggy)
│   └── README.md       # GO2 configuration details
├── utils/
│   ├── terrain.py      # Wall generation and terrain systems
│   └── helpers.py      # Training utilities
└── scripts/
    ├── train.py        # Training script
    └── play.py         # Policy evaluation
rsl_rl/
└── modules/            # Visual actor-critic networks
```

### Key Components

**Configuration (`go2_config.py`)**
- Reward scaling for forward movement and wall avoidance
- Camera setup (64x64 depth, 87° FOV, 3m range)  
- 100 parallel environments for training

**Environment (`go2_env.py`)**
- Depth camera processing and collision detection
- Visual RL integration with camera mounting
- Extends base LeggedRobot with vision capabilities
- `check_camera` variable controls depth image visualization (needs CLI argument integration)

**Terrain System (`utils/terrain.py`)**
- Wall generation for dodging scenarios
- Trimesh terrain support for complex obstacles

**Visual Policy (`rsl_rl/modules/`)**
- VisualActorCritic for depth image processing
- CNN encoder for visual feature extraction
- Actor-critic architecture with shared vision backbone

## Quick Start

### Installation
```bash
# Clone repository
git clone <repo-url>
cd go2_rl

# Install dependencies  
pip install -r requirements.txt
```

### Training
```bash
# Train GO2 with visual navigation
python legged_gym/scripts/train.py --task=go2

# Monitor training
tensorboard --logdir logs/
```

### Evaluation
```bash
# Test trained policy
python legged_gym/scripts/play.py --task=go2 --load_run <run_name>
```

## Configuration

### Robot Parameters
- **Control**: Position control, 20 N⋅m/rad stiffness
- **Camera**: 64x64 depth images, 87° FOV, 3m range
- **Training**: 100 parallel environments, 30s episodes

### Reward Structure
```python
tracking_lin_vel = 8.0    # Forward movement
collision = -2.0          # Wall avoidance
base_height = -1.0        # Posture stability  
orientation = -1.0        # Body orientation
```

## References

This work builds upon several excellent open-source projects:

- [**Extreme Parkour**](https://github.com/chengxuxin/extreme-parkour) - Advanced quadruped locomotion
- [**Helpful Doggybot**](https://github.com/WooQi57/Helpful-Doggybot) - Visual navigation integration  
- [**Unitree RL Gym**](https://github.com/unitreerobotics/unitree_rl_gym) - Base GO2 simulation framework

## Safety & Parameters

⚠️ **IMPORTANT**: GO2 control hyperparameters are carefully tuned for hardware safety. Modifications should be validated in simulation before deployment.

### Critical Parameters
- Joint position limits: Verified against URDF specifications
- Torque limits: Hardware-safe values only  
- Contact force thresholds: Prevent damage from impacts

## Development Status

🚧 **Work in Progress**: This repository is actively being cleaned and organized for future CPSL research. Current priorities:

- [ ] Code organization and documentation
- [ ] Hardware deployment pipeline
- [ ] Performance benchmarking  
- [ ] Real robot validation
- [ ] Camera visualization CLI integration

## Contributing

This repository will be uploaded to the official CPSL GO2 repo. Please maintain code quality and safety standards for all contributions.

## License

MIT License - See LICENSE file for details.

---

**CPSL Robotics Lab**  
*Advancing quadruped robotics through reinforcement learning*