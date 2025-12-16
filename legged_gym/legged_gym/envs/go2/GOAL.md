# Project Goal: GHN-Based Vision Policy for Go2 Parkour

## Overview

Train a vision-based student policy for depth-based quadruped locomotion on the Unitree Go2 robot, then use Graph HyperNetwork (GHN) to search for optimal depth encoder architectures.

## Two-Phase Approach

### Phase 1: Baseline Student Training (Current)

**Goal:** Validate sim2real transfer with a simple, controlled setup.

- Train CNN+MLP student to copy teacher via behavioral cloning
- Input: D435i depth image + proprioception
- Task: Parkour locomotion (hurdles, flat, steps)
- Real-world: Controlled environment matching sim (minimal disturbance)

**Architecture:**
```
Depth Encoder (CNN):
  depth [58, 87] → CNN → depth_latent [32]

Student Actor (MLP, fixed from teacher):
  [proprio + depth_latent] → MLP [256,256,256] → actions [12]
```

### Phase 2: GHN Depth Encoder Search (Future)

Once sim2real baseline works:
- **GHN samples diverse depth encoder (CNN) architectures only**
- Student actor MLP remains fixed (copied from teacher)
- Same BC training objective: minimize ||action_teacher - action_student||
- Find depth encoder architectures that transfer well to real hardware

**What varies:** Depth encoder CNN architecture
**What stays fixed:** Student actor MLP (same as teacher actor)

## Hardware Setup

**Robot:** Unitree Go2
**Camera:** Intel D435i mounted on head
- Position: [0.28m forward, 0.15m up]
- Resolution: 87x58 (downsampled)
- Depth range: 0.3m - 3.0m

**Real Environment:**
- Parkour steps matching sim dimensions
- Controlled lighting
- Minimal external disturbances

## Training Pipeline

```
1. Teacher (privileged)     2. Student (vision-only)
   LiDAR + terrain info        D435i depth + proprio
          ↓                            ↓
      RL (PPO)                   BC (copy teacher)
          ↓                            ↓
   Expert actions              Deploy on real Go2
```

## Current Settings (Demo)

- **Depth augmentation:** Disabled
- **Domain randomization:** Disabled
- **Single terrain:** parkour_step only
- **Real env:** Controlled to match sim exactly

## Success Criteria

### Phase 1
- [ ] BC loss converges in sim
- [ ] Student climbs steps in sim like teacher
- [ ] Transfer to real Go2 works

### Phase 2
- [ ] GHN predicts working weights for depth encoder CNNs
- [ ] Architecture search finds depth encoders that transfer better than baseline

## File Structure

```
go2/
  go2_config.py          # Teacher config (privileged LiDAR)
  go2_student_config.py  # Student config (depth camera)
  GOAL.md                # This file

rsl_rl/modules/
  depth_backbone.py      # Depth encoder architectures (CNN)
                         # - SimpleDepthEncoder (Phase 1 baseline)
                         # - GHN-sampled encoders (Phase 2)
  actor_critic.py        # Student actor MLP (fixed, copied from teacher)

rsl_rl/runners/
  on_policy_runner.py    # BC training loop (learn_vision)

ppuda/                   # GHN library for Phase 2
  ghn/                   # Graph HyperNetwork implementation
  deepnets1m/            # Architecture graph representation
```
