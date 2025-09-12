#!/usr/bin/env python3

import torch
import sys
import os

# Add the hyperppo directory to path
sys.path.insert(0, '/home/nvidiasims/jw_ws/cpsl_go2_rl_repo/rsl_rl/rsl_rl/hyperppo')

from src.model import CnnMlpNetwork
import json

def test_cnn_architecture(arch_id, batch_size=192):
    print(f"Testing architecture {arch_id} with batch size {batch_size}")
    
    # Load architecture config
    with open('/home/nvidiasims/jw_ws/cpsl_go2_rl_repo/rsl_rl/rsl_rl/hyperppo/configs/architecture_go2_depth84.json', 'r') as f:
        data = json.load(f)
    
    arch = data['architectures'][arch_id]
    print(f"CNN config: {arch['cnn_config']}")
    print(f"CNN output size: {arch['cnn_output_size']}")
    
    # Create the network
    network = CnnMlpNetwork(
        cnn_config=arch['cnn_config'],
        cnn_mlp_config=arch['cnn_mlp_config'],
        mlp_config=arch['mlp_config'],
        state_mlp_config=arch['state_mlp_config'],
        state_dim=48,
        input_channels=1,
        output_dim=12,
        input_size=84
    ).cuda()
    
    # Create test inputs
    depth_input = torch.randn(batch_size, 1, 84, 84, device='cuda:0')
    state_input = torch.randn(batch_size, 48, device='cuda:0')
    
    print(f"Input shapes: depth={depth_input.shape}, state={state_input.shape}")
    
    try:
        with torch.no_grad():
            output = network(depth_input, state_input)
        print(f"SUCCESS: Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        print(f"Error type: {type(e)}")
        return False

if __name__ == "__main__":
    print("Testing specific CNN architectures that cause CUDA errors...")
    print("=" * 60)
    
    # Test the architectures that were causing issues
    test_archs = [495, 286, 0, 1]  # Include some working ones for comparison
    
    for arch_id in test_archs:
        print()
        success = test_cnn_architecture(arch_id, batch_size=192)
        print("-" * 40)
        
        if not success:
            print("Testing with smaller batch size...")
            success_small = test_cnn_architecture(arch_id, batch_size=32)
            if success_small:
                print("SUCCESS with smaller batch - likely a memory/batch size issue")
            else:
                print("FAILED even with small batch - architecture issue")
        print("=" * 40)