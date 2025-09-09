#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal test script to validate HyperPPO integration logic
Tests the hyperActor initialization and architecture loading
"""

import sys
import os
sys.path.append('/home/jiwoo/ws/go2_rl_jw')
sys.path.append('/home/jiwoo/ws/go2_rl_jw/rsl_rl')

import torch
import json

def test_hyperppo_integration():
    print("Testing HyperPPO Integration Logic")
    
    try:
        # Test 1: Load architecture configurations
        config_path = "/home/jiwoo/ws/go2_rl_jw/rsl_rl/rsl_rl/hyperppo/configs/architecture_go2_depth64_corrected.json"
        print("Loading architecture config from: " + config_path)
        
        with open(config_path, 'r') as f:
            arch_data = json.load(f)
        
        metadata = arch_data['metadata']
        architectures = arch_data['architectures']
        
        print("Loaded " + str(len(architectures)) + " architectures")
        print("   Input channels: " + str(metadata['input_channels']))
        print("   Input size: " + str(metadata['input_size']))
        print("   State dim: " + str(metadata['state_dim']))
        print("   Output dim: " + str(metadata['output_dim']))
        
        # Test 2: Try importing HyperPPO components
        print("\nTesting HyperPPO component imports")
        
        try:
            from rsl_rl.rsl_rl.hyperppo.src.core import hyperActor
            print("Successfully imported hyperActor")
        except ImportError as e:
            print("Failed to import hyperActor: " + str(e))
            return False
        
        try:
            from rsl_rl.rsl_rl.hyperppo.src.model import CnnMlpNetwork
            print("Successfully imported CnnMlpNetwork")
        except ImportError as e:
            print("Failed to import CnnMlpNetwork: " + str(e))
            return False
            
        try:
            from rsl_rl.rsl_rl.hyperppo.src.ghn_modules import MLP_GHN
            print("Successfully imported MLP_GHN")
        except ImportError as e:
            print("Failed to import MLP_GHN: " + str(e))
            return False
        
        # Test 3: Try creating hyperActor instance
        print("\nTesting hyperActor initialization")
        
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print("   Using device: " + str(device))
        
        try:
            hyper_actor = hyperActor(
                act_dim=12,  # GO2 action dimension
                obs_dim=48,  # GO2 state observation dimension  
                architecture_config_path=config_path,
                meta_batch_size=4,
                device=device,
                multi_gpu=False,
                architecture_sampling_mode="uniform"
            )
            print("Successfully created hyperActor instance")
            print("   Number of architectures: " + str(len(hyper_actor.all_models)))
            print("   Meta batch size: " + str(hyper_actor.meta_batch_size))
            print("   Architecture sampling mode: " + str(hyper_actor.architecture_sampling_mode))
        except Exception as e:
            print("Failed to create hyperActor: " + str(e))
            import traceback
            traceback.print_exc()
            return False
        
        # Test 4: Test architecture sampling and weight generation  
        print("\nTesting architecture sampling and weight generation")
        
        try:
            hyper_actor.change_graph(repeat_sample=True)
            print("Successfully changed architecture and generated weights")
            print("   Current sampled indices: " + str(hyper_actor.sampled_indices))
        except Exception as e:
            print("Failed architecture change: " + str(e))
            import traceback
            traceback.print_exc()
            return False
        
        # Test 5: Test forward pass with dummy data
        print("\nTesting forward pass with dummy data")
        
        try:
            batch_size = 8  # Small test batch
            # Depth image: 1 channel, 64x64
            dummy_depth = torch.randn(batch_size, 1, 64, 64).to(device)
            # State observations: 48 dimensional 
            dummy_state = torch.randn(batch_size, 48).to(device)
            
            # Forward pass
            mu, log_std = hyper_actor.forward(dummy_depth, dummy_state)
            
            print("Successfully completed forward pass")
            print("   Output mu shape: " + str(mu.shape))
            print("   Output log_std shape: " + str(log_std.shape))
            print("   Expected action shape: (" + str(batch_size) + ", 12)")
            
            # Verify output shapes
            if mu.shape == (batch_size, 12) and log_std.shape == (batch_size, 12):
                print("Output shapes are correct")
            else:
                print("Output shapes are incorrect")
                return False
                
        except Exception as e:
            print("Failed forward pass: " + str(e))
            import traceback
            traceback.print_exc()
            return False
        
        print("\nAll HyperPPO integration tests passed!")
        return True
        
    except Exception as e:
        print("Integration test failed: " + str(e))
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hyperppo_integration()
    if success:
        print("\nHyperPPO integration is ready for deployment!")
        sys.exit(0)
    else:
        print("\nHyperPPO integration needs fixes before deployment.")
        sys.exit(1)