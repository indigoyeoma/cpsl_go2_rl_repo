#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple validation test to check configuration files and imports
"""

import sys
import os
import json

def test_config_validation():
    print("Testing HyperPPO Configuration Validation")
    
    try:
        # Test 1: Load GO2 configuration
        sys.path.append('/home/jiwoo/ws/go2_rl_jw')
        
        try:
            from legged_gym.envs.go2.go2_config import GO2RoughCfgPPO, GO2RoughCfg
            print("Successfully imported GO2 configuration")
            
            # Check runner config
            config = GO2RoughCfgPPO()
            print("   Policy class: " + str(config.runner.policy_class_name))
            print("   Algorithm class: " + str(config.runner.algorithm_class_name))
            print("   Experiment name: " + str(config.runner.experiment_name))
            
        except Exception as e:
            print("Failed to import GO2 config: " + str(e))
            import traceback
            traceback.print_exc()
            return False
        
        # Test 2: Load architecture configurations
        config_path = "/home/jiwoo/ws/go2_rl_jw/rsl_rl/rsl_rl/hyperppo/configs/architecture_go2_depth64_corrected.json"
        print("\nTesting architecture config loading")
        
        if not os.path.exists(config_path):
            print("Architecture config file does not exist: " + config_path)
            return False
        
        with open(config_path, 'r') as f:
            arch_data = json.load(f)
        
        metadata = arch_data['metadata']
        architectures = arch_data['architectures']
        
        print("Successfully loaded architecture config")
        print("   Number of architectures: " + str(len(architectures)))
        print("   Input channels: " + str(metadata['input_channels']))
        print("   Input size: " + str(metadata['input_size']))
        print("   State dim: " + str(metadata['state_dim']))
        print("   Output dim: " + str(metadata['output_dim']))
        
        # Test 3: Check HyperOnPolicyRunner import
        print("\nTesting HyperOnPolicyRunner import")
        try:
            from rsl_rl.rsl_rl.runners.hyper_on_policy_runner import HyperOnPolicyRunner
            print("Successfully imported HyperOnPolicyRunner")
        except Exception as e:
            print("Failed to import HyperOnPolicyRunner: " + str(e))
            import traceback
            traceback.print_exc()
            return False
        
        # Test 4: Check task registry modification
        print("\nTesting task registry")
        try:
            from legged_gym.utils.task_registry import task_registry
            print("Successfully imported task registry")
        except Exception as e:
            print("Failed to import task registry: " + str(e))
            import traceback
            traceback.print_exc()
            return False
        
        print("\nAll configuration validation tests passed!")
        return True
        
    except Exception as e:
        print("Configuration validation failed: " + str(e))
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config_validation()
    if success:
        print("\nHyperPPO configuration is valid!")
        sys.exit(0)
    else:
        print("\nHyperPPO configuration needs fixes.")
        sys.exit(1)