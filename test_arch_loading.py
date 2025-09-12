#!/usr/bin/env python3

import json
import sys
import os

# Add the path to the RSL_RL modules
sys.path.append('/home/nvidiasims/jw_ws/cpsl_go2_rl_repo/rsl_rl')

def test_architecture_loading():
    config_path = "rsl_rl/rsl_rl/hyperppo/configs/architecture_go2_depth84.json"
    
    print(f"Loading architectures from: {config_path}")
    
    # Load JSON data
    with open(config_path, 'r') as f:
        arch_data = json.load(f)
    
    metadata = arch_data['metadata']
    architectures = arch_data['architectures']
    
    print(f"Metadata keys: {list(metadata.keys())}")
    print(f"Number of architectures in JSON: {len(architectures)}")
    
    # Check if all architectures have required fields
    valid_count = 0
    for i, arch in enumerate(architectures):
        required_fields = ['cnn_config', 'mlp_config', 'arch_descriptor']
        if all(field in arch for field in required_fields):
            valid_count += 1
        else:
            print(f"Architecture {i} is missing fields: {[f for f in required_fields if f not in arch]}")
            if i < 5 or i > len(architectures) - 5:  # Show first and last few
                print(f"  Architecture {i}: {list(arch.keys())}")
    
    print(f"Valid architectures: {valid_count}")
    
    # Test index range
    indices = list(range(len(architectures)))
    print(f"Index range: [{min(indices)}, {max(indices)}]")
    
    # Test if we can sample 2 architectures
    import numpy as np
    sampled = np.random.choice(indices, 2, replace=False)
    print(f"Sample indices: {sampled}")
    
    return len(architectures), valid_count

if __name__ == "__main__":
    total, valid = test_architecture_loading()
    print(f"\nSUMMARY: {valid}/{total} valid architectures")