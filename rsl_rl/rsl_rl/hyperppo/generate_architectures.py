#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diverse Architecture Generator for GHN Training

Creates diverse CNN+MLP architecture configurations for training GHN with variety.
"""

import argparse
import os
import random
import torch
import time
import json
import platform
import subprocess
from src.model import CnnMlpNetwork


# -------------------------- Utilities for timing ------------------------------

def _is_cuda_device(device):
    return isinstance(device, str) and device.startswith('cuda')

def _synchronize_if_cuda(device):
    if _is_cuda_device(device):
        torch.cuda.synchronize()


# -------------------------- Hardware Detection --------------------------------

def get_cpu_name():
    """Get CPU name"""
    try:
        if platform.system() == "Windows":
            return platform.processor()
        elif platform.system() == "Darwin":  # macOS
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else platform.processor()
        else:  # Linux
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            return line.split(':')[1].strip()
            except:
                pass
            return platform.processor()
    except:
        return platform.processor() or "Unknown CPU"

def get_gpu_name():
    """Get GPU name if CUDA is available"""
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except:
            return "Unknown CUDA GPU"
    return None

def get_hardware_name():
    """Get just the hardware name - GPU if available, otherwise CPU"""
    gpu_name = get_gpu_name()
    if gpu_name:
        return gpu_name
    else:
        return get_cpu_name()


# -------------------------- Core helpers --------------------------------------

def calculate_cnn_output_size(cnn_config, input_size):
    """Calculate the flattened CNN output size for a given configuration"""
    size = input_size
    for layer in cnn_config:
        kernel = layer['kernel']
        stride = layer['stride']
        padding = layer['padding']
        size = (size + 2 * padding - kernel) // stride + 1
    
    # Return flattened size: height * width * channels
    channels = cnn_config[-1]['channels'] if cnn_config else 1
    return size * size * channels

def calculate_complexity_score(cnn_config, mlp_config):
    """Calculate architecture complexity score for curriculum ordering"""
    # Primary: Total depth (CNN + MLP layers) - most important for GHN stability
    total_depth = len(cnn_config) + len(mlp_config)
    
    # Secondary: CNN parameter count (most expensive computationally)
    cnn_params = 0
    prev_channels = 1  # Depth input (single channel)
    for layer in cnn_config:
        channels = layer['channels']
        kernel = layer['kernel'] 
        # Conv params: (in_channels * kernel^2 + 1) * out_channels
        cnn_params += (prev_channels * kernel * kernel + 1) * channels
        prev_channels = channels
    
    # Tertiary: MLP parameter count (rough proxy)
    mlp_params = sum(mlp_config)
    
    # Quaternary: Average channel count (affects feature learning difficulty)
    avg_channels = sum(layer['channels'] for layer in cnn_config) / len(cnn_config) if cnn_config else 0
    
    # Complexity score: prioritize depth → CNN params → MLP params → channels
    complexity = (
        total_depth * 1000000 +        # Depth most important (1M weight)
        cnn_params * 100 +             # CNN params secondary (100 weight)  
        mlp_params * 10 +              # MLP params tertiary (10 weight)
        avg_channels * 1               # Channel count for fine-tuning (1 weight)
    )
    
    return complexity




# -------------------------- Main generation & benchmarking --------------------

def main(args):
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Get hardware name
    hardware_name = get_hardware_name()
    print(f"Hardware: {hardware_name}")
    
    print(f"Generating ALL possible architectures: CNN{args.cnn_layer_options} + MLP{args.mlp_layer_options}")
    print("####")
    
    # Create architectures with independent layer dimensions
    architectures = []
    max_channels = max(args.cnn_channel_options)
    max_kernel = max(args.cnn_kernel_options)
    max_desc_length = 0
    
    # Calculate total possible unique architectures
    from itertools import product
    
    total_unique = 0
    for cnn_layers in args.cnn_layer_options:
        cnn_combinations = len(args.cnn_channel_options) * len(args.cnn_kernel_options)
        cnn_unique = cnn_combinations ** cnn_layers
        for mlp_layers in args.mlp_layer_options:
            mlp_unique = len(args.mlp_dim_options) ** mlp_layers
            total_unique += cnn_unique * mlp_unique
    
    print(f"Total possible architectures: {total_unique}")
    
    # Generate ALL architectures
    sample_randomly = False
    
    # Generate ALL possible architectures first, then sort by complexity
    all_architectures = []
    arch_id = 0
    
    for cnn_layers in args.cnn_layer_options:
        for mlp_layers in args.mlp_layer_options:
            # Generate CNN configurations with increasing channels
            # Generate all kernel combinations for this depth
            kernel_configs = list(product(args.cnn_kernel_options, repeat=cnn_layers))
            
            # Generate all increasing channel combinations for this depth
            increasing_channel_configs = []
            for channels in product(args.cnn_channel_options, repeat=cnn_layers):
                # Only keep configurations where channels are non-decreasing
                if all(channels[i] <= channels[i+1] for i in range(len(channels)-1)):
                    increasing_channel_configs.append(channels)
            
            # Generate all MLP layer combinations for this depth  
            mlp_layer_configs = list(product(args.mlp_dim_options, repeat=mlp_layers))
            
            # Combine CNN and MLP configurations
            for channels_combo in increasing_channel_configs:
                for kernels_combo in kernel_configs:
                    for mlp_combo in mlp_layer_configs:
                        # Create CNN config with increasing channels
                        cnn_config = []
                        for i in range(cnn_layers):
                            cnn_config.append({
                                "channels": channels_combo[i],
                                "kernel": kernels_combo[i],
                                "stride": 2, 
                                "padding": 1
                            })
                        
                        # Calculate actual CNN output size for this architecture
                        actual_cnn_output_size = calculate_cnn_output_size(cnn_config, args.input_size)
                        
                        # Skip architectures that would cause empty tensors
                        if actual_cnn_output_size <= 0:
                            continue
                        
                        # Parallel architecture: 
                        # Vision: CNN → flatten → CNN_MLP(256)
                        # State: state_dim → State_MLP(256)
                        # Combined: concat(512) → MLP → actions
                        cnn_mlp_config = [args.cnn_output_dim]  # Vision branch output: 256
                        state_mlp_config = [args.state_mlp_dim]  # State branch output: 256
                        
                        # MLP takes concatenated features (256 vision + 256 state = 512)
                        first_mlp_input = args.cnn_output_dim + args.state_mlp_dim  # 512
                        
                        # Regular MLP config: [512, hidden1, hidden2, ...]
                        mlp_config = [first_mlp_input] + list(mlp_combo)
                        
                        # Create numerical descriptor: [ch1, k1, ch2, k2, ..., mlp1, mlp2, ...]
                        arch_descriptor = []
                        for layer in cnn_config:
                            arch_descriptor.extend([layer["channels"], layer["kernel"]])
                        arch_descriptor.extend(mlp_config)
                        
                        # Create string token descriptor for HELP: ["16ch3k", "32ch3k", "512mlp", ...]
                        token_descriptor = []
                        for layer in cnn_config:
                            token_descriptor.append(f"{layer['channels']}ch{layer['kernel']}k")
                        for dim in mlp_config:
                            token_descriptor.append(f"{dim}mlp")
                        
                        # Track max descriptor length for padding
                        max_desc_length = max(max_desc_length, len(arch_descriptor))
                        
                        # Calculate complexity score for sorting
                        complexity = calculate_complexity_score(cnn_config, mlp_config)
                        
                        arch = {
                            "id": arch_id, 
                            "cnn_config": cnn_config,
                            "cnn_mlp_config": cnn_mlp_config,
                            "state_mlp_config": state_mlp_config,  # State branch MLP
                            "mlp_config": mlp_config, 
                            "arch_descriptor": arch_descriptor,
                            "token_descriptor": token_descriptor,  # String tokens for HELP
                            "complexity": complexity,
                            "cnn_output_size": actual_cnn_output_size  # Store for GHN-2
                        }
                        all_architectures.append(arch)
                        arch_id += 1
    
    # Sort by complexity (smallest first)
    all_architectures.sort(key=lambda x: x["complexity"])
    
    # Use all generated architectures
    selected_architectures = all_architectures
    
    # Validate selected architectures by actually testing them
    print(f"Validating {len(selected_architectures)} selected architectures...")
    architectures = []
    for i, arch in enumerate(selected_architectures):
        try:
            test_network = CnnMlpNetwork(
                cnn_config=arch['cnn_config'],
                cnn_mlp_config=arch['cnn_mlp_config'],
                mlp_config=arch['mlp_config'],
                state_mlp_config=arch['state_mlp_config'],
                state_dim=args.state_dim,
                input_channels=1,
                output_dim=args.output_dim,
                input_size=args.input_size
            )
            
            # Test forward pass with dummy input  
            dummy_image = torch.randn(1, 1, args.input_size, args.input_size)
            dummy_state = torch.randn(1, args.state_dim)
            with torch.no_grad():
                _ = test_network(dummy_image, dummy_state)
                
            # If we get here, the architecture is valid
            architectures.append(arch)
            del test_network, dummy_image, dummy_state, _
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            if (i + 1) % 100 == 0 or i == 0:
                print(f"Validated {i+1}/{len(selected_architectures)} architectures...")
            
        except Exception as e:
            print(f"⚠️ Skipped invalid architecture {arch['id']}: {e}")
            continue
    
    print(f"Final valid architectures: {len(architectures)}/{len(selected_architectures)}")
    print("####")
    
    # Get the biggest architecture for teacher model (last in sorted list)
    teacher_complexity = all_architectures[-1]['complexity']  # Store before deletion
    teacher_architecture = all_architectures[-1].copy()  # Biggest complexity
    teacher_architecture["id"] = "teacher"  # Special ID for teacher
    del teacher_architecture["complexity"]
    
    # Remove complexity field and reassign sequential IDs for regular architectures
    for i, arch in enumerate(architectures):
        arch["id"] = i
        del arch["complexity"]
    
    print(f"Teacher: {len(teacher_architecture['cnn_config'])} CNN + {len(teacher_architecture['mlp_config'])} MLP layers")
    
    # Pad all descriptors to fixed length (16) for GHN compatibility
    # Format: [ch1,k1,ch2,k2,ch3,k3,ch4,k4,fixed_mlp_dim,mlp1,mlp2,mlp3,mlp4,0] - padded with zeros
    # Note: fixed_mlp_dim represents the fixed first MLP layer after flattening
    FIXED_DESC_LENGTH = 16
    # Pad regular architectures
    for arch in architectures:
        # Pad numerical descriptor to fixed length
        while len(arch["arch_descriptor"]) < FIXED_DESC_LENGTH:
            arch["arch_descriptor"].append(0)
        # Truncate if too long (shouldn't happen with current config)
        arch["arch_descriptor"] = arch["arch_descriptor"][:FIXED_DESC_LENGTH]
        
        # Pad token descriptor to fixed length with empty strings
        while len(arch["token_descriptor"]) < FIXED_DESC_LENGTH:
            arch["token_descriptor"].append("")
        arch["token_descriptor"] = arch["token_descriptor"][:FIXED_DESC_LENGTH]
    
    # Pad teacher architecture descriptors
    while len(teacher_architecture["arch_descriptor"]) < FIXED_DESC_LENGTH:
        teacher_architecture["arch_descriptor"].append(0)
    teacher_architecture["arch_descriptor"] = teacher_architecture["arch_descriptor"][:FIXED_DESC_LENGTH]
    
    # Create token descriptor for teacher if it doesn't exist
    if "token_descriptor" not in teacher_architecture:
        teacher_token_descriptor = []
        for layer in teacher_architecture["cnn_config"]:
            teacher_token_descriptor.append(f"{layer['channels']}ch{layer['kernel']}k")
        for dim in teacher_architecture["mlp_config"]:
            teacher_token_descriptor.append(f"{dim}mlp")
        teacher_architecture["token_descriptor"] = teacher_token_descriptor
    
    while len(teacher_architecture["token_descriptor"]) < FIXED_DESC_LENGTH:
        teacher_architecture["token_descriptor"].append("")
    teacher_architecture["token_descriptor"] = teacher_architecture["token_descriptor"][:FIXED_DESC_LENGTH]
    
    # Create metadata with unified max_shape for single ConvDecoder (handles both CNN and MLP)
    max_mlp_dim = max(args.mlp_dim_options)
    max_channels = max(args.cnn_channel_options)
    
    # Maximum first MLP layer size is just the processed CNN output
    max_first_mlp_input = args.cnn_output_dim
    
    # Simplified GHN config - HyperPPO style treats all weights uniformly
    # Use simple max shape that works for both CNN and MLP weights
    unified_max_dim = 256  # Fixed reasonable size for both conv and MLP weights  
    unified_max_kernel = 3  # Fixed 3x3 kernels only
    
    data = {
        "metadata": {
            "input_channels": 1, "input_size": args.input_size, "output_dim": args.output_dim,
            "state_dim": args.state_dim,
            "cnn_output_dim": args.cnn_output_dim,
            "state_mlp_dim": args.state_mlp_dim,
            "hardware_name": hardware_name,  # Add hardware name
            "ghn_config": {
                "ghn_max_shape": [unified_max_dim, unified_max_dim, unified_max_kernel, unified_max_kernel],  # HyperPPO-style ConvDecoder
                "simple_classification": True  # All weights → cls_w, simplified approach
            },
            "arch_descriptor_config": {
                "total_length": FIXED_DESC_LENGTH,
                "arch_descriptor_dim": FIXED_DESC_LENGTH,
                "description": "Format: [ch1,k1,ch2,k2,ch3,k3,ch4,k4,mlp_input,mlp1,mlp2,mlp3] - mlp_input=concat(vision+state)"
            }
        },
        "architectures": architectures,
        "teacher_model": teacher_architecture
    }
    
    # Save file with compact formatting for multiple architectures
    output_dir = os.path.dirname(args.output)
    if output_dir:  # Only create if there's a directory part
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        # Write metadata
        f.write('{\n  "metadata": {\n')
        f.write(f'    "input_channels": 1,\n')
        f.write(f'    "input_size": {args.input_size},\n')
        f.write(f'    "output_dim": {args.output_dim},\n')
        f.write(f'    "state_dim": {args.state_dim},\n')
        f.write(f'    "cnn_output_dim": {args.cnn_output_dim},\n')
        f.write(f'    "state_mlp_dim": {args.state_mlp_dim},\n')
        
        # Write hardware name
        f.write(f'    "hardware_name": "{hardware_name}",\n')
        max_mlp_dim = max(args.mlp_dim_options)
        max_kernel = max(args.cnn_kernel_options)
        f.write(f'    "ghn_config": {{\n')
        f.write(f'      "ghn_max_shape": [{unified_max_dim},{unified_max_dim},{unified_max_kernel},{unified_max_kernel}]\n')
        f.write(f'    }},\n')
        f.write(f'    "arch_descriptor_config": {{\n      "total_length": {FIXED_DESC_LENGTH},\n      "arch_descriptor_dim": {FIXED_DESC_LENGTH},\n      "description": "Format: [ch1,k1,ch2,k2,ch3,k3,ch4,k4,mlp_input,mlp1,mlp2,mlp3] - mlp_input=CNN_output"\n    }}\n  }},\n')
        
        # Write architectures
        f.write('  "architectures": [\n')
        for arch_idx, arch in enumerate(architectures):
            arch_comma = ',' if arch_idx < len(architectures) - 1 else ''
            f.write(f'    {{\n      "id": {arch["id"]},\n      "cnn_config": [\n')
            
            # Write CNN config
            for layer_idx, layer in enumerate(arch["cnn_config"]):
                layer_comma = ',' if layer_idx < len(arch["cnn_config"]) - 1 else ''
                f.write(f'        {{"channels": {layer["channels"]},"kernel": {layer["kernel"]},"stride": {layer["stride"]},"padding": {layer["padding"]}}}{layer_comma}\n')
            
            f.write('      ],\n')
            if arch.get("cnn_mlp_config"):
                f.write(f'      "cnn_mlp_config": [{",".join(map(str, arch["cnn_mlp_config"]))}],\n')
            if arch.get("state_mlp_config"):
                f.write(f'      "state_mlp_config": [{",".join(map(str, arch["state_mlp_config"]))}],\n')
            f.write(f'      "mlp_config": [{",".join(map(str, arch["mlp_config"]))}],\n')
            f.write(f'      "arch_descriptor": [{",".join(map(str, arch["arch_descriptor"]))}],\n')
            token_list = ",".join(f'"{token}"' for token in arch["token_descriptor"])
            f.write(f'      "token_descriptor": [{token_list}]\n')
            f.write(f'    }}{arch_comma}\n')
        
        f.write('  ],\n')
        
        # Write teacher model
        f.write('  "teacher_model": {\n')
        f.write(f'    "id": "{teacher_architecture["id"]}",\n')
        f.write(f'    "cnn_config": [\n')
        
        # Write teacher CNN config
        for layer_idx, layer in enumerate(teacher_architecture["cnn_config"]):
            layer_comma = ',' if layer_idx < len(teacher_architecture["cnn_config"]) - 1 else ''
            f.write(f'      {{"channels": {layer["channels"]},"kernel": {layer["kernel"]},"stride": {layer["stride"]},"padding": {layer["padding"]}}}{layer_comma}\n')
        
        f.write('    ],\n')
        if teacher_architecture.get("cnn_mlp_config"):
            f.write(f'    "cnn_mlp_config": [{",".join(map(str, teacher_architecture["cnn_mlp_config"]))}],\n')
        if teacher_architecture.get("state_mlp_config"):
            f.write(f'    "state_mlp_config": [{",".join(map(str, teacher_architecture["state_mlp_config"]))}],\n')
        f.write(f'    "mlp_config": [{",".join(map(str, teacher_architecture["mlp_config"]))}],\n')
        f.write(f'    "arch_descriptor": [{",".join(map(str, teacher_architecture["arch_descriptor"]))}],\n')
        teacher_token_list = ",".join(f'"{token}"' for token in teacher_architecture["token_descriptor"])
        f.write(f'    "token_descriptor": [{teacher_token_list}]\n')
        f.write('  }\n}')
    
    print(f"Saved {len(architectures)} architectures + teacher to {args.output}")
    print(f"GHN max_shape: {data['metadata']['ghn_config']['ghn_max_shape']}, descriptor_length: {FIXED_DESC_LENGTH}")
    print("####")
    
    print(f"Architecture generation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ALL possible CNN+MLP architectures for GHN training")
    parser.add_argument("--output", type=str, default="configs/architecture_go2_depth84_kernel3.json", help="Output JSON file path")
    parser.add_argument("--name", type=str, default="", help="Name to append to filename (e.g., --name rtx4090 creates file_rtx4090.json)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--input_size", type=int, default=84, help="Input image size (width/height)")
    parser.add_argument("--output_dim", type=int, default=12, help="Final output dimension (action space)")
    parser.add_argument("--state_dim", type=int, default=48, help="State vector dimension")
    parser.add_argument("--cnn_output_dim", type=int, default=128, help="CNN branch output dimension")
    parser.add_argument("--state_mlp_dim", type=int, default=128, help="State MLP branch output dimension")
    parser.add_argument("--cnn_layer_options", type=int, nargs="+", default=[3,4,5], help="CNN layer count options")
    parser.add_argument("--mlp_layer_options", type=int, nargs="+", default=[2], help="MLP layer count options")
    parser.add_argument("--cnn_channel_options", type=int, nargs="+", default=[8,16,32,64], help="CNN channel options")
    parser.add_argument("--cnn_kernel_options", type=int, nargs="+", default=[3], help="CNN kernel size options")
    parser.add_argument("--mlp_dim_options", type=int, nargs="+", default=[64,128, 256,512], help="MLP hidden dimension options")
    
    args = parser.parse_args()
    
    # Modify output filename if name is provided
    if args.name:
        # Split the filename and extension
        output_path = args.output
        if output_path.endswith('.json'):
            base_name = output_path[:-5]  # Remove .json
            args.output = f"{base_name}_{args.name}.json"
        else:
            args.output = f"{output_path}_{args.name}"
    
    main(args)