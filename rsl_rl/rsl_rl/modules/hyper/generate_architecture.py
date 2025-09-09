#!/usr/bin/env python3
"""
Diverse Architecture Generator for GHN Training

Creates diverse CNN+MLP architecture configurations for training GHN with variety.
"""

import argparse
import os
import random

def calculate_cnn_output_size(cnn_config, input_size):
    """Calculate the flattened CNN output size for a given configuration"""
    size = input_size
    for layer in cnn_config:
        kernel = layer['kernel']
        stride = layer['stride']
        padding = layer['padding']
        size = (size + 2 * padding - kernel) // stride + 1
    
    # Return flattened size: height * width * channels
    channels = cnn_config[-1]['channels'] if cnn_config else 3
    return size * size * channels

def calculate_complexity_score(cnn_config, mlp_config):
    """Calculate architecture complexity score for curriculum ordering"""
    # Primary: Total depth (CNN + MLP layers) - most important for GHN stability
    total_depth = len(cnn_config) + len(mlp_config)
    
    # Secondary: CNN parameter count (most expensive computationally)
    cnn_params = 0
    prev_channels = 3  # RGB input
    for layer in cnn_config:
        channels = layer['channels']
        kernel = layer['kernel'] 
        # Conv params: (in_channels * kernel^2 + 1) * out_channels
        cnn_params += (prev_channels * kernel * kernel + 1) * channels
        prev_channels = channels
    
    # Tertiary: MLP parameter count
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

def main():
    parser = argparse.ArgumentParser(description="Generate diverse CNN+MLP architectures for GHN training")
    parser.add_argument('--cnn_channel_options', type=int, nargs='+', default=[96], help='CNN channel options (default: [96])')
    parser.add_argument('--cnn_kernel_options', type=int, nargs='+', default=[3], help='CNN kernel options (default: [3])')
    parser.add_argument('--cnn_layer_options', type=int, nargs='+', default=[3,4,5], help='CNN layer options (default: [5])')
    parser.add_argument('--mlp_dim_options', type=int, nargs='+', default=[256], help='MLP dimension options (default: [256])')
    parser.add_argument('--mlp_layer_options', type=int, nargs='+', default=[3,4], help='MLP layer options (default: [3])')
    parser.add_argument('--cnn_output_dim', type=int, default=256, help='Fixed CNN output dimension after flattening+MLP (default: 256)')
    # State processing removed - pure image to actions
    parser.add_argument('--num_architectures', type=int, default=16, help='Number of architectures to generate (default: 1)')
    parser.add_argument('--input_size', type=int, default=64, help='Input image size (default: 64)')
    parser.add_argument('--output_dim', type=int, default=8, help='Output dimension (default: 8)')
    parser.add_argument('--output', type=str, default='configs/architecture_img64_image_only.json', help='Output file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()
    
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print(f"🔧 Generating {args.num_architectures} diverse CNN+MLP architectures:")
    print(f"   CNN channels: {args.cnn_channel_options}, kernels: {args.cnn_kernel_options}, layers: {args.cnn_layer_options}")
    print(f"   MLP dims: {args.mlp_dim_options}, layers: {args.mlp_layer_options}")
    print(f"   Flow: RGB({args.input_size}x{args.input_size}x3) -> CNN -> Flatten -> MLP -> Actions({args.output_dim})")
    
    # Create architectures with independent layer dimensions
    architectures = []
    max_channels = max(args.cnn_channel_options)
    max_kernel = max(args.cnn_kernel_options)
    max_desc_length = 0
    
    # Generate ALL possible unique architectures systematically
    print(f"🎯 Generating all possible unique architectures systematically...")
    
    # Calculate total possible unique architectures
    total_unique = 0
    for cnn_layers in args.cnn_layer_options:
        cnn_combinations = len(args.cnn_channel_options) * len(args.cnn_kernel_options)
        cnn_unique = cnn_combinations ** cnn_layers
        for mlp_layers in args.mlp_layer_options:
            mlp_unique = len(args.mlp_dim_options) ** mlp_layers
            total_unique += cnn_unique * mlp_unique
    
    # Check if we should sample randomly or generate all
    if args.num_architectures >= total_unique:
        print(f"⚠️  Requested {args.num_architectures} architectures, but only {total_unique} unique combinations possible!")
        print(f"🔧 Using all unique architectures: {total_unique}")
        args.num_architectures = total_unique
        sample_randomly = False
    else:
        sample_randomly = True
        print(f"🎲 Will randomly sample {args.num_architectures} architectures from {total_unique} possible combinations")
    
    # Generate all unique combinations systematically, ordered by complexity (smallest first)
    from itertools import product
    
    print(f"🔄 Generating all unique combinations and sorting by complexity...")
    
    # Generate ALL possible architectures first, then sort by complexity
    all_architectures = []
    arch_id = 0
    
    for cnn_layers in args.cnn_layer_options:
        for mlp_layers in args.mlp_layer_options:
            # Generate all CNN layer combinations for this depth
            cnn_layer_configs = list(product(
                *[list(product(args.cnn_channel_options, args.cnn_kernel_options)) for _ in range(cnn_layers)]
            ))
            
            # Generate all MLP layer combinations for this depth  
            mlp_layer_configs = list(product(args.mlp_dim_options, repeat=mlp_layers))
            
            # Combine CNN and MLP configurations
            for cnn_combo in cnn_layer_configs:
                for mlp_combo in mlp_layer_configs:
                    # Create CNN config with variable layers only (no fixed CNN layer)
                    cnn_config = []
                    for channels, kernel in cnn_combo:
                        cnn_config.append({
                            "channels": channels,
                            "kernel": kernel,
                            "stride": 2, 
                            "padding": 1
                        })
                    
                    # Calculate actual CNN output size for this architecture
                    actual_cnn_output_size = calculate_cnn_output_size(cnn_config, args.input_size)
                    
                    # Pure image-to-actions: CNN → flatten → CNN_MLP → MLP → actions
                    # Store actual CNN output size so GHN-2 can generate correct weight shapes
                    cnn_mlp_config = [args.cnn_output_dim]  # Output dimension after CNN MLP
                    
                    # MLP takes CNN MLP output
                    first_mlp_input = args.cnn_output_dim
                    print(f"Architecture {arch_id}: CNN(PPUDA-calculated→{args.cnn_output_dim}) → MLP")
                    
                    # Regular MLP config: [cnn_output_dim, hidden1, hidden2, ...]
                    mlp_config = [first_mlp_input] + list(mlp_combo)
                    
                    # Create descriptor: [ch1, k1, ch2, k2, ..., mlp1, mlp2, ...]
                    arch_descriptor = []
                    for layer in cnn_config:
                        arch_descriptor.extend([layer["channels"], layer["kernel"]])
                    arch_descriptor.extend(mlp_config)
                    
                    # Track max descriptor length for padding
                    max_desc_length = max(max_desc_length, len(arch_descriptor))
                    
                    # Calculate complexity score for sorting
                    complexity = calculate_complexity_score(cnn_config, mlp_config)
                    
                    arch = {
                        "id": arch_id, 
                        "cnn_config": cnn_config,
                        "cnn_mlp_config": cnn_mlp_config,
                        "mlp_config": mlp_config, 
                        "arch_descriptor": arch_descriptor,
                        "complexity": complexity,
                        "cnn_output_size": actual_cnn_output_size  # Store for GHN-2
                    }
                    all_architectures.append(arch)
                    arch_id += 1
    
    print(f"📊 Generated {len(all_architectures)} total unique architectures")
    
    # Sort by complexity (smallest first)
    all_architectures.sort(key=lambda x: x["complexity"])
    print(f"🔢 Complexity range: {all_architectures[0]['complexity']:.0f} - {all_architectures[-1]['complexity']:.0f}")
    
    # Select architectures based on sampling strategy
    if sample_randomly:
        # Random sampling with seed for reproducibility
        architectures = random.sample(all_architectures, args.num_architectures)
        print(f"🎲 Randomly sampled {len(architectures)} architectures from {len(all_architectures)} total")
    else:
        # Take the first N smallest architectures
        architectures = all_architectures[:args.num_architectures]
        print(f"✂️  Selected {len(architectures)} smallest architectures")
    
    # Get the biggest architecture for teacher model (last in sorted list)
    teacher_complexity = all_architectures[-1]['complexity']  # Store before deletion
    teacher_architecture = all_architectures[-1].copy()  # Biggest complexity
    teacher_architecture["id"] = "teacher"  # Special ID for teacher
    del teacher_architecture["complexity"]
    
    # Remove complexity field and reassign sequential IDs for regular architectures
    for i, arch in enumerate(architectures):
        arch["id"] = i
        del arch["complexity"]
    
    print(f"🎓 Teacher model: {len(teacher_architecture['cnn_config'])} CNN layers, {len(teacher_architecture['mlp_config'])} MLP layers (complexity: {teacher_complexity:.0f})")
    
    # Pad all descriptors to fixed length (16) for GHN compatibility
    # Format: [ch1,k1,ch2,k2,ch3,k3,ch4,k4,fixed_mlp_dim,mlp1,mlp2,mlp3,mlp4,0] - padded with zeros
    # Note: fixed_mlp_dim represents the fixed first MLP layer after flattening
    FIXED_DESC_LENGTH = 16
    # Pad regular architectures
    for arch in architectures:
        # Pad to fixed length
        while len(arch["arch_descriptor"]) < FIXED_DESC_LENGTH:
            arch["arch_descriptor"].append(0)
        # Truncate if too long (shouldn't happen with current config)
        arch["arch_descriptor"] = arch["arch_descriptor"][:FIXED_DESC_LENGTH]
    
    # Pad teacher architecture descriptor  
    while len(teacher_architecture["arch_descriptor"]) < FIXED_DESC_LENGTH:
        teacher_architecture["arch_descriptor"].append(0)
    teacher_architecture["arch_descriptor"] = teacher_architecture["arch_descriptor"][:FIXED_DESC_LENGTH]
    
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
            "input_channels": 3, "input_size": args.input_size, "output_dim": args.output_dim,
            "cnn_output_dim": args.cnn_output_dim,
            "ghn_config": {
                "ghn_max_shape": [unified_max_dim, unified_max_dim, unified_max_kernel, unified_max_kernel],  # HyperPPO-style ConvDecoder
                "simple_classification": True  # All weights → cls_w, simplified approach
            },
            "arch_descriptor_config": {
                "total_length": FIXED_DESC_LENGTH,
                "arch_descriptor_dim": FIXED_DESC_LENGTH,
                "description": "Format: [ch1,k1,ch2,k2,ch3,k3,ch4,k4,mlp_input,mlp1,mlp2,mlp3] - mlp_input=CNN_output"
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
        f.write(f'    "input_channels": 3,\n')
        f.write(f'    "input_size": {args.input_size},\n')
        f.write(f'    "output_dim": {args.output_dim},\n')
        f.write(f'    "cnn_output_dim": {args.cnn_output_dim},\n')
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
            f.write(f'      "mlp_config": [{",".join(map(str, arch["mlp_config"]))}],\n')
            f.write(f'      "arch_descriptor": [{",".join(map(str, arch["arch_descriptor"]))}]\n')
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
        f.write(f'    "mlp_config": [{",".join(map(str, teacher_architecture["mlp_config"]))}],\n')
        f.write(f'    "arch_descriptor": [{",".join(map(str, teacher_architecture["arch_descriptor"]))}]\n')
        f.write('  }\n}')
    
    print(f"✅ Saved {len(architectures)} architectures + 1 teacher model to {args.output}")
    print(f"📊 Simplified GHN ConvDecoder Max Shape:")
    print(f"   Shape: {data['metadata']['ghn_config']['ghn_max_shape']} [max_dim, max_dim, kernel, kernel]")
    print(f"   Simplified approach: All weights → cls_w (HyperPPO style)")
    print(f"   Descriptor length: {FIXED_DESC_LENGTH}")
    print(f"🎯 Architecture samples: {[(arch['id'], len(arch['cnn_config']), len(arch['mlp_config'])) for arch in architectures[:3]]}{'...' if len(architectures) > 3 else ''}")
    print(f"🎓 Teacher model details: CNN layers={len(teacher_architecture['cnn_config'])}, MLP layers={len(teacher_architecture['mlp_config'])}")

if __name__ == "__main__":
    main()