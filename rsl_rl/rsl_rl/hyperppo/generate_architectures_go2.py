#!/usr/bin/env python3

import argparse
import os
import random
import torch
import json
from src.model import CnnMlpNetwork


def get_cpu_name():
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':')[1].strip()
    except:
        pass
    return "Unknown CPU"

def get_gpu_name():
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except:
            return "Unknown CUDA GPU"
    return None

def get_hardware_name():
    gpu_name = get_gpu_name()
    return gpu_name if gpu_name else get_cpu_name()

def calculate_cnn_output_size(cnn_config, input_size):
    size = input_size
    for layer in cnn_config:
        size = (size + 2 * layer['padding'] - layer['kernel']) // layer['stride'] + 1
    channels = cnn_config[-1]['channels'] if cnn_config else 1  # 1 channel for depth
    return size * size * channels

def calculate_complexity_score(cnn_config, mlp_config):
    total_depth = len(cnn_config) + len(mlp_config)
    cnn_params = 0
    prev_channels = 1  # 1 channel for depth input
    for layer in cnn_config:
        channels = layer['channels']
        kernel = layer['kernel'] 
        cnn_params += (prev_channels * kernel * kernel + 1) * channels
        prev_channels = channels
    
    mlp_params = sum(mlp_config)
    avg_channels = sum(layer['channels'] for layer in cnn_config) / len(cnn_config) if cnn_config else 0
    
    return (total_depth * 1000000 + cnn_params * 100 + mlp_params * 10 + avg_channels * 1)

def main(args):
    random.seed(args.seed)
    hardware_name = get_hardware_name()
    print(f"Hardware: {hardware_name}")
    
    from itertools import product
    
    # Calculate total number of architectures before generation
    total_architectures = 0
    for cnn_layers in args.cnn_layer_options:
        # Calculate increasing channel combinations for this depth
        channel_combos = 0
        for channels in product(args.cnn_channel_options, repeat=cnn_layers):
            if all(channels[i] <= channels[i+1] for i in range(len(channels)-1)):
                channel_combos += 1
        
        kernel_combos = len(args.cnn_kernel_options) ** cnn_layers
        
        for mlp_layers in args.mlp_layer_options:
            mlp_combos = len(args.mlp_dim_options) ** mlp_layers
            total_architectures += channel_combos * kernel_combos * mlp_combos
    
    print(f"Total architectures to generate: {total_architectures}")
    print("-" * 50)
    
    all_architectures = []
    arch_id = 0
    
    for cnn_layers in args.cnn_layer_options:
        for mlp_layers in args.mlp_layer_options:
            kernel_configs = list(product(args.cnn_kernel_options, repeat=cnn_layers))
            
            increasing_channel_configs = []
            for channels in product(args.cnn_channel_options, repeat=cnn_layers):
                if all(channels[i] <= channels[i+1] for i in range(len(channels)-1)):
                    increasing_channel_configs.append(channels)
            
            mlp_layer_configs = list(product(args.mlp_dim_options, repeat=mlp_layers))
            
            for channels_combo in increasing_channel_configs:
                for kernels_combo in kernel_configs:
                    for mlp_combo in mlp_layer_configs:
                        cnn_config = []
                        for i in range(cnn_layers):
                            cnn_config.append({
                                "channels": channels_combo[i],
                                "kernel": kernels_combo[i],
                                "stride": 2, 
                                "padding": 1
                            })
                        
                        actual_cnn_output_size = calculate_cnn_output_size(cnn_config, args.input_size)
                        if actual_cnn_output_size <= 0:
                            continue
                        
                        cnn_mlp_config = [args.cnn_output_dim]
                        state_mlp_config = [args.state_mlp_dim]
                        first_mlp_input = args.cnn_output_dim + args.state_mlp_dim
                        mlp_config = [first_mlp_input] + list(mlp_combo)
                        
                        arch_descriptor = []
                        for layer in cnn_config:
                            arch_descriptor.extend([layer["channels"], layer["kernel"]])
                        arch_descriptor.extend(mlp_config)
                        
                        token_descriptor = []
                        for layer in cnn_config:
                            token_descriptor.append(f"{layer['channels']}ch{layer['kernel']}k")
                        for dim in mlp_config:
                            token_descriptor.append(f"{dim}mlp")
                        
                        complexity = calculate_complexity_score(cnn_config, mlp_config)
                        
                        arch = {
                            "id": arch_id, 
                            "cnn_config": cnn_config,
                            "cnn_mlp_config": cnn_mlp_config,
                            "state_mlp_config": state_mlp_config,
                            "mlp_config": mlp_config, 
                            "arch_descriptor": arch_descriptor,
                            "token_descriptor": token_descriptor,
                            "complexity": complexity,
                            "cnn_output_size": actual_cnn_output_size
                        }
                        all_architectures.append(arch)
                        arch_id += 1
    
    all_architectures.sort(key=lambda x: x["complexity"])
    
    print(f"Validating {len(all_architectures)} architectures...")
    architectures = []
    invalid_count = 0
    
    for idx, arch in enumerate(all_architectures):
        if (idx + 1) % 100 == 0:
            print(f"Progress: {idx + 1}/{len(all_architectures)} architectures validated...")
        
        try:
            test_network = CnnMlpNetwork(
                cnn_config=arch['cnn_config'],
                cnn_mlp_config=arch['cnn_mlp_config'],
                mlp_config=arch['mlp_config'],
                state_mlp_config=arch['state_mlp_config'],
                state_dim=args.state_dim,
                input_channels=1,  # 1 channel for depth image
                output_dim=args.output_dim,
                input_size=args.input_size
            )
            
            dummy_image = torch.randn(1, 1, args.input_size, args.input_size)  # 1 channel depth
            dummy_state = torch.randn(1, args.state_dim)
            with torch.no_grad():
                _ = test_network(dummy_image, dummy_state)
                
            architectures.append(arch)
            del test_network, dummy_image, dummy_state, _
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception:
            invalid_count += 1
            continue
    
    print("-" * 50)
    print(f"Validation complete!")
    print(f"  Valid architectures: {len(architectures)}")
    print(f"  Invalid architectures: {invalid_count}")
    print(f"  Total processed: {len(all_architectures)}")
    
    teacher_architecture = all_architectures[-1].copy()
    teacher_architecture["id"] = "teacher"
    del teacher_architecture["complexity"]
    
    for i, arch in enumerate(architectures):
        arch["id"] = i
        del arch["complexity"]
    
    FIXED_DESC_LENGTH = 16
    for arch in architectures:
        while len(arch["arch_descriptor"]) < FIXED_DESC_LENGTH:
            arch["arch_descriptor"].append(0)
        arch["arch_descriptor"] = arch["arch_descriptor"][:FIXED_DESC_LENGTH]
        
        while len(arch["token_descriptor"]) < FIXED_DESC_LENGTH:
            arch["token_descriptor"].append("")
        arch["token_descriptor"] = arch["token_descriptor"][:FIXED_DESC_LENGTH]
    
    while len(teacher_architecture["arch_descriptor"]) < FIXED_DESC_LENGTH:
        teacher_architecture["arch_descriptor"].append(0)
    teacher_architecture["arch_descriptor"] = teacher_architecture["arch_descriptor"][:FIXED_DESC_LENGTH]
    
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
    
    data = {
        "metadata": {
            "input_channels": 1,  # 1 channel for depth images 
            "input_size": args.input_size, 
            "output_dim": args.output_dim,
            "state_dim": args.state_dim,
            "cnn_output_dim": args.cnn_output_dim,
            "state_mlp_dim": args.state_mlp_dim,
            "hardware_name": hardware_name,
            "ghn_config": {
                "ghn_max_shape": [512, 512, 3, 3],  # Max MLP dim 512, kernel size 3x3
                "simple_classification": True
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
    
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("-" * 50)
    print(f"Architecture generation complete!")
    print(f"  Output file: {args.output}")
    print(f"  Total architectures saved: {len(architectures)} + 1 teacher")
    print(f"  Architecture distribution:")
    
    # Count architectures by CNN depth
    cnn_depth_counts = {}
    for arch in architectures:
        depth = len(arch['cnn_config'])
        cnn_depth_counts[depth] = cnn_depth_counts.get(depth, 0) + 1
    
    for depth in sorted(cnn_depth_counts.keys()):
        print(f"    CNN depth {depth}: {cnn_depth_counts[depth]} architectures")
    
    print(f"  Teacher architecture: {len(teacher_architecture['cnn_config'])} CNN + {len(teacher_architecture['mlp_config'])-1} MLP layers")
    print(f"  GHN max shape: {data['metadata']['ghn_config']['ghn_max_shape']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ALL possible CNN+MLP architectures for GHN training")
    parser.add_argument("--output", type=str, default="configs_go2/architecture_go2_depth84.json", help="Output JSON file path")
    parser.add_argument("--name", type=str, default="", help="Name to append to filename (e.g., --name rtx4090 creates file_rtx4090.json)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--input_size", type=int, default=84, help="Input image size (width/height)")
    parser.add_argument("--output_dim", type=int, default=12, help="Final output dimension (action space)")
    parser.add_argument("--state_dim", type=int, default=48, help="State vector dimension")
    parser.add_argument("--cnn_output_dim", type=int, default=256, help="CNN branch output dimension")
    parser.add_argument("--state_mlp_dim", type=int, default=256, help="State MLP branch output dimension")
    parser.add_argument("--cnn_layer_options", type=int, nargs="+", default=[3,4], help="CNN layer count options - optimized for depth")
    parser.add_argument("--mlp_layer_options", type=int, nargs="+", default=[2], help="MLP layer count options - 2 layers work best")
    parser.add_argument("--cnn_channel_options", type=int, nargs="+", default=[16,32,48,64,96,128], help="CNN channel options - diverse range for depth features")
    parser.add_argument("--cnn_kernel_options", type=int, nargs="+", default=[3], help="CNN kernel size options - 3x3 works best for 84x84 depth")
    parser.add_argument("--mlp_dim_options", type=int, nargs="+", default=[256,384,512], help="MLP hidden dimension options - larger dims for better capacity")
    
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
    
    # Manipulation tasks
    #'PickCube-v1': {'state_dim': 29, 'output_dim': 8},
    #'PushCube-v1': {'state_dim': 25, 'output_dim': 8},
    #'StackCube-v1': {'state_dim': 25, 'output_dim': 8},  
    # PushT-v1 : {'state_dim': 21, 'output_dim': 7},  
    