import torch
import os

def visualize_pt_file(file_path):
    """Visualize the structure of a .pt file"""
    print(f"\n{'='*60}")
    print(f"VISUALIZING: {file_path}")
    print(f"{'='*60}")
    
    # Load the .pt file
    data = torch.load(file_path)
    
    print(f"Type: {type(data)}")
    print(f"Number of outer keys: {len(data)}")
    
    # Show outer keys
    print(f"\nOuter keys: {list(data.keys())}")
    
    # Show structure of first few entries
    print(f"\nStructure of first 3 entries:")
    for i, (outer_key, inner_dict) in enumerate(data.items()):
        if i >= 3:  # Only show first 3
            break
            
        print(f"\n  Outer key {outer_key}:")
        print(f"    Type: {type(inner_dict)}")
        print(f"    Inner keys: {list(inner_dict.keys())}")
        
        # Show details of each inner key
        for inner_key, value in inner_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"      {inner_key}: tensor shape {value.shape}, dtype {value.dtype}")
                print(f"        Sample values: {value[:5]}...")  # First 5 values
            else:
                print(f"      {inner_key}: {type(value)} = {value}")
    
    # Print norm of first five vectors from inner keys
    print(f"\nNorm of first five vectors from inner keys:")
    vector_count = 0
    for outer_key, inner_dict in data.items():
        for inner_key, value in inner_dict.items():
            if isinstance(value, torch.Tensor) and inner_key != 'correct_answer':
                norm_val = torch.linalg.norm(value).item()
                print(f"  Outer key {outer_key}, Inner key {inner_key}: norm = {norm_val:.6f}")
                vector_count += 1
                if vector_count >= 5:  # Only print first 5 vectors
                    break
        if vector_count >= 5:
            break
    
    # Calculate norm of all vector entries
    print(f"\nCalculating norms of all vector entries...")
    all_norms = []
    total_vectors = 0
    
    for outer_key, inner_dict in data.items():
        for inner_key, value in inner_dict.items():
            if isinstance(value, torch.Tensor) and inner_key != 'correct_answer':
                norm_val = torch.linalg.norm(value).item()
                all_norms.append(norm_val)
                total_vectors += 1
    
    if all_norms:
        highest_norm = max(all_norms)
        lowest_norm = min(all_norms)
        sum_of_norms = sum(all_norms)
        mean_norm = sum_of_norms / len(all_norms)
        
        print(f"\nNorm Statistics:")
        print(f"  Total vectors: {total_vectors}")
        print(f"  Highest norm: {highest_norm:.6f}")
        print(f"  Lowest norm: {lowest_norm:.6f}")
        print(f"  Mean norm: {mean_norm:.6f}")
        print(f"  Sum of all norms: {sum_of_norms:.6f}")
    else:
        print(f"\nNo vector entries found!")
    
    # Show summary statistics
    print(f"\nSummary:")
    print(f"  Total outer keys: {len(data)}")
    
    # Count inner keys
    inner_keys = set()
    tensor_shapes = set()
    for outer_key, inner_dict in data.items():
        inner_keys.update(inner_dict.keys())
        for inner_key, value in inner_dict.items():
            if isinstance(value, torch.Tensor):
                tensor_shapes.add(value.shape)
    
    print(f"  Unique inner keys: {sorted(inner_keys)}")
    print(f"  Tensor shapes found: {tensor_shapes}")
    
    # Show sample of correct answers
    print(f"\nSample correct answers:")
    for i, (outer_key, inner_dict) in enumerate(data.items()):
        if i < 5:  # Show first 5
            correct_answer = inner_dict.get('correct_answer', 'N/A')
            print(f"  Entry {outer_key}: {correct_answer}")

def main():
    # Check if .pt files exist
    layer_6_file = "./Layer_6_img_tokens.pt"
    layer_16_file = "./Layer_16_img_tokens.pt"
    
    if os.path.exists(layer_6_file):
        visualize_pt_file(layer_6_file)
    else:
        print(f"File not found: {layer_6_file}")
    
    if os.path.exists(layer_16_file):
        visualize_pt_file(layer_16_file)
    else:
        print(f"File not found: {layer_16_file}")

if __name__ == "__main__":
    main() 