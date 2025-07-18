import os
import torch
import numpy as np
import glob

def analyze_ice_ring_vectors(steer_layer=16):
    """
    Read all .pt files in the current directory, compute mean vectors and their norms
    """
    
    # Get all .pt files in the current directory
    pt_files = glob.glob("*.pt")
    
    print(f"=== Analyzing Ice Ring Steering Vectors (Layer {steer_layer}) ===")
    print(f"Found {len(pt_files)} .pt files:")
    for file in pt_files:
        print(f"  - {file}")
    print()
    
    results = {}
    
    for pt_file in pt_files:
        print(f"Processing: {pt_file}")
        try:
            # Load the steering data
            steering_data = torch.load(pt_file)
            
            # Extract steering vector for the specified layer
            steer_acts = []
            for idx in steering_data:
                if steer_layer in steering_data[idx]:
                    steer_acts.extend(steering_data[idx][steer_layer])
            
            if not steer_acts:
                print(f"  ✗ No steering vectors found for layer {steer_layer}")
                continue
            
            # Compute mean vector
            steer_acts = np.array(steer_acts)
            mean_vector = np.mean(steer_acts, axis=0)
            mean_tensor = torch.from_numpy(mean_vector)
            
            # Compute norm
            norm = torch.norm(mean_tensor, p=2)
            
            # Store results
            results[pt_file] = {
                'mean_vector': mean_tensor,
                'norm': norm,
                'num_vectors': len(steer_acts),
                'vector_shape': steer_acts.shape
            }
            
            print(f"  ✓ Loaded {len(steer_acts)} vectors")
            print(f"  ✓ Mean vector shape: {mean_vector.shape}")
            print(f"  ✓ Mean vector norm: {norm:.4f}")
            print()
            
        except Exception as e:
            print(f"  ✗ Error processing {pt_file}: {e}")
            print()
    
    # Print summary
    print("=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    if results:
        print(f"{'File':<50} {'Norm':<10} {'Num Vectors':<12}")
        print("-" * 60)
        
        for file, data in results.items():
            print(f"{file:<50} {data['norm']:<10.4f} {data['num_vectors']:<12}")
        
        # Compute overall statistics
        norms = [data['norm'] for data in results.values()]
        print("-" * 60)
        print(f"Mean norm across all files: {np.mean(norms):.4f}")
        print(f"Std norm across all files: {np.std(norms):.4f}")
        print(f"Min norm: {np.min(norms):.4f}")
        print(f"Max norm: {np.max(norms):.4f}")
        
        # Find files with min and max norms
        min_file = min(results.keys(), key=lambda x: results[x]['norm'])
        max_file = max(results.keys(), key=lambda x: results[x]['norm'])
        print(f"File with min norm: {min_file}")
        print(f"File with max norm: {max_file}")
        
    else:
        print("No valid steering vectors found!")
    
    return results

if __name__ == "__main__":
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Analyze vectors for layer 16 (you can change this)
    results = analyze_ice_ring_vectors(steer_layer=16) 