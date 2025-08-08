"""
Binary Labels Activations Analysis
Analyze class-specific neurons from training mean SAE vectors
"""
import torch
import numpy as np
from collections import defaultdict

def create_binary_vectors(training_mean_sae_path, threshold=0.0001):
    """
    Create binary vectors from training mean SAE activations
    Args:
        training_mean_sae_path: path to training mean SAE .pt file
        threshold: threshold for binary conversion
    Returns:
        binary_vectors: dictionary with binary vectors for each class
        original_vectors: dictionary with original vectors
    """
    print("Loading training mean SAE vectors...")
    training_mean_sae = torch.load(training_mean_sae_path)
    
    print(f"Found {len(training_mean_sae)} anomaly classes:")
    for class_name in training_mean_sae.keys():
        print(f"  - {class_name}")
    
    binary_vectors = {}
    original_vectors = {}
    
    print(f"\nCreating binary vectors with threshold {threshold}...")
    
    for class_name, vector in training_mean_sae.items():
        # Store original vector
        original_vectors[class_name] = vector
        
        # Create binary vector: 1 if >= threshold, 0 if < threshold
        binary_vector = (vector >= threshold).float()
        binary_vectors[class_name] = binary_vector
        
        # Print statistics
        num_active = binary_vector.sum().item()
        total_neurons = len(binary_vector)
        print(f"{class_name}: {num_active}/{total_neurons} neurons active ({num_active/total_neurons*100:.2f}%)")
    
    return binary_vectors, original_vectors

def find_class_specific_neurons(binary_vectors):
    """
    Find neurons that are specific to each class (not overlapping with other classes)
    Args:
        binary_vectors: dictionary with binary vectors for each class
    Returns:
        class_specific_neurons: dictionary with class-specific neuron indices
        overlapping_neurons: dictionary with overlapping neuron indices
    """
    print("\n" + "="*80)
    print("ANALYZING CLASS-SPECIFIC NEURONS")
    print("="*80)
    
    class_names = list(binary_vectors.keys())
    class_specific_neurons = {}
    overlapping_neurons = {}
    
    # Convert to numpy for easier manipulation
    binary_arrays = {name: vec.numpy() for name, vec in binary_vectors.items()}
    
    for target_class in class_names:
        print(f"\n--- Analyzing {target_class} ---")
        
        # Get neurons active in target class
        target_active = np.where(binary_arrays[target_class] == 1)[0]
        print(f"Total active neurons in {target_class}: {len(target_active)}")
        
        # Find neurons that are NOT active in any other class
        class_specific = target_active.copy()
        
        for other_class in class_names:
            if other_class != target_class:
                other_active = np.where(binary_arrays[other_class] == 1)[0]
                # Remove neurons that are also active in other class
                class_specific = np.setdiff1d(class_specific, other_active)
        
        class_specific_neurons[target_class] = class_specific
        overlapping_neurons[target_class] = np.setdiff1d(target_active, class_specific)
        
        print(f"Class-specific neurons: {len(class_specific)}")
        
        if len(class_specific) > 0:
            print(f"Class-specific neuron indices: {class_specific.tolist()}")
        else:
            print("No class-specific neurons found!")
    
    return class_specific_neurons, overlapping_neurons

def analyze_neuron_overlaps(binary_vectors):
    """
    Analyze overlaps between different classes
    Args:
        binary_vectors: dictionary with binary vectors for each class
    """
    print("\n" + "="*80)
    print("NEURON OVERLAP ANALYSIS")
    print("="*80)
    
    class_names = list(binary_vectors.keys())
    binary_arrays = {name: vec.numpy() for name, vec in binary_vectors.items()}
    
    # Create overlap matrix
    overlap_matrix = {}
    
    for i, class1 in enumerate(class_names):
        overlap_matrix[class1] = {}
        active1 = np.where(binary_arrays[class1] == 1)[0]
        
        for j, class2 in enumerate(class_names):
            if i <= j:  # Only compute upper triangle
                active2 = np.where(binary_arrays[class2] == 1)[0]
                overlap = np.intersect1d(active1, active2)
                overlap_matrix[class1][class2] = len(overlap)
                
                if i != j:  # Not diagonal
                    print(f"{class1} ∩ {class2}: {len(overlap)} overlapping neurons")
    
    return overlap_matrix

def print_detailed_analysis(class_specific_neurons, overlapping_neurons, original_vectors):
    """
    Print detailed analysis of class-specific neurons
    Args:
        class_specific_neurons: dictionary with class-specific neuron indices
        overlapping_neurons: dictionary with overlapping neuron indices (not printed)
        original_vectors: dictionary with original activation vectors
    """
    print("\n" + "="*80)
    print("DETAILED NEURON ANALYSIS")
    print("="*80)
    
    for class_name in class_specific_neurons.keys():
        print(f"\n--- {class_name} ---")
        
        # Class-specific neurons
        specific_neurons = class_specific_neurons[class_name]
        if len(specific_neurons) > 0:
            print(f"Class-specific neurons ({len(specific_neurons)}):")
            print(f"  Indices: {specific_neurons.tolist()}")
            
            # Show activation values for class-specific neurons
            original_vector = original_vectors[class_name]
            specific_activations = original_vector[specific_neurons]
            print(f"  Activation values: {specific_activations.tolist()}")
            print(f"  Mean activation: {specific_activations.mean():.6f}")
            print(f"  Max activation: {specific_activations.max():.6f}")
        else:
            print("No class-specific neurons!")

def analyze_activation_distributions(original_vectors):
    """
    Analyze the distribution of activation values to suggest better thresholds
    Args:
        original_vectors: dictionary with original activation vectors
    """
    print("\n" + "="*80)
    print("ACTIVATION VALUE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    for class_name, vector in original_vectors.items():
        print(f"\n--- {class_name} ---")
        
        # Calculate percentiles
        percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]
        values = vector.numpy()
        
        print(f"Total neurons: {len(values)}")
        print(f"Mean activation: {values.mean():.8f}")
        print(f"Std activation: {values.std():.8f}")
        print(f"Min activation: {values.min():.8f}")
        print(f"Max activation: {values.max():.8f}")
        
        print("Percentiles:")
        for p in percentiles:
            percentile_val = np.percentile(values, p)
            num_above = (values >= percentile_val).sum()
            print(f"  {p}th percentile: {percentile_val:.8f} ({num_above} neurons above)")
        
        # Count neurons above different thresholds
        thresholds = [0.0001, 0.001, 0.01, 0.1, 1.0]
        print("Neurons above thresholds:")
        for threshold in thresholds:
            num_above = (values >= threshold).sum()
            print(f"  ≥ {threshold}: {num_above} neurons ({num_above/len(values)*100:.2f}%)")

def find_optimal_threshold(original_vectors, target_specific_ratio=0.1):
    """
    Find a threshold that gives a reasonable number of class-specific neurons
    Args:
        original_vectors: dictionary with original activation vectors
        target_specific_ratio: target ratio of class-specific neurons
    """
    print("\n" + "="*80)
    print("FINDING OPTIMAL THRESHOLD")
    print("="*80)
    
    # Test different thresholds
    thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    
    for threshold in thresholds:
        print(f"\n--- Testing threshold {threshold} ---")
        
        # Create binary vectors with this threshold
        binary_vectors = {}
        for class_name, vector in original_vectors.items():
            binary_vectors[class_name] = (vector >= threshold).float()
        
        # Find class-specific neurons
        class_specific_neurons, _ = find_class_specific_neurons(binary_vectors)
        
        # Calculate statistics
        total_specific = sum(len(neurons) for neurons in class_specific_neurons.values())
        total_neurons = len(list(original_vectors.values())[0])
        specific_ratio = total_specific / total_neurons
        
        print(f"Total class-specific neurons: {total_specific}")
        print(f"Specific neuron ratio: {specific_ratio:.4f} ({specific_ratio*100:.2f}%)")
        
        # Show breakdown by class
        for class_name, specific_neurons in class_specific_neurons.items():
            print(f"  {class_name}: {len(specific_neurons)} specific neurons")
        
        if specific_ratio > target_specific_ratio:
            print(f"✓ This threshold gives reasonable class-specific neurons!")
            return threshold
    
    print("No threshold found that gives sufficient class-specific neurons.")
    return None

def main():
    print("="*80)
    print("BINARY LABELS ACTIVATIONS ANALYSIS")
    print("="*80)
    
    # Path to training mean SAE vectors
    training_mean_sae_path = "SAE_reconstruct_plots_train/training_mean_sae.pt"
    
    # Create binary vectors with original threshold
    binary_vectors, original_vectors = create_binary_vectors(training_mean_sae_path, threshold=0.0001)
    
    # Analyze activation distributions
    analyze_activation_distributions(original_vectors)
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(original_vectors, target_specific_ratio=0.01)
    
    if optimal_threshold is not None:
        print(f"\nRecommended threshold: {optimal_threshold}")
        
        # Recreate binary vectors with optimal threshold
        binary_vectors, _ = create_binary_vectors(training_mean_sae_path, threshold=optimal_threshold)
    
    # Find class-specific neurons
    class_specific_neurons, overlapping_neurons = find_class_specific_neurons(binary_vectors)
    
    # Analyze overlaps
    overlap_matrix = analyze_neuron_overlaps(binary_vectors)
    
    # Print detailed analysis
    print_detailed_analysis(class_specific_neurons, overlapping_neurons, original_vectors)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_specific = sum(len(neurons) for neurons in class_specific_neurons.values())
    total_overlapping = sum(len(neurons) for neurons in overlapping_neurons.values())
    
    print(f"Total class-specific neurons: {total_specific}")
    print(f"Total overlapping neurons: {total_overlapping}")
    
    for class_name, specific_neurons in class_specific_neurons.items():
        print(f"{class_name}: {len(specific_neurons)} class-specific neurons")
    
    print("\n" + "="*80)
    print("Analysis completed!")
    print("="*80)

if __name__ == "__main__":
    main() 