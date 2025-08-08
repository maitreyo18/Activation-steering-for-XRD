"""
Global Interpretation Analysis for SAE on MiniGPT-4 Activations
Following PatchSAE methodology for global image interpretation
"""
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tqdm import tqdm
from geom_median.torch import compute_geometric_median
import einops
from collections import defaultdict

# ============================================================================
# CONFIGURATION - EDIT THESE THREE LINES
# ============================================================================
training_activations_path = "./training_activations/Layer_6_img_tokens.pt"  # Path to training activations .pt file
sae_path = "./sae_checkpoints/sae_final_l1_0.001_150.pth"  # Path to trained SAE .pth file
batch_size = 32  # Batch size for processing
# ============================================================================

# Font configuration
font_path = '/home/biswasm/Arial_Narrow/arialnarrow.ttf'
font_prop = FontProperties(fname=font_path)

# Set matplotlib to use Arial Narrow
plt.rcParams['font.family'] = 'Arial Narrow'
plt.rcParams['font.size'] = 16

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration (matching your sae_trainer.py)
gpu_id = 0
training_gpu = f"cuda:{gpu_id}"
d_in = 5120
d_sae = d_in * 4  # 4x expansion like PatchSAE
l1_coefficient = 1e-3  # Add L1 coefficient to match sae_trainer.py
use_ghost_grads = True  # Add ghost grads flag to match sae_trainer.py

class SAE(nn.Module):
    def __init__(self, d_in=5120, d_sae=20480):
        super(SAE, self).__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        
        # Encoder weights and bias (matching PatchSAE)
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_in, self.d_sae, dtype=torch.float32, device=training_gpu)
            )
        )
        self.b_enc = nn.Parameter(
            torch.zeros(self.d_sae, dtype=torch.float32, device=training_gpu)
        )
        
        # Decoder weights and bias (matching PatchSAE)
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_sae, self.d_in, dtype=torch.float32, device=training_gpu)
            )
        )
        self.b_dec = nn.Parameter(
            torch.zeros(self.d_in, dtype=torch.float32, device=training_gpu)
        )
        
        # Initialize decoder weights to unit norm (matching PatchSAE)
        with torch.no_grad():
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
    
    def forward(self, x, dead_neuron_mask=None):
        x = x.to(torch.float32)
        
        # Remove decoder bias as per Anthropic/PatchSAE
        sae_in = x - self.b_dec
        
        # Encoder: sae_in @ W_enc + b_enc (using einops like PatchSAE)
        hidden_pre = einops.einsum(
            sae_in,
            self.W_enc,
            "... d_in, d_in d_sae -> ... d_sae",
        ) + self.b_enc
        feature_acts = torch.nn.functional.relu(hidden_pre)
        
        # Decoder: feature_acts @ W_dec + b_dec (using einops like PatchSAE)
        sae_out = einops.einsum(
            feature_acts,
            self.W_dec,
            "... d_sae, d_sae d_in -> ... d_in",
        ) + self.b_dec
        
        # MSE Loss (L2 normalized like PatchSAE)
        mse_loss = torch.pow((sae_out - x.float()), 2) / (x**2).sum(dim=-1, keepdim=True).sqrt()
        
        # Ghost grads for dead neurons (matching PatchSAE exactly)
        mse_loss_ghost_resid = torch.tensor(0.0, dtype=torch.float32, device=training_gpu)
        if use_ghost_grads and self.training and dead_neuron_mask is not None and dead_neuron_mask.sum() > 0:
            # Ghost protocol from PatchSAE
            residual = x - sae_out
            l2_norm_residual = torch.norm(residual, dim=-1)
            
            # Handle both 2D and 3D tensors like PatchSAE
            if len(hidden_pre.size()) == 3:
                feature_acts_dead_neurons_only = torch.exp(
                    hidden_pre[:, :, dead_neuron_mask]
                )
            else:
                feature_acts_dead_neurons_only = torch.exp(
                    hidden_pre[:, dead_neuron_mask]
                )
            ghost_out = feature_acts_dead_neurons_only @ self.W_dec[dead_neuron_mask, :]
            l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
            norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)
            if len(hidden_pre.size()) == 3:
                ghost_out = ghost_out * norm_scaling_factor[:, :, None].detach()
            else:
                ghost_out = ghost_out * norm_scaling_factor[:, None].detach()
            
            mse_loss_ghost_resid = torch.pow((ghost_out - residual.detach().float()), 2) / (residual.detach()**2).sum(dim=-1, keepdim=True).sqrt()
            mse_rescaling_factor = (mse_loss / (mse_loss_ghost_resid + 1e-6)).detach()
            mse_loss_ghost_resid = mse_rescaling_factor * mse_loss_ghost_resid
        
        mse_loss_ghost_resid = mse_loss_ghost_resid.mean()
        mse_loss = mse_loss.mean()
        
        # L1 Loss (sparsity) - matching PatchSAE
        sparsity = torch.abs(feature_acts).sum(dim=-1).mean(dim=(0,))
        l1_loss = l1_coefficient * sparsity  # Use L1 coefficient like sae_trainer.py
        
        # Total loss
        loss = mse_loss + l1_loss + mse_loss_ghost_resid
        
        loss_dict = {
            "mse_loss": mse_loss,
            "l1_loss": l1_loss.mean(),
            "mse_loss_ghost_resid": mse_loss_ghost_resid,
            "loss": loss.mean(),
        }
        
        return sae_out, feature_acts, loss_dict
    
    def set_decoder_norm_to_unit_norm(self):
        """Set decoder weights to unit norm (matching PatchSAE)"""
        with torch.no_grad():
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
    
    def remove_gradient_parallel_to_decoder_directions(self):
        """Remove gradient components parallel to decoder directions (matching PatchSAE exactly)"""
        if self.W_dec.grad is not None:
            parallel_component = einops.einsum(
                self.W_dec.grad,
                self.W_dec.data,
                "d_sae d_in, d_sae d_in -> d_sae",
            )
            
            self.W_dec.grad -= einops.einsum(
                parallel_component,
                self.W_dec.data,
                "d_sae, d_sae d_in -> d_sae d_in",
            )

def load_activations_from_pt_file(pt_file):
    """
    Load activations from .pt file and organize by anomaly type
    Args:
        pt_file: path to the .pt file to load
    Returns: dictionary organized by anomaly type
    """
    print(f"Loading test activations from: {pt_file}")
    
    # Load the .pt file
    data = torch.load(pt_file)
    
    print(f"Data loaded: {len(data)} samples")
    
    # Organize by anomaly type
    anomaly_activations = defaultdict(list)
    
    for idx, sample_data in data.items():
        if 'correct_answer' in sample_data:
            anomaly_type = sample_data['correct_answer']
            
            # Extract all patch activations (1-32) from the sample
            sample_patches = []
            for patch_num in range(1, 33):  # Patches 1-32
                if patch_num in sample_data:
                    activation = sample_data[patch_num]
                    if activation is not None:
                        sample_patches.append(activation)
            
            if sample_patches:
                # Stack patches for this sample
                sample_tensor = torch.stack(sample_patches)
                anomaly_activations[anomaly_type].append(sample_tensor)
    
    # Convert to tensors and print statistics
    for anomaly_type, activations_list in anomaly_activations.items():
        if activations_list:
            activations_tensor = torch.stack(activations_list)
            anomaly_activations[anomaly_type] = activations_tensor.float()
            print(f"Anomaly type '{anomaly_type}': {activations_tensor.shape} "
                  f"({activations_tensor.shape[0]} images, {activations_tensor.shape[1]} patches per image)")
    
    return anomaly_activations

def load_sae_model(sae_path):
    """
    Load trained SAE model from checkpoint
    Args:
        sae_path: path to the saved SAE .pth file
    Returns: loaded SAE model
    """
    print(f"Loading SAE model from: {sae_path}")
    
    # Load checkpoint
    checkpoint = torch.load(sae_path, map_location="cpu")
    
    # Extract config
    config = checkpoint["config"]
    d_in = config["d_in"]
    d_sae = config["d_sae"]
    
    print(f"SAE config - d_in: {d_in}, d_sae: {d_sae}")
    
    # Create SAE model
    sae = SAE(d_in=d_in, d_sae=d_sae)
    sae = sae.to(training_gpu)
    
    # Load state dict
    sae.load_state_dict(checkpoint["model_state_dict"])
    sae.eval()
    
    print(f"SAE model loaded successfully")
    return sae

def compute_sae_statistics(sae_activations):
    """
    Compute sum activations and sparsity statistics for SAE features (PatchSAE style)
    Args:
        sae_activations: tensor of shape [num_images, num_patches, d_sae]
    Returns:
        sum_acts: sum of activations across patches for each feature
        sparsity: number of patches where each feature is active (> 0)
    """
    # Sum across patches for each feature (PatchSAE aggregation)
    sum_acts = sae_activations.sum(dim=1)  # [num_images, d_sae]
    
    # Count patches where each feature is active (> 0 for patch-level)
    sparsity = (sae_activations > 0).sum(dim=1)  # [num_images, d_sae]
    
    return sum_acts, sparsity

def process_anomaly_batch(sae, anomaly_activations, anomaly_type, batch_size=32):
    """
    Process activations for a specific anomaly type
    Args:
        sae: trained SAE model
        anomaly_activations: activations for this anomaly type
        anomaly_type: name of the anomaly type
        batch_size: batch size for processing
    Returns:
        global_feature_stats: aggregated feature statistics
    """
    print(f"Processing anomaly type: {anomaly_type}")
    
    # Move activations to GPU
    anomaly_activations = anomaly_activations.to(training_gpu)
    
    # Calculate total number of batches
    num_batches = (len(anomaly_activations) + batch_size - 1) // batch_size
    
    # Storage for aggregated statistics
    all_sum_acts = []
    all_sparsity = []
    
    print(f"Processing {num_batches} batches for {anomaly_type}...")
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc=f"Processing {anomaly_type}"):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(anomaly_activations))
            
            # Get batch of activations
            batch_activations = anomaly_activations[start_idx:end_idx]
            
            # Print processing each image in the batch
            for img_idx in range(len(batch_activations)):
                print(f"Processing image {start_idx + img_idx + 1}/{len(anomaly_activations)} for {anomaly_type}")
            
            # Forward pass through SAE
            sae_out, feature_acts, loss_dict = sae(batch_activations)
            
            # Compute statistics (PatchSAE style)
            batch_sum_acts, batch_sparsity = compute_sae_statistics(feature_acts)
            
            all_sum_acts.append(batch_sum_acts)
            all_sparsity.append(batch_sparsity)
    
    # Print global processing message after processing all images
    print(f"Done processing {len(anomaly_activations)} images for {anomaly_type}. Doing global processing...")
    
    # Concatenate all batches
    all_sum_acts = torch.cat(all_sum_acts, dim=0)  # [num_images, d_sae]
    all_sparsity = torch.cat(all_sparsity, dim=0)    # [num_images, d_sae]
    
    # Aggregate across images (global interpretation)
    global_sum_acts = all_sum_acts.sum(dim=0)  # [d_sae] - total feature strength
    global_sparsity = all_sparsity.float().sum(dim=0)    # [d_sae] - total feature coverage
    
    return {
        'sum_acts': global_sum_acts.cpu().numpy(),
        'sparsity': global_sparsity.cpu().numpy(),
        'all_sum_acts': all_sum_acts.cpu().numpy(),
        'all_sparsity': all_sparsity.cpu().numpy()
    }

def create_neuron_firing_histogram(anomaly_stats, anomaly_type, save_dir):
    """
    Create histogram showing neuron firing patterns for an anomaly type
    Args:
        anomaly_stats: statistics for this anomaly type
        anomaly_type: name of the anomaly type
        save_dir: directory to save the plot
    """
    sum_acts = anomaly_stats['sum_acts']
    
    # Create figure with larger size
    plt.figure(figsize=(18, 10))
    
    # Create bar chart with prominent blue color
    neuron_indices = np.arange(len(sum_acts))
    bars = plt.bar(neuron_indices, sum_acts, alpha=1.0, color='#0066CC', width=18.0)
    
    # Find top neurons (top 60) based on sum activations
    top_60_indices = np.argsort(sum_acts)[-60:][::-1]
    top_60_values = sum_acts[top_60_indices]
    
    # For plotting, use only top 10 neurons with red markers
    top_10_indices = top_60_indices[:10]
    top_10_values = top_60_values[:10]
    
    # Mark top 10 neurons with red dots - use plot instead of scatter for exact positioning
    for idx, val in zip(top_10_indices, top_10_values):
        plt.plot(idx, val, 'o', color='red', markersize=10, zorder=5, 
                markeredgecolor='red', markeredgewidth=0)
    
    # Alternative method - if above doesn't work, try this:
    # plt.scatter(top_indices, top_values, color='red', s=100, zorder=5, 
    #            marker='o', edgecolors='none', clip_on=False)
    
    # Customize plot with much larger font sizes
    plt.title(f'Image-level SAE latent activations - {anomaly_type}', 
              fontsize=32, fontweight='bold', fontproperties=font_prop, pad=20)
    plt.xlabel('Neuron Index', fontsize=28, fontproperties=font_prop, labelpad=15)
    plt.ylabel('Sum Activation Value', fontsize=28, fontproperties=font_prop, labelpad=15)
    
    # Set tick parameters with much larger font sizes
    plt.tick_params(axis='both', which='major', labelsize=24, width=2, length=6)
    ax = plt.gca()
    
    # Explicitly set font properties for all tick labels
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(24)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(24)
    
    # Make tick marks more visible
    ax.tick_params(axis='x', which='major', pad=10)
    ax.tick_params(axis='y', which='major', pad=10)
    
    # Remove grid lines
    plt.grid(False)
    
    # Set x-axis limits to remove any padding
    plt.xlim(0, len(sum_acts))
    
    # Set y-axis to start exactly at 0 with no padding
    plt.ylim(0, None)
    
    # Adjust layout with more padding
    plt.tight_layout(pad=3.0)
    
    # Save plot with higher DPI
    clean_anomaly_type = anomaly_type.rstrip('.')  # Remove trailing period
    save_path = os.path.join(save_dir, f'neuron_firing_{clean_anomaly_type.replace(" ", "_")}_plot.png')
    plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved histogram for {anomaly_type} to {save_path}")
    
    # Print top 120 neurons
    print(f"\nTop 120 neurons for {anomaly_type}:")
    
    # Calculate how many images each neuron was fired in
    all_sum_acts = anomaly_stats['all_sum_acts']  # [num_images, d_sae]
    total_images = all_sum_acts.shape[0]
    fired_counts = (all_sum_acts > 1e-06).sum(axis=0)  # [d_sae] - count of images where each neuron fired (> 1e-06 threshold)
    
    # Get top 120 neurons for this class
    top_120_indices = np.argsort(sum_acts)[-120:][::-1]
    
    for i, idx in enumerate(top_120_indices):
        val = sum_acts[idx]
        fired_count = fired_counts[idx]
        print(f"  {i+1}. Neuron {idx}: {val:.4f} ({fired_count}/{total_images})")

def main():
    print("="*80)
    print("Global Interpretation Analysis for SAE on MiniGPT-4 Activations")
    print("Following PatchSAE methodology")
    print("="*80)
    
    # Check if GPU is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU.")
    
    print(f"Using GPU: {training_gpu}")
    print(f"Input dimension: {d_in}")
    print(f"SAE dimension: {d_sae} (4x expansion)")
    print(f"Batch size: {batch_size}")
    
    # Create save directory
    save_dir = "SAE_reconstruct_plots_train"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Plots will be saved to: {save_dir}")
    
    # Load test activations organized by anomaly type
    anomaly_activations = load_activations_from_pt_file(training_activations_path)
    
    # Load trained SAE model
    sae = load_sae_model(sae_path)
    
    # Process each anomaly type
    print("\n" + "="*80)
    print("Processing Global Interpretations")
    print("="*80)
    
    all_anomaly_stats = {}
    
    for anomaly_type, activations in anomaly_activations.items():
        if len(activations) > 0:
            print(f"\nProcessing {anomaly_type} ({len(activations)} images)...")
            
            # Process this anomaly type
            anomaly_stats = process_anomaly_batch(sae, activations, anomaly_type, batch_size)
            all_anomaly_stats[anomaly_type] = anomaly_stats
            
            # Create histogram
            create_neuron_firing_histogram(anomaly_stats, anomaly_type, save_dir)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for anomaly_type, stats in all_anomaly_stats.items():
        sum_acts = stats['sum_acts']
        sparsity = stats['sparsity']
        
        print(f"\n{anomaly_type}:")
        print(f"  Total feature strength: {sum_acts.sum():.6f}")
        print(f"  Max feature strength: {sum_acts.max():.6f}")
        print(f"  Total feature coverage: {sparsity.sum():.2f} patches")
        print(f"  Total active features: {(sum_acts > 0).sum()}")
        print(f"  Sparsity ratio: {1.0 - (sum_acts > 0).mean():.4f}")
    
    # Calculate and save mean SAE activations for each anomaly class
    print("\n" + "="*80)
    print("MEAN SAE ACTIVATIONS BY ANOMALY CLASS")
    print("="*80)
    
    mean_sae_activations = {}
    
    for anomaly_type, stats in all_anomaly_stats.items():
        # Get the number of samples in this anomaly class
        num_samples = len(anomaly_activations[anomaly_type])
        
        # Calculate mean of summed activations (sum across patches and images, then divide by num_samples)
        mean_acts = stats['sum_acts'] / num_samples
        
        # Convert to torch.float32 tensor
        mean_acts_tensor = torch.tensor(mean_acts, dtype=torch.float32)
        
        mean_sae_activations[anomaly_type] = mean_acts_tensor
        
        print(f"\n{anomaly_type} (mean across {num_samples} samples):")
        print(f"  Shape: {mean_acts_tensor.shape}")
        print(f"  Mean activation: {mean_acts_tensor.mean():.6f}")
        print(f"  Max activation: {mean_acts_tensor.max():.6f}")
        print(f"  Min activation: {mean_acts_tensor.min():.6f}")
        print(f"  First 10 values: {mean_acts_tensor[:10]}")
    
    # Save the mean activations as .pt file
    save_path = os.path.join(save_dir, "training_mean_sae.pt")
    torch.save(mean_sae_activations, save_path)
    print(f"\nMean SAE activations saved to: {save_path}")
    
    print("\n" + "="*80)
    print("Analysis completed successfully!")
    print(f"All plots saved to: {save_dir}")
    print("="*80)

if __name__ == "__main__":
    main() 