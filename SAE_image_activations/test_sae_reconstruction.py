"""
Test SAE Reconstruction on Test Images
Closely aligned with PatchSAE methodology and sae_trainer.py structure
"""
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import argparse
from geom_median.torch import compute_geometric_median
import einops

# ============================================================================
# CONFIGURATION - EDIT THESE THREE LINES
# ============================================================================
test_activations_path = "./test_activations/Layer_6_img_tokens_test.pt"  # Path to test activations .pt file
sae_path = "./sae_checkpoints/sae_final_l1_0.005_200.pth"  # Path to trained SAE .pth file
batch_size = 32  # Batch size for processing
# ============================================================================

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
    Load activations from a single .pt file and prepare for testing
    Args:
        pt_file: path to the .pt file to load
    Returns: concatenated activations tensor
    """
    print(f"Loading test activations from: {pt_file}")
    
    # Load the .pt file
    data = torch.load(pt_file)
    
    print(f"Data loaded: {len(data)} samples")
    
    # Extract all patch activations (1-32) from the data
    all_activations = []
    
    for idx in data:
        for patch_num in range(1, 33):  # Patches 1-32
            if patch_num in data[idx]:
                activation = data[idx][patch_num]
                if activation is not None:
                    all_activations.append(activation)
    
    # Stack all activations and convert to float32
    activations_tensor = torch.stack(all_activations)
    activations_tensor = activations_tensor.float()  # Convert to float32
    
    print(f"Total test activations loaded: {activations_tensor.shape}")
    print(f"Activation shape: {activations_tensor.shape[1]}")
    print(f"Activation dtype: {activations_tensor.dtype}")
    
    return activations_tensor

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

def calculate_rmse(original, reconstructed):
    """
    Calculate RMSE between original and reconstructed activations
    Args:
        original: original activations tensor
        reconstructed: reconstructed activations tensor
    Returns: RMSE value
    """
    # Calculate MSE first
    mse = torch.mean((original - reconstructed) ** 2)
    
    # Calculate RMSE
    rmse = torch.sqrt(mse)
    
    return rmse.item()

def test_sae_reconstruction(sae, test_activations, batch_size=32):
    """
    Test SAE reconstruction on test activations
    Args:
        sae: trained SAE model
        test_activations: test activations tensor
        batch_size: batch size for processing
    Returns: reconstruction metrics
    """
    print(f"Testing SAE reconstruction on {len(test_activations)} activations")
    print(f"Batch size: {batch_size}")
    
    # Move activations to GPU
    test_activations = test_activations.to(training_gpu)
    
    # Calculate total number of batches
    num_batches = (len(test_activations) + batch_size - 1) // batch_size
    
    # Metrics storage
    all_rmse = []
    all_mse_loss = []
    all_l1_loss = []
    all_reconstruction_loss = []
    
    print(f"Processing {num_batches} batches...")
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Testing SAE Reconstruction"):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(test_activations))
            
            # Get batch of activations
            batch_activations = test_activations[start_idx:end_idx]
            
            # Forward pass through SAE
            sae_out, feature_acts, loss_dict = sae(batch_activations)
            
            # Print feature activations sparsity for each patch in the batch
            print(f"Batch {batch_idx + 1}/{num_batches}:")
            for patch_idx in range(feature_acts.shape[0]):
                patch_sparsity = feature_acts[patch_idx].nonzero().shape[0]
                total_features = feature_acts[patch_idx].numel()
                sparsity_ratio = 1.0 - (patch_sparsity / total_features)
                print(f"  Patch {patch_idx + 1}: feature_acts.nonzero().shape = {feature_acts[patch_idx].nonzero().shape}, "
                      f"Sparsity = {sparsity_ratio:.4f} ({patch_sparsity}/{total_features} non-zero)")
            
            # Calculate RMSE for this batch
            batch_rmse = calculate_rmse(batch_activations, sae_out)
            all_rmse.append(batch_rmse)
            
            # Store loss metrics
            all_mse_loss.append(loss_dict["mse_loss"].item())
            all_l1_loss.append(loss_dict["l1_loss"].item())
            all_reconstruction_loss.append(loss_dict["loss"].item())
            
            # Print batch results
            print(f"Batch {batch_idx + 1}/{num_batches}: RMSE = {batch_rmse:.6f}, "
                  f"MSE Loss = {loss_dict['mse_loss'].item():.6f}, "
                  f"L1 Loss = {loss_dict['l1_loss'].item():.6f}")
    
    # Calculate overall metrics
    overall_rmse = np.mean(all_rmse)
    overall_mse_loss = np.mean(all_mse_loss)
    overall_l1_loss = np.mean(all_l1_loss)
    overall_reconstruction_loss = np.mean(all_reconstruction_loss)
    
    # Calculate statistics
    rmse_std = np.std(all_rmse)
    rmse_min = np.min(all_rmse)
    rmse_max = np.max(all_rmse)
    
    return {
        "overall_rmse": overall_rmse,
        "rmse_std": rmse_std,
        "rmse_min": rmse_min,
        "rmse_max": rmse_max,
        "overall_mse_loss": overall_mse_loss,
        "overall_l1_loss": overall_l1_loss,
        "overall_reconstruction_loss": overall_reconstruction_loss,
        "all_rmse": all_rmse,
        "all_mse_loss": all_mse_loss,
        "all_l1_loss": all_l1_loss,
        "all_reconstruction_loss": all_reconstruction_loss
    }

def main():
    print("="*60)
    print("SAE Reconstruction Test on Test Images")
    print("Closely aligned with PatchSAE methodology")
    print("="*60)
    
    # Check if GPU is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU.")
    
    print(f"Using GPU: {training_gpu}")
    print(f"Input dimension: {d_in}")
    print(f"SAE dimension: {d_sae} (4x expansion)")
    print(f"Batch size: {batch_size}")
    
    # Load test activations
    test_activations = load_activations_from_pt_file(test_activations_path)
    
    # Load trained SAE model
    sae = load_sae_model(sae_path)
    
    # Test SAE reconstruction
    print("\n" + "="*60)
    print("Testing SAE Reconstruction")
    print("="*60)
    
    metrics = test_sae_reconstruction(sae, test_activations, batch_size)
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL RECONSTRUCTION RESULTS")
    print("="*60)
    print(f"Overall RMSE: {metrics['overall_rmse']:.6f}")
    print(f"RMSE Std Dev: {metrics['rmse_std']:.6f}")
    print(f"RMSE Min: {metrics['rmse_min']:.6f}")
    print(f"RMSE Max: {metrics['rmse_max']:.6f}")
    print(f"Overall MSE Loss: {metrics['overall_mse_loss']:.6f}")
    print(f"Overall L1 Loss: {metrics['overall_l1_loss']:.6f}")
    print(f"Overall Reconstruction Loss: {metrics['overall_reconstruction_loss']:.6f}")
    print(f"Total test activations: {len(test_activations)}")
    print(f"Total patches processed: {len(test_activations)}")
    
    # Print RMSE distribution
    print(f"\nRMSE Distribution:")
    print(f"  Mean ± Std: {metrics['overall_rmse']:.6f} ± {metrics['rmse_std']:.6f}")
    print(f"  Range: [{metrics['rmse_min']:.6f}, {metrics['rmse_max']:.6f}]")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main() 