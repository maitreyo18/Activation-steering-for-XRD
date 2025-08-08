"""
SAE Trainer for MiniGPT-4 Image Token Activations
Adapted from PatchSAE for MiniGPT-4 activations
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import os
from tqdm import tqdm
import random
from geom_median.torch import compute_geometric_median
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import einops

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Configuration (matching PatchSAE)
gpu_id = 0
training_gpu = f"cuda:{gpu_id}"
d_in = 5120
d_sae = d_in * 4  # 4x expansion like PatchSAE
l1_coefficient = 1e-3
learning_rate = 3e-4
total_training_tokens = 19938 * 150  # 2 * 19,938 activations
batch_size = 1024
feature_sampling_window = 2000
dead_feature_window = 1000
dead_feature_threshold = 1e-8
use_ghost_grads = True
feature_reinit_scale = 0.2
resample_batches = 32

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
        l1_loss = l1_coefficient * sparsity
        
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
    Load activations from a single .pt file and prepare for training
    Args:
        pt_file: path to the .pt file to load
    Returns: concatenated activations tensor
    """
    print(f"Loading activations from: {pt_file}")
    
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
    
    print(f"Total activations loaded: {activations_tensor.shape}")
    print(f"Activation shape: {activations_tensor.shape[1]}")
    print(f"Activation dtype: {activations_tensor.dtype}")
    
    return activations_tensor

def initialize_b_dec_with_geometric_median(sae, activations):
    """Initialize decoder bias with geometric median (matching PatchSAE)"""
    print("Initializing b_dec with geometric median of activations")
    previous_b_dec = sae.b_dec.clone().cpu()
    all_activations = activations.detach().cpu()
    
    out = compute_geometric_median(
        all_activations, skip_typechecks=True, maxiter=100, per_component=False
    ).median
    
    if len(out.shape) == 2:
        out = out.mean(dim=0)
    
    previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
    distances = torch.norm(all_activations - out, dim=-1)
    
    print(f"Previous distances: {previous_distances.median(0).values.mean().item()}")
    print(f"New distances: {distances.median(0).values.mean().item()}")
    
    sae.b_dec.data = torch.tensor(out, dtype=torch.float32, device=training_gpu)

def main():
    print("="*60)
    print("SAE Trainer for MiniGPT-4 Image Token Activations (PatchSAE Style)")
    print("="*60)
    
    # Check if GPU is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU.")
    
    print(f"Using GPU: {training_gpu}")
    print(f"Input dimension: {d_in}")
    print(f"SAE dimension: {d_sae} (4x expansion)")
    print(f"Using single precision (float32)")
    print(f"Batch size: {batch_size}")
    print(f"Total training tokens: {total_training_tokens}")
    print(f"L1 coefficient: {l1_coefficient}")
    print(f"Learning rate: {learning_rate}")
    
    # Specify the .pt file to load here
    pt_file = "./Layer_6_img_tokens.pt"  # Read .pt file
    
    if not os.path.exists(pt_file):
        raise FileNotFoundError(f"Required .pt file not found: {pt_file}")
    
    activations = load_activations_from_pt_file(pt_file)
    
    # Move to GPU
    activations = activations.to(training_gpu)
    
    # Calculate and print norm statistics for activations
    print(f"Calculating norm statistics for activations...")
    all_norms = []
    for i in range(activations.shape[0]):
        norm_val = torch.linalg.norm(activations[i]).item()
        all_norms.append(norm_val)
    
    norm_sum = sum(all_norms)
    norm_mean = norm_sum / len(all_norms)
    norm_highest = max(all_norms)
    norm_lowest = min(all_norms)
    
    print(f"Activation Statistics:")
    print(f"  Total number of samples: {len(all_norms)}")
    print(f"  Highest norm: {norm_highest:.6f}")
    print(f"  Lowest norm: {norm_lowest:.6f}")
    print(f"  Average norm: {norm_mean:.6f}")
    print(f"  Sum of all norms: {norm_sum:.6f}")
    
    # Initialize SAE
    sae = SAE(d_in=d_in, d_sae=d_sae)
    sae = sae.to(training_gpu)
    
    # Initialize decoder bias with geometric median (matching PatchSAE)
    initialize_b_dec_with_geometric_median(sae, activations)
    
    # Initialize optimizer (matching PatchSAE)
    optimizer = optim.Adam(sae.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    
    # Initialize learning rate schedule (constant like PatchSAE)
    def lr(epoch):
        return 1.0  # Constant learning rate
    
    scheduler = LambdaLR(optimizer, lr)
    
    # Create save directory
    save_dir = "./sae_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    # Training statistics (matching PatchSAE)
    act_freq_scores = torch.zeros(d_sae, device=training_gpu)
    n_forward_passes_since_fired = torch.zeros(d_sae, device=training_gpu)
    n_frac_active_tokens = 0
    n_training_tokens = 0
    n_training_steps = 0
    ghost_grad_neuron_mask = None
    
    print(f"Training for {total_training_tokens} tokens...")
    print(f"Total samples: 19,938")
    print(f"Batches per epoch: {activations.shape[0] // batch_size}")
    
    # Training loop (matching PatchSAE)
    pbar = tqdm(total=total_training_tokens, desc="Training SAE")
    
    try:
        while n_training_tokens < total_training_tokens:
            # Get batch of activations
            if n_training_tokens + batch_size <= activations.shape[0]:
                batch_indices = torch.randperm(activations.shape[0])[:batch_size]
                sae_acts = activations[batch_indices]
            else:
                # Handle last batch
                remaining = total_training_tokens - n_training_tokens
                batch_indices = torch.randperm(activations.shape[0])[:remaining]
                sae_acts = activations[batch_indices]
            
            n_training_tokens += sae_acts.size(0)
            
            # Training step
            optimizer.zero_grad()
            sae.train()
            sae.set_decoder_norm_to_unit_norm()
            
            # Log and reset feature sparsity every feature_sampling_window steps
            if (n_training_steps + 1) % feature_sampling_window == 0:
                print(f"Resetting sparsity stats at step {n_training_steps}")
                act_freq_scores = torch.zeros(d_sae, device=training_gpu)
                n_frac_active_tokens = 0
            
            # Ghost grad neuron mask
            ghost_grad_neuron_mask = (
                n_forward_passes_since_fired > dead_feature_window
            ).bool()
            
            # Forward pass
            sae_out, feature_acts, loss_dict = sae(sae_acts, ghost_grad_neuron_mask)
            
            # Update sparsity statistics (matching PatchSAE)
            with torch.no_grad():
                did_fire = (feature_acts > 0).float().sum(-2) > 0
                act_freq_scores += (feature_acts.abs() > 0).float().sum(0)
                n_forward_passes_since_fired += 1
                n_forward_passes_since_fired[did_fire] = 0
                n_frac_active_tokens += sae_out.size(0)
            
            # Backward pass
            loss_dict["loss"].backward()
            sae.remove_gradient_parallel_to_decoder_directions()
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Update progress bar (matching PatchSAE format)
            pbar.set_description(
                f"{n_training_steps}| MSE Loss {loss_dict['mse_loss'].item():.3f} | L1 {loss_dict['l1_loss'].item():.3f}"
            )
            pbar.update(sae_out.size(0))
            
            # Calculate and print metrics for each step
            # Calculate sparsity metrics
            feature_freq = act_freq_scores / n_frac_active_tokens
            dead_features = (feature_freq < dead_feature_threshold).float().mean().item()
            
            # Calculate performance metrics
            per_token_l2_loss = (sae_out - sae_acts).pow(2).sum(dim=-1).mean().squeeze()
            total_variance = sae_acts.pow(2).sum(-1).mean()
            explained_variance = 1 - per_token_l2_loss / total_variance
            l0 = (feature_acts > 0).float().sum(-1).mean()
            
            print(f"Step {n_training_steps}: MSE={loss_dict['mse_loss'].item():.4f}, "
                  f"L1={loss_dict['l1_loss'].item():.4f}, "
                  f"L0={l0.item():.2f}, "
                  f"Var={explained_variance.mean().item():.4f}, "
                  f"Dead={dead_features:.4f}")
            
            # Save checkpoint every 100,000 tokens
            if n_training_tokens % 20000 == 0 and n_training_tokens > 0:
                checkpoint_path = os.path.join(save_dir, f"sae_{n_training_tokens}_tokens.pth")
                torch.save({
                    'n_training_tokens': n_training_tokens,
                    'model_state_dict': sae.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss_dict["loss"].item(),
                    'mse': loss_dict["mse_loss"].item(),
                    'l1': loss_dict["l1_loss"].item(),
                    'config': {
                        'd_in': d_in,
                        'd_sae': d_sae,
                        'l1_coefficient': l1_coefficient,
                        'learning_rate': learning_rate
                    }
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
            
            n_training_steps += 1
            
    finally:
        print("Saving final checkpoint")
        final_path = os.path.join(save_dir, f"sae_final_l1_{l1_coefficient}_150.pth")
        torch.save({
            'model_state_dict': sae.state_dict(),
            'config': {
                'd_in': d_in,
                'd_sae': d_sae,
                'l1_coefficient': l1_coefficient,
                'learning_rate': learning_rate
            }
        }, final_path)
    
    pbar.close()
    
    print(f"\nTraining completed!")
    print(f"Final model saved to: {final_path}")
    print(f"Model architecture: {d_in} → {d_sae} → {d_in}")
    print(f"Total parameters: {sum(p.numel() for p in sae.parameters()):,}")
    print(f"Using single precision (float32) throughout")

if __name__ == "__main__":
    main() 