import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openTSNE import TSNE
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
import torch
import os

# Load Arial Narrow font
font_path = '/home/biswasm/Arial_Narrow/arialnarrow.ttf'
font_prop = FontProperties(fname=font_path)

# Set font globally
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial Narrow', 'DejaVu Sans', 'Liberation Sans']

def load_activation_data(file_path):
    """Load activation data from .pt file"""
    data = torch.load(file_path)
    
    # Extract data
    image_tokens_activations = []
    final_token_activations = []
    correct_answers = []
    
    for idx in range(len(data)):
        image_tokens_activations.append(data[idx]['image_tokens'].numpy())
        final_token_activations.append(data[idx]['final_token'].numpy())
        correct_answers.append(data[idx]['correct_answer'])
    
    return np.array(image_tokens_activations), np.array(final_token_activations), correct_answers

def create_tsne_plot(activations, labels, title, filename, layer_num):
    """Create t-SNE plot for given activations"""
    
    # Run t-SNE
    tsne = TSNE(
        perplexity=12,
        #n_components=2,
        random_state=42,
        n_iter=3000,
        initialization="pca",
        metric="cosine"
    )
    
    tsne_results = tsne.fit(activations)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'tsne_x': tsne_results[:, 0],
        'tsne_y': tsne_results[:, 1],
        'correct_answer': labels
    })
    
    # Define colors for the 4 groups
    answer_colors = {
        'Ice ring.': '#FF9999',      # Light red
        'Loop scattering.': '#99CCFF', # Light blue
        'Non-uniform detector.': '#99CC99', # Light green
        'Background ring.': '#FFCC99'  # Light orange
    }
    
    # Create figure with clean style
    plt.figure(figsize=(12, 7))
    sns.set_style("white")
    
    # Plot points with colors based on correct answer
    for answer in answer_colors.keys():
        mask = plot_df['correct_answer'] == answer
        if mask.any():
            plt.scatter(
                plot_df.loc[mask, 'tsne_x'],
                plot_df.loc[mask, 'tsne_y'],
                marker='o',
                color=answer_colors[answer],
                s=100,
                alpha=0.7,
                edgecolors='none',
                label=answer
            )
    
    # Create legend with proper font size - outside the plot
    legend = plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=True,
        edgecolor='lightgrey',
        fontsize=18
    )
    
    # Manually set font properties for legend text
    for text in legend.get_texts():
        text.set_fontproperties(font_prop)
        text.set_fontsize(18)
    
    # Add subtle border
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color('#cccccc')
        spine.set_linewidth(0.8)
    
    # Set axis labels
    plt.xlabel('Dimension 1', fontsize=26, fontproperties=font_prop)
    plt.ylabel('Dimension 2', fontsize=26, fontproperties=font_prop)
    plt.title(title, fontsize=26, fontproperties=font_prop)
    
    # Make tick labels larger - use tick_params for better control
    plt.tick_params(axis='both', which='major', labelsize=19)
    
    # Alternative method: manually set tick labels with font properties
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(22)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(22)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()

def main():
    # Create plots directory
    plots_dir = './plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get all .pt files in the current directory
    pt_files = [f for f in os.listdir('.') if f.startswith('activations_layer_') and f.endswith('.pt')]
    pt_files.sort()  # Sort to process in order
    
    print(f"Found {len(pt_files)} activation files:")
    for f in pt_files:
        print(f"  - {f}")
    
    # Process each layer file
    for layer_file in pt_files:
        # Extract layer number from filename
        layer_num = layer_file.split('_')[2].split('.')[0]
        
        print(f"\nProcessing {layer_file} (Layer {layer_num})...")
        
        if os.path.exists(layer_file):
            # Load activation data
            image_tokens_acts, final_token_acts, correct_answers = load_activation_data(layer_file)
            
            print(f"  Loaded {len(image_tokens_acts)} samples")
            print(f"  Image tokens activation shape: {image_tokens_acts.shape}")
            print(f"  Final token activation shape: {final_token_acts.shape}")
            print(f"  Unique correct answers: {set(correct_answers)}")
            
            # Create t-SNE plot for image tokens
            create_tsne_plot(
                activations=image_tokens_acts,
                labels=correct_answers,
                title=f'Layer {layer_num}: Image Tokens Activation t-SNE',
                filename=f'{plots_dir}/layer_{layer_num}_img_tokens.png',
                layer_num=layer_num
            )
            
            # Create t-SNE plot for final token
            create_tsne_plot(
                activations=final_token_acts,
                labels=correct_answers,
                title=f'Layer {layer_num}: Final Token Activation t-SNE',
                filename=f'{plots_dir}/layer_{layer_num}_final_token.png',
                layer_num=layer_num
            )
            
            print(f"  Plots saved: layer_{layer_num}_img_tokens.png, layer_{layer_num}_final_token.png")
            
        else:
            print(f"  File {layer_file} not found!")
    
    print(f"\nAll plots saved in {plots_dir}/")
    print(f"Total plots created: {len(pt_files) * 2}")

if __name__ == "__main__":
    main()