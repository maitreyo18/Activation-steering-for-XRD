import torch
import os

input_dir = "./"
output_file = "activation_xrd_img_tokens_12-24_merged.pt"

# Get sorted list of .pt files
pt_files = sorted(
    [f for f in os.listdir(input_dir) if f.startswith("diff_act_bgm_ring_12-24_") and f.endswith(".pt")],
    key=lambda x: int(x.split("_")[-1].split(".")[0])
)

merged = {}

# Outer key index
outer_index = 0

for pt_file in pt_files:
    path = os.path.join(input_dir, pt_file)
    data = torch.load(path, map_location='cpu')
    
    for orig_outer in data:
        merged[outer_index] = {}
        for inner_key in data[orig_outer]:
            merged[outer_index][inner_key] = data[orig_outer][inner_key]
        outer_index += 1

# Save merged output
torch.save(merged, output_file)
print(f"Merged {len(pt_files)} files into '{output_file}' with {outer_index} outer keys.")
