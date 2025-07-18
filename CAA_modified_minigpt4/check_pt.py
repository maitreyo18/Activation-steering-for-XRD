import torch

data = torch.load("loop_scattering_act_diff_12-20_5_diffuse_scattering.pt")
print(f"Outer keys: {list(data.keys())}")
print()

first_image_data = data[0]
print(f"Inner keys: {list(first_image_data.keys())}")
print()

for layer_key in first_image_data.keys():
    layer_data = first_image_data[layer_key]
    print(f"Layer {layer_key}:")
    print(f"  Type: {type(layer_data)}")
    print(f"  Length: {len(layer_data)}")
    if len(layer_data) > 0:
        print(f"  First element type: {type(layer_data[0])}")
        print(f"  First element shape: {layer_data[0].shape}")
        print(f"  Is torch tensor: {torch.is_tensor(layer_data[0])}")
    print()