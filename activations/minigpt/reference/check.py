import torch

def inspect_pt_file(file_path):
    try:
        data = torch.load(file_path)
        print(f"File: {file_path}")
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            print("Keys:", data.keys())
            for key, value in data.items():
                print(f"  {key}: {type(value)}")
                if isinstance(value, dict):
                    print(f"    Subkeys: {value.keys()}")
                    for subkey, subvalue in value.items():
                        print(f"      {subkey}: {type(subvalue)}")
                        if isinstance(subvalue, list):
                            print(f"        Length: {len(subvalue)}")
                            if subvalue:
                                print(f"        First element type: {type(subvalue[0])}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# Inspect both files
inspect_pt_file("./reference_activations.pt")
inspect_pt_file("./reference_activations_original.pt")