import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from tqdm import tqdm

# ---------------------------
# Device Utility
# ---------------------------
def get_device() -> torch.device:
    """Returns the best available device (cuda if available, else cpu)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------
# MLP Classifier Architecture
# ---------------------------
class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron for multi-class classification.
    Architecture: [d_model] -> [1024, ReLU] -> [512, ReLU] -> [256, ReLU] -> [num_classes, softmax]
    """
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x

# ---------------------------
# Dataset Class
# ---------------------------
class ActivationDataset(Dataset):
    """
    Dataset for activations and labels.
    activations: [N, d_model]
    labels: [N] (integer class indices)
    """
    def __init__(self, activations: torch.Tensor, labels: torch.Tensor):
        self.acts = activations
        self.labels = labels

    def __len__(self) -> int:
        return len(self.acts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.acts[idx], self.labels[idx]

# ---------------------------
# Training Function
# ---------------------------
def train_mlp_classifier(
    activations: torch.Tensor,
    labels: torch.Tensor,
    d_model: int,
    num_classes: int,
    batch_size: int = 32,
    num_epochs: int = 30,
    lr: float = 1e-3,
    device: torch.device = None,
    verbose: bool = True
) -> MLPClassifier:
    """
    Trains an MLP classifier on the given activations and labels.
    Args:
        activations: torch.Tensor of shape [N, d_model]
        labels: torch.Tensor of shape [N]
        d_model: int, input feature dimension
        num_classes: int, number of output classes
        batch_size: int, batch size
        num_epochs: int, number of epochs
        lr: float, learning rate for Adam optimizer
        device: torch.device, device to use
        verbose: bool, print progress if True
    Returns:
        Trained MLPClassifier model.
    """
    if device is None:
        device = get_device()
    activations = activations.to(device)
    labels = labels.to(device)

    dataset = ActivationDataset(activations, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLPClassifier(d_model=d_model, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        total_loss = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        if verbose:
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model

def save_mlp_classifier(model: MLPClassifier, path: str):
    """
    Saves the state_dict of a trained MLPClassifier to the specified path.
    Args:
        model: Trained MLPClassifier instance.
        path: File path to save the model weights (e.g., 'mlp_classifier.pth').
    """
    torch.save(model.state_dict(), path)


def load_mlp_classifier(path: str, d_model: int, num_classes: int, device: torch.device = None) -> MLPClassifier:
    """
    Loads an MLPClassifier from a saved state_dict.
    Args:
        path: File path to the saved model weights.
        d_model: Input feature dimension (must match the original model).
        num_classes: Number of output classes (must match the original model).
        device: Device to load the model onto (default: best available).
    Returns:
        An MLPClassifier instance with loaded weights.
    """
    if device is None:
        device = get_device()
    model = MLPClassifier(d_model, num_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model 