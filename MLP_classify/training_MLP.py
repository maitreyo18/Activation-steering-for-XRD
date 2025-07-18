import torch
import argparse
from MLP_classify import MLPClassifier
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_path', type=str, default='./activation_train_4-10_MLP.pt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--save_path', type=str, default='mlp_classifier.pth')
    parser.add_argument('--layer', type=int, default=4, help='Layer key to use for activations')
    parser.add_argument(
        '--options',
        nargs='+',
        default=[],
        help='Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = torch.load(args.pt_path)
    # Sort by key to ensure order
    keys = sorted(data.keys())
    X = torch.stack([data[k][args.layer] for k in keys]).float()
    y = torch.tensor([data[k]['label'] for k in keys], dtype=torch.long)
    d_model = X.shape[1]
    num_classes = int(y.max().item()) + 1
    print(f"Loaded {len(X)} samples. d_model={d_model}, num_classes={num_classes}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = X.to(device)
    y = y.to(device)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = MLPClassifier(d_model, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

    # Save the model with epoch and lr in the filename
    save_name = f"mlp_classifier_layer{args.layer}_epoch{args.epochs}_lr{args.lr}.pth"
    torch.save(model.state_dict(), save_name)
    print(f"Saved trained model to {save_name}")

if __name__ == '__main__':
    main() 