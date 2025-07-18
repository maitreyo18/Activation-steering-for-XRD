import torch
import argparse
from MLP_classify import MLPClassifier
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_path', type=str, default='./activation_test_4-10_MLP.pt')
    parser.add_argument('--model_path', type=str, default='./mlp_classifier_layer4_epoch250_lr0.0005.pth', help='Path to trained .pth model')
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
    keys = sorted(data.keys())
    X = torch.stack([data[k][args.layer] for k in keys]).float()
    y = torch.tensor([data[k]['label'] for k in keys], dtype=torch.long)
    d_model = X.shape[1]
    num_classes = int(y.max().item()) + 1

    print(f"X shape: {X.shape}")
    print(f"Number of classes: {num_classes}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = X.to(device)
    y = y.to(device)

    model = MLPClassifier(d_model, num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model successfully loaded...")

    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        total_correct = 0
        for i in range(len(X)):
            print("="*100)
            print(f"Sample {i}: True label: {y[i].item()}")
            print(f"Probabilities: {probs[i].cpu().numpy()}")
            print(f"Predicted: {preds[i].item()}")
            if preds[i].item() == y[i].item():
                total_correct += 1
        accuracy = total_correct / len(X)
        print(f"\nAccuracy: {total_correct}/{len(X)} = {accuracy*100:.2f}%")

if __name__ == '__main__':
    main() 