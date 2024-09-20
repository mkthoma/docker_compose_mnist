import json
import torch
import torch.nn.functional as F
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from model import Net  # Import the model


def test_epoch(model, data_loader, device):
    """Function to evaluate the model for one epoch."""
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0

    # Disable gradient computation during evaluation
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Sum batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Calculate average loss and accuracy
    test_loss /= len(data_loader.dataset)
    accuracy = 100.0 * correct / len(data_loader.dataset)

    # Print and return the results as a dictionary
    out = {"Test loss": test_loss, "Accuracy": accuracy}
    print(out)
    return out


def main():
    parser = argparse.ArgumentParser(description="MNIST Evaluation Script")

    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--checkpoint", default="/opt/mount/model/mnist_cnn.pt", help="path to the saved checkpoint"
    )
    parser.add_argument(
        "--save-dir", default="/opt/mount", help="directory where evaluation results will be saved"
    )
    parser.add_argument(
        "--device", default="cpu", help="Device to run the evaluation on (default: cpu)"
    )

    args = parser.parse_args()

    # Set the correct device (CPU or CUDA)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # DataLoader settings
    kwargs = {
        "batch_size": args.test_batch_size,
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": False,  # No need to shuffle for evaluation
    }

    # Data transformation and loading for testing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('/opt/mount/data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, **kwargs)

    # Initialize the model
    model = Net().to(device)

    # Load the model checkpoint
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_file():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"Checkpoint not found at {args.checkpoint}!")
        exit(1)

    # Perform the evaluation
    eval_results = test_epoch(model, test_loader, device)

    # Save the evaluation results to JSON
    eval_results_path = Path(args.save_dir) / "model/eval_results.json"
    eval_results_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    with eval_results_path.open("w") as f:
        json.dump(eval_results, f)
    print(f"Evaluation results saved to {eval_results_path}")


if __name__ == "__main__":
    main()
