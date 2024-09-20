import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import torch.multiprocessing as mp
from model import Net  # Ensure the model is imported


def train(rank, args, model, dataset, dataloader_kwargs):
    """Training code for MNIST using Hogwild with multiprocessing."""
    torch.manual_seed(args.seed + rank)  # Set seed per process for reproducibility

    # Create DataLoader for training
    train_loader = DataLoader(dataset, **dataloader_kwargs)

    # Set up optimizer with the learning rate and momentum provided by argparse
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Set model to training mode
    model.train()

    # Train for the number of epochs provided by argparse
    for epoch in range(1, args.epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.to(args.device))  # Move data to correct device
            loss = F.nll_loss(output, target.to(args.device))  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            if batch_idx % args.log_interval == 0:
                print(
                    f"Process {rank} - Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )

            # Exit early for dry-run
            if args.dry_run:
                break


def main():
    parser = argparse.ArgumentParser(description="MNIST Training Script")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,  # Default to 1 epoch as required
        metavar="N",
        help="number of epochs to train (default: 1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=2,
        metavar="N",
        help="how many training processes to use (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--device", default="cpu", help="Device to train the model on (default: cpu)"
    )
    args = parser.parse_args()

    # Set the correct device (CPU or CUDA)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device

    # Initialize the model and move it to the correct device
    model = Net().to(device)

    # Set up multi-processing
    mp.set_start_method('spawn', force=True)
    model.share_memory()  # Share the model between processes for Hogwild training

    # Data transformation and loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST('/opt/mount/data', train=True, download=True, transform=transform)

    # DataLoader settings
    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 1,  # Using 1 worker per process
        "pin_memory": True,
        "shuffle": True,
    }

    # Start the multi-process Hogwild training
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, args, model, train_dataset, kwargs))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


    # Define the checkpoint path
    checkpoint_dir = "/opt/mount/model"
    checkpoint_path = f"{checkpoint_dir}/mnist_cnn.pt"

    # Create the directory if it doesn't exist
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Save the model checkpoint to the mounted volume directory
    torch.save(model.state_dict(), checkpoint_path)

    print(f"Training completed. Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
