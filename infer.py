import json
import random
import torch
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
from model import Net


def infer(model, dataset, save_dir, device, num_samples=5):
    """Perform inference on randomly selected MNIST images and save results."""
    model.eval()
    results_dir = Path(save_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Randomly select 'num_samples' images from the dataset
    indices = random.sample(range(len(dataset)), num_samples)

    for idx in indices:
        image, _ = dataset[idx]
        image = image.to(device)

        with torch.no_grad():
            # Make a prediction
            output = model(image.unsqueeze(0))
        pred = output.argmax(dim=1, keepdim=True).item()

        # Convert the image to save it
        img = Image.fromarray((image.squeeze().cpu().numpy() * 255).astype('uint8')).convert("L")
        img.save(results_dir / f"{pred}.png")  # Save image with predicted number as filename


def main():
    # Define the directory to save results
    save_dir = "/app/mnist"

    # Load the trained model checkpoint
    checkpoint_path = "/app/mnist/mnist_cnn.pt"

    # Initialize the model and load the state_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)

    # Check if checkpoint exists
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_file():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}!")
        exit(1)

    # MNIST dataset with necessary transformations (only test data is used for inference)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./mnist_data', train=False, download=True, transform=transform)

    # Perform inference
    infer(model, test_dataset, save_dir, device)

    print("Inference completed. Results saved in the 'results' folder.")


if __name__ == "__main__":
    main()
