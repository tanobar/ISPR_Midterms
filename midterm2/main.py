import os
import torch
from rbm_scratch import RBM
from torchvision import datasets, transforms


def encode_test_images(rbm, test_loader, device):
    # Dictionary to store one image per digit (0-9)
    digit_representations = {}

    # Iterate through the test dataset
    for batch, labels in test_loader:
        batch = batch.to(device)  # Move to the correct device
        for img, label in zip(batch, labels):
            label = label.item()  # Convert label tensor to integer
            if label not in digit_representations:
                # Compute hidden neuron activations
                hidden_activations = rbm.visible_to_hidden(img.view(1, -1))
                digit_representations[label] = hidden_activations
            # Stop if we have one image per digit
            if len(digit_representations) == 10:
                break
        if len(digit_representations) == 10:
            break

    return digit_representations


def main():
    visible_dim = 784  # For MNIST
    hidden_dim = 256

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    rbm4_path = 'rbm4.pth'
    rbm8_path = 'rbm8.pth'

    # Initialize RBMs
    rbm4 = RBM(visible_dim, hidden_dim, k=4, lr=0.01, device=device)
    rbm8 = RBM(visible_dim, hidden_dim, k=8, lr=0.01, device=device)

    # Load the MNIST training data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(mnist_data, batch_size=10, shuffle=True)

    # Check and load/train RBM4
    if os.path.exists(rbm4_path):
        print("RBM4 model found. Loading model...")
        rbm4.load_model(rbm4_path)
    else:
        print("RBM4 model not found. Training RBM4...")
        rbm4.train(data_loader, epochs=100)
        rbm4.save_model(rbm4_path)

    # Check and load/train RBM8
    if os.path.exists(rbm8_path):
        print("RBM8 model found. Loading model...")
        rbm8.load_model(rbm8_path)
    else:
        print("RBM8 model not found. Training RBM8...")
        rbm8.train(data_loader, epochs=100)
        rbm8.save_model(rbm8_path)

    # Load MNIST validation data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    mnist_test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(mnist_test_data, batch_size=10, shuffle=False)

    # Evaluate on validation data
    test_error4 = rbm4.reconstruction_error(test_loader)
    print(f"Validation Reconstruction Error with k = 4: {test_error4}")

    test_error8 = rbm8.reconstruction_error(test_loader)
    print(f"Validation Reconstruction Error with k = 8: {test_error8}")

    print('After training')

    """# Encode test images using RBM4
    print("Encoding test images using RBM4...")
    rbm4_representations = encode_test_images(rbm4, test_loader, device)
    for digit, activation in rbm4_representations.items():
        print(f"Digit {digit}: Hidden activations (RBM4) -> {activation}")

    # Encode test images using RBM8
    print("Encoding test images using RBM8...")
    rbm8_representations = encode_test_images(rbm8, test_loader, device)
    for digit, activation in rbm8_representations.items():
        print(f"Digit {digit}: Hidden activations (RBM8) -> {activation}")

    print("Encoding complete.")"""


if __name__ == "__main__":
    main()