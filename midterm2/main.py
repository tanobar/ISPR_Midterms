import torch
from rbm_scratch import RBM
from torchvision import datasets, transforms

def main():
    visible_dim = 784  # For MNIST
    hidden_dim = 128

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    rbm4 = RBM(visible_dim, hidden_dim, k=4, lr=0.01, device=device)
    rbm8 = RBM(visible_dim, hidden_dim, k=8, lr=0.01, device=device)

    # load the MNIST training data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(mnist_data, batch_size=10, shuffle=True)

    # training
    rbm4.train(data_loader, epochs=10)
    rbm8.train(data_loader, epochs=10)

    # load MNIST validation data
    mnist_val_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(mnist_val_data, batch_size=10, shuffle=False)

    # Evaluate on validation data
    val_error4 = rbm4.reconstruction_error(val_loader)
    print(f"Validation Reconstruction Error with k = 4: {val_error4}")

    val_error8 = rbm8.reconstruction_error(val_loader)
    print(f"Validation Reconstruction Error with k = 8: {val_error8}")

    # save the models
    torch.save(rbm4, 'rbm4.pth')
    torch.save(rbm8, 'rbm8.pth')



if __name__ == "__main__":
    main()