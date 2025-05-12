import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dae import DAE



def main():

    ddae_model_path = 'dense_dae.pth'
    cdae_model_path = 'conv_dae.pth'

    # transform for CIFAR-10, normalizing to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Range: [-1, 1]
    ])

    # Load TR set
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True)

    # Load TS set
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

    print("Data loaded successfully.")

    ddae = DAE(mode='dense')
    if os.path.exists(ddae_model_path):
        print("Model found. Loading the model...")
        ddae.load_state_dict(torch.load(ddae_model_path))
    else:
        print("Model not found. Training the model...")
        ddae.train_model(train_loader)
        torch.save(ddae.state_dict(), ddae_model_path)
        print("Model saved successfully.")
    
    print("DDAE model ready.")

    # Quantitative evaluation
    psnr_score, ssim_score = ddae.evaluate(test_loader)
    print(f"Test PSNR: {psnr_score:.2f} dB, SSIM: {ssim_score:.4f}")

    # Qualitative visualization
    #ddae.visualize_results(test_loader)

    cdae = DAE(mode='conv')
    if os.path.exists(cdae_model_path):
        print("Model found. Loading the model...")
        cdae.load_state_dict(torch.load(cdae_model_path))
    else:
        print("Model not found. Training the model...")
        cdae.train_model(train_loader)
        torch.save(cdae.state_dict(), cdae_model_path)
        print("Model saved successfully.")

    print("CDAE model ready.")

    # Quantitative evaluation
    psnr_score, ssim_score = cdae.evaluate(test_loader)
    print(f"Test PSNR: {psnr_score:.2f} dB, SSIM: {ssim_score:.4f}")

    # Qualitative visualization
    cdae.visualize_results(test_loader)

if __name__ == "__main__":
    main()