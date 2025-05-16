import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


class DAE(nn.Module):
    def __init__(self, mode=None):
        super(DAE, self).__init__()

        self.mode = mode

        if mode == 'conv':
            # Convolutional Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),   # 32x32
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), # 16x16
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1) # 8x8
            )
            # Convolutional Decoder
            self.decoder = nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1),  # 8x8
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, padding=1),   # 16x16
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 3, 3, padding=1),     # 32x32
                nn.Tanh()
            )
        elif mode == 'dense':
            # Dense Encoder
            self.encoder = nn.Sequential(
                nn.Linear(3 * 32 * 32, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU()
            )
            # Dense Decoder
            self.decoder = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 3 * 32 * 32),
                nn.Tanh()
            )
        else:
            raise ValueError("Invalid mode. Choose 'conv' or 'dense'.")


    def forward(self, x):
        if self.mode == 'conv':
            x = self.encoder(x)
            x = self.decoder(x)
        else:
            x = x.view(x.size(0), -1)  # Flatten to (batch_size, 3072)
            x = self.encoder(x)
            x = self.decoder(x)
            x = x.view(-1, 3, 32, 32)  # Reshape to image dimensions
        return x
    

    def add_noise(self, inputs, noise_factor=0.2):
        noise = noise_factor * torch.randn_like(inputs)
        noisy = inputs + noise
        return torch.clamp(noisy, -1., 1.)  # CIFAR-10 is normalized to [-1, 1]


    def train_model(self, train_loader, num_epochs=20, lr=1e-4, print_interval=100):
        #parameter print_interval (int): Print loss every N examples

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        self.train()  # Set to training mode
        for epoch in range(num_epochs):
            for batch_idx, (clean_imgs, _) in enumerate(train_loader):
                clean_imgs = clean_imgs.to(device)
                
                # Add noise and reconstruct
                noisy_imgs = self.add_noise(clean_imgs)
                reconstructed = self(noisy_imgs)
                
                # Compute loss and update
                loss = self.criterion(reconstructed, clean_imgs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Print progress
                if batch_idx % print_interval == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch {batch_idx}, Loss: {loss.item():.4f}")


    def evaluate(self, test_loader, noise_factor=0.2):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        """Compute PSNR and SSIM on test set"""
        self.eval()  # Set to evaluation mode
        total_psnr = 0.0
        total_ssim = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for clean_imgs, _ in test_loader:
                clean_imgs = clean_imgs.to(device)
                noisy_imgs = self.add_noise(clean_imgs, noise_factor)
                reconstructed = self(noisy_imgs)
                
                # Convert to numpy (CPU) for metric calculation
                clean_np = clean_imgs.cpu().numpy()
                recon_np = reconstructed.cpu().numpy()
                
                # Compute metrics per image
                for i in range(clean_np.shape[0]):
                    # PSNR (higher is better)
                    total_psnr += psnr(clean_np[i], recon_np[i], data_range=2.0)  # data_range=2 for [-1,1]
                    
                    # SSIM (higher is better, multichannel=True for RGB)
                    total_ssim += ssim(clean_np[i].transpose(1,2,0), 
                                     recon_np[i].transpose(1,2,0), 
                                     data_range=2.0, 
                                     channel_axis=2)
                
                num_samples += clean_np.shape[0]
        
        avg_psnr = total_psnr / num_samples
        avg_ssim = total_ssim / num_samples
        return avg_psnr, avg_ssim
    

    def visualize_results(self, test_loader, num_images=5):
        """Plot noisy vs reconstructed vs clean images"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        self.eval()
        with torch.no_grad():
            clean_imgs, _ = next(iter(test_loader))
            clean_imgs = clean_imgs.to(device)[:num_images]
            noisy_imgs = self.add_noise(clean_imgs)
            reconstructed = self(noisy_imgs)
            
            # Denormalize to [0,1] for plotting
            clean_imgs = (clean_imgs + 1) / 2
            noisy_imgs = (noisy_imgs + 1) / 2
            reconstructed = (reconstructed + 1) / 2
            
            fig, axes = plt.subplots(num_images, 3, figsize=(10, num_images*2))
            for i in range(num_images):
                axes[i,0].imshow(noisy_imgs[i].cpu().permute(1,2,0))
                axes[i,0].set_title("Noisy")
                axes[i,1].imshow(reconstructed[i].cpu().permute(1,2,0))
                axes[i,1].set_title("Reconstructed")
                axes[i,2].imshow(clean_imgs[i].cpu().permute(1,2,0))
                axes[i,2].set_title("Clean")
                for ax in axes[i]:
                    ax.axis('off')
            plt.tight_layout()
            plt.show()