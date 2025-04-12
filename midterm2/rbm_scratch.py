import torch
import torch.nn.functional as F
from tqdm import tqdm


class RBM:
    def __init__(self, visible_dim, hidden_dim, k=1, lr=0.01, device='cpu'):
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.k = k  # number of Gibbs sampling steps
        self.learning_rate = lr
        self.device = device  # Device to run the computations on (e.g., 'cuda' or 'cpu')
        
        # weights and biases initialization
        self.W = torch.randn(hidden_dim, visible_dim, device=self.device) * 0.01
        self.v_bias = torch.zeros(visible_dim, device=self.device)
        self.h_bias = torch.zeros(hidden_dim, device=self.device)
    
    def sample_from_p(self, p):
        # bernoulli sampling given probabilities
        return F.relu(torch.sign(p - torch.rand_like(p, device=self.device)))
    
    def visible_to_hidden(self, v):  # forward pass
        # compute probabilities of hidden units given visible units
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return p_h
    
    def hidden_to_visible(self, h):  # backward pass
        # compute probabilities of visible units given hidden units
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return p_v

    def contrastive_divergence(self, v):
        # Move input to the correct device
        v = v.to(self.device)

        # positive phase
        p_h_given_v = self.visible_to_hidden(v)
        h_sample = self.sample_from_p(p_h_given_v)
        positive_grad = torch.mm(h_sample.t(), v)

        # gibbs sampling (negative phase)
        v_sample = v
        for _ in range(self.k):
            p_h_given_v = self.visible_to_hidden(v_sample)
            h_sample = self.sample_from_p(p_h_given_v)
            p_v_given_h = self.hidden_to_visible(h_sample)
            v_sample = self.sample_from_p(p_v_given_h)

        # negative phase
        p_h_given_v_sample = self.visible_to_hidden(v_sample)
        negative_grad = torch.mm(p_h_given_v_sample.t(), v_sample)

        # update weights and biases
        self.W += self.learning_rate * (positive_grad - negative_grad) / v.size(0)
        self.v_bias += self.learning_rate * torch.sum(v - v_sample, dim=0) / v.size(0)
        self.h_bias += self.learning_rate * torch.sum(p_h_given_v - p_h_given_v_sample, dim=0) / v.size(0)

    def train(self, data_loader, epochs=10):
        for epoch in range(epochs):
            epoch_error = 0
            print(f"Epoch {epoch + 1}/{epochs}")
            for batch in tqdm(data_loader, desc="Training Batches", leave=False):
                # Extract data from batch (ignore labels)
                batch, _ = batch  # Unpack the tuple (data, labels)
                batch = batch.view(-1, self.visible_dim).to(self.device)  # Flatten input and move to device
                self.contrastive_divergence(batch)
                
                # Compute reconstruction error
                v_reconstructed = self.hidden_to_visible(self.sample_from_p(self.visible_to_hidden(batch)))
                epoch_error += torch.sum((batch - v_reconstructed) ** 2).item()
            
            print(f"Epoch {epoch + 1}/{epochs}, Reconstruction Error: {epoch_error}")

    def reconstruction_error(self, data_loader):
        total_error = 0
        for batch in tqdm(data_loader, desc="Evaluating Reconstruction Error"):
            # Extract data from batch (ignore labels)
            batch, _ = batch  # Unpack the tuple (data, labels)
            batch = batch.view(-1, self.visible_dim).to(self.device)  # Flatten input and move to device
            v_reconstructed = self.hidden_to_visible(self.sample_from_p(self.visible_to_hidden(batch)))
            total_error += torch.sum((batch - v_reconstructed) ** 2).item()
        return total_error / len(data_loader.dataset)

    def save_model(self, path):
        torch.save({'W': self.W, 'v_bias': self.v_bias, 'h_bias': self.h_bias}, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.W = checkpoint['W'].to(self.device)
        self.v_bias = checkpoint['v_bias'].to(self.device)
        self.h_bias = checkpoint['h_bias'].to(self.device)