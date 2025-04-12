import torch
import torch.nn as nn
import torch.nn.functional as F


class RBM(nn.Module):
    def __init__(self, visible_dim, hidden_dim, k=1, lr=0.01):
        super(RBM, self).__init__()
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.k = k  # number of Gibbs sampling steps
        self.learning_rate = lr
        
        # weights and biases initialization
        self.W = nn.Parameter(torch.randn(hidden_dim, visible_dim) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(visible_dim))
        self.h_bias = nn.Parameter(torch.zeros(hidden_dim))
    
    def sample_from_p(self, p):
        # bernoulli sampling given probabilities
        return F.relu(torch.sign(p - torch.rand_like(p)))
    
    def visible_to_hidden(self, v): #forward pass
        # compute probabilities of hidden units given visible units
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return p_h
    
    def hidden_to_visible(self, h): #backward pass
        # compute probabilities of visible units given hidden units
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return p_v

    def contrastive_divergence(self, v):
        # positive phase
        p_h_given_v = self.visible_to_hidden(v)
        h_sample = self.sample_from_p(p_h_given_v)
        positive_grad = torch.matmul(h_sample.t(), v)

        # gibbs sampling (negative phase)
        v_sample = v
        for _ in range(self.k):
            p_h_given_v = self.visible_to_hidden(v_sample)
            h_sample = self.sample_from_p(p_h_given_v)
            p_v_given_h = self.hidden_to_visible(h_sample)
            v_sample = self.sample_from_p(p_v_given_h)

        # negative phase
        p_h_given_v_sample = self.visible_to_hidden(v_sample)
        negative_grad = torch.matmul(p_h_given_v_sample.t(), v_sample)

        # update weights and biases
        self.W.data += self.learning_rate * (positive_grad - negative_grad) / v.size(0)
        self.v_bias.data += self.learning_rate * torch.sum(v - v_sample, dim=0) / v.size(0)
        self.h_bias.data += self.learning_rate * torch.sum(p_h_given_v - p_h_given_v_sample, dim=0) / v.size(0)