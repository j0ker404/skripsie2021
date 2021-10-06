import torch
import torch.nn as nn


class DQN(nn.Module):
    """Deep Q network class

    Args:
        nn ([torch.nn]): NN base class
    """

    def __init__(self, lr, device, layer_sizes):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.input_size = self.layer_sizes[0]
        self.hidden_layer_size = self.layer_sizes[1]
        self.output_size = self.layer_sizes[2]
        self.seq = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_layer_size),
            nn.PReLU(),
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
            nn.PReLU(),
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
            nn.PReLU(),
            nn.Linear(self.hidden_layer_size, self.output_size)
        )
        self.learning_rate = lr
        self.device = device

    def forward(self, x):
        logits = self.seq(x)
        return logits


class DQNv2(nn.Module):
    """Deep Q network class

    Args:
        nn ([torch.nn]): NN base class
    """

    def __init__(self, lr, device, layer_sizes):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.input_size = self.layer_sizes[0]
        self.hidden_layer_size = self.layer_sizes[1]
        self.output_size = self.layer_sizes[2]
        self.seq = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_layer_size),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_size, self.output_size)
        )
        self.learning_rate = lr
        self.device = device

    def forward(self, x):
        logits = self.seq(x)
        return logits