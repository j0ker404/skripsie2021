'''
    DQN
'''
import torch
from torch import nn

class DQN(nn.Module):


    def __init__(self, lr, device, layer_sizes, seed=1423) -> None:
        super().__init__()
        # torch.manual_seed(seed)
        # self.hidden_layer_size = 64
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


class DQNv2(nn.Module):

    def __init__(self, lr, device, layer_sizes, seed=1423):
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