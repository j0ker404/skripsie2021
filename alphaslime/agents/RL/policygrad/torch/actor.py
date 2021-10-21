import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, layer_sizes, alpha):
        super(ActorNetwork, self).__init__()
        input_dim = layer_sizes[0]
        layer1_dim = layer_sizes[1]
        layer2_dim = layer_sizes[2]
        output_dim = n_actions
        self.actor = nn.Sequential(
                nn.Linear(input_dim, layer1_dim),
                nn.ReLU(),
                nn.Linear(layer1_dim, layer2_dim),
                nn.ReLU(),
                nn.Linear(layer2_dim, output_dim),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        print('Actor Device used: {}'.format(self.device))

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        
        return dist
    
    def save_model(self, path):
        """Save NN model to disk

        Args:
            path (str): file path to save file
                        in the form of: $PATH$.pt
        """
        T.save(self.state_dict(), path)

    def load_model(self, path):
        """Load NN model from disk

        Args:
            path (str): file path to load file
                        in the form of: $PATH$.pt
        """
        self.load_state_dict(T.load(path, map_location=self.device))
