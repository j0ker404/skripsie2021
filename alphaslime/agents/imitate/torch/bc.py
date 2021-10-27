import torch as T
import torch.nn as nn
import torch.optim as optim


class BCPolicyNet(nn.Module):
    """Policy Network for 
        for behavoiur cloning

    Args:
        nn (nn.Module): base class
    """
    def __init__(self, layer_sizes:list, alpha):
        super().__init__()

        self.layer_in_dim = layer_sizes[0]
        self.layer_1_dim = layer_sizes[1]
        self.layer_2_dim = layer_sizes[2]
        self.layer_out_dim = layer_sizes[-1]
        self.alpha = alpha
        self.layers = nn.Sequential(
            nn.Linear(self.layer_in_dim, self.layer_1_dim),
            nn.ReLU(),
            nn.Linear(self.layer_1_dim, self.layer_2_dim),
            nn.ReLU(),
            nn.Linear(self.layer_2_dim, self.layer_out_dim),
            nn.Softmax(-1)
        )

        # set optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.loss_func = T.nn.MSELoss()
        # self.optimizer = optim.SGD(self.parameters(), lr=self.alpha)

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        print('BCPolicyNet Device used: {}'.format(self.device))


    
    def forward(self, state):
        x = self.layers(state)        
        return x
    
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

