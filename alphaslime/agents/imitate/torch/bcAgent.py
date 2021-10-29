import torch as T
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from alphaslime.agents.agent import Agent
from alphaslime.agents.imitate.torch.episodeDataset import EpisodeDataset, EpisodesDataset
from alphaslime.store.config import Config
from other.cartpole.algs.policygrad.ppo_training_configs import EPISODES
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

class BCAgent(Agent):

    def __init__(self, CONSTANTS: Config, config:Config) -> None:
        super().__init__(CONSTANTS)

        layer_sizes = config.get('policy_dims')
        self.alpha = config.get('alpha')
        self.batch_size = config.get('batch_size')
        self.n_epochs = config.get('n_epochs')

        self.verbose = config.get('verbose') 

        self.MODEL_CHECKPOINT_PATH = config.get('model_chkpt_path')
        # # create directory if not present
        # if not os.path.exists(self.MODEL_CHECKPOINT_PATH):
        #     os.makedirs(self.MODEL_CHECKPOINT_PATH)


        self.BCNetType = config.get('BCNetType')
        self.policyNet = self.BCNetType(layer_sizes, self.alpha)

        # create empty training data
        self.clear_training_data()

    def clear_training_data(self):
        """Clear the training data

            Training data:
                - loss_list
        """
        self.loss_list = []


    def get_training_data(self) -> dict:
        """Return current recorded training data
            
            Data in form:
            training_data = {
                'losses': self.loss_list
            }

        Returns:
            dict: Returned training data
        """
        training_data = {
            'losses': self.loss_list
        }

        return training_data     

    
    def train(self, training_config:Config):
        EPISODES = training_config.get('EPISODES')
        eps_dir = training_config.get('expert_episodes_data_path')
        shuffle = training_config.get('shuffle_batches')
        is_progress =  training_config.get('is_progress')
        
        batch_size = self.batch_size
        num_eps = EPISODES
        episode_dir = eps_dir
        expert_episodes_data = EpisodesDataset(num_eps, episode_dir)
        expert_episodes_dataloader = DataLoader(expert_episodes_data, batch_size=batch_size, shuffle=shuffle)
        self.optimize_model(expert_episodes_dataloader, is_progress=is_progress)
 

    def optimize_model(self, expert_episodes_dataloader:DataLoader, is_progress=False):

        ranger = range(self.n_epochs)
        if is_progress:
            ranger = tqdm(ranger)
        # we want to store the loss per batch

        optimizer = self.policyNet.optimizer
        loss_fn = self.policyNet.loss_func

        loss_epoch = []
        self.latest_epoch_loss = None
        for epoch in ranger:
            print(f"Epoch {epoch+1}\n-------------------------------")
            size = len(expert_episodes_dataloader.dataset)
            
            loss_per_batch = []
            for batch, X in enumerate(expert_episodes_dataloader):
                print('Batch {} of epoch {}'.format(batch, epoch))
                batch_episodes_loss = []
                loss = T.zeros([]).to(self.policyNet.device)
                # X episode paths, len(X)=batch_size
                # for each episode
                for episode_path in X:

                    episode_data = EpisodeDataset(episode_path)
                    # iterate through each time step of episode
                    for state_t, actions_t_index, reward_t in episode_data:
                        
                        # create an one-hot tensor for action_t_index
                        action_t = T.zeros((len(self.action_table),)).to(self.policyNet.device)
                        action_t[actions_t_index] = 1

                        # convert state to Tensor
                        state_Tensor = T.tensor(state_t, dtype=T.float).to(self.policyNet.device)
                        # Compute prediction and loss
                        pred_action = self.policyNet(state_Tensor).to(self.policyNet.device)
                        print('pred_action.device = {}'.format(pred_action.device))
                        print('action_t.device = {}'.format(action_t.device))
                        loss += loss_fn(pred_action, action_t)
                        
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch % 100 == 0 and self.verbose:
                    loss_item, current = loss.item(), batch * len(X)
                    print(f"loss: {loss_item:>7f}  [{current:>5d}/{size:>5d}]")

                # store total loss per batch
                loss_per_batch.append(loss.item())

            # store total losses for all batches per epoch
            loss_epoch.append(loss_per_batch)
            
            self.latest_epoch_loss = np.max(loss_per_batch)
            # save model after each epoch
            path = self.MODEL_CHECKPOINT_PATH + 'epoch_' + str(epoch) + '_loss_' + str(self.latest_epoch_loss)
            self.save_model(path)

        # store loss data is loss_list
        self.loss_list = loss_epoch

    def get_action(self, observation):
        action_index = None
        state = T.tensor([observation], dtype=T.float).to(self.policyNet.device)
        with T.no_grad():

            act:Tensor = self.policyNet(state)
            action_index = act.argmax()
        
        print(action_index)
        return action_index

    def forward(self, obs, action):
        pass

    def save_model(self, path):
        """Save bc model
        base_path (str): Base file path name
                            in the form of 
                            $PATH$/name_model
        """
        if self.verbose:
            print('... saving model ...')    
        policy_path = path + '_bc.pt'
        self.policyNet.save_model(policy_path)
    
    def load_model(self, path):
        """Load actor and critic models

            paths (list): Path BC model
        """
        if self.verbose:
            print('... loading model ...')
        self.policyNet.load_model(path)
