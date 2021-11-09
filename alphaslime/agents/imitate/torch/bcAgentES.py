import torch as T
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from alphaslime.agents.agent import Agent
from alphaslime.agents.imitate.torch.bcAgent import BCAgent
from alphaslime.agents.imitate.torch.episodeDataset import EpisodeDataset, EpisodesDataset
from alphaslime.store.config import Config
from other.cartpole.algs.policygrad.ppo_training_configs import EPISODES
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='valid_checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        T.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss




class BCAgentES(BCAgent):
    '''
        Extend BCAgent to have Early Stopping
    '''

    def __init__(self, CONSTANTS: Config, config: Config) -> None:
        super().__init__(CONSTANTS, config)

    def clear_training_data(self):
        """Clear the training data

            Training data:
                - loss_list
        """
        self.loss_list = []
        self.valid_loss_list = []


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
            'losses': self.loss_list,
            'valid_losses': self.valid_loss_list,
        }

        return training_data     


    def train(self, training_config:Config):
        EPISODES = training_config.get('EPISODES')
        eps_dir = training_config.get('expert_episodes_data_path')
        shuffle = training_config.get('shuffle_batches')
        is_progress =  training_config.get('is_progress')
        # early stopping patience; how long to wait after last time validation loss improved.
        patience =  training_config.get('patience')
        valid_episode_dir = training_config.get('valid_episodes_data_path')
        num_eps_valid = training_config.get('EPISODES_VALID')
        self.ES_CHKPT_PATH = training_config.get('ES_CHKPT_PATH')
        
        batch_size = self.batch_size
        num_eps = EPISODES
        episode_dir = eps_dir
        expert_episodes_data = EpisodesDataset(num_eps, episode_dir)
        expert_episodes_dataloader = DataLoader(expert_episodes_data, batch_size=batch_size, shuffle=shuffle)

        valid_episodes_data = EpisodesDataset(num_eps_valid, valid_episode_dir)
        valid_episodes_dataloader = DataLoader(valid_episodes_data, batch_size=batch_size, shuffle=shuffle)

        self.optimize_model(expert_episodes_dataloader, patience, valid_episodes_dataloader, is_progress=is_progress)
 

    def optimize_model(self, expert_episodes_dataloader:DataLoader, patience:int, valid_episodes_dataloader:DataLoader, is_progress=False):

        ranger = range(self.n_epochs)
        if is_progress:
            ranger = tqdm(ranger)
        # we want to store the loss per batch

        optimizer = self.policyNet.optimizer
        loss_fn = self.policyNet.loss_func
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=self.ES_CHKPT_PATH)
        loss_epoch = []
        valid_loss_epoch = []
        self.latest_epoch_loss = None
        for epoch in ranger:
            print(f"Epoch {epoch+1}\n-------------------------------")
            size = len(expert_episodes_dataloader.dataset)
            
            loss_per_batch = []
            ###################
            # train the model #
            ###################
            self.policyNet.train() # prep model for training
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
                        # print('pred_action.device = {}'.format(pred_action.device))
                        # print('action_t.device = {}'.format(action_t.device))
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

            ######################    
            # validate the model #
            ######################
            valid_loss_per_batch = []
            self.policyNet.eval() # prep model for evaluation
            for batch, X in enumerate(valid_episodes_dataloader):
                print('Batch {} of validation epoch {}'.format(batch, epoch))
                valid_loss = T.zeros([]).to(self.policyNet.device)
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
                        # print('pred_action.device = {}'.format(pred_action.device))
                        # print('action_t.device = {}'.format(action_t.device))
                        valid_loss += loss_fn(pred_action, action_t)
                
                # store total loss per valid batch
                valid_loss_per_batch.append(valid_loss.item())
                
            # print training/validation statistics 
            # calculate average loss over an epoch
            train_loss = np.average(loss_per_batch)
            valid_loss = np.average(valid_loss_per_batch)

            print_msg = (f'[{epoch:>{self.n_epochs}}/{epoch:>{self.n_epochs}}] ' +
                    f'train_loss: {train_loss:.5f} ' +
                    f'valid_loss: {valid_loss:.5f}')
            print(print_msg)

    
            # store total valid losses for all batches per epoch
            valid_loss_epoch.append(loss_per_batch)

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, self.policyNet)
            
            if early_stopping.early_stop:
                print("Early stopping on epoch {}".format(epoch))
                break


        # store loss data is loss_list
        self.loss_list = loss_epoch
        # self.loss_list = valid_loss_epoch
        self.valid_loss_list = valid_loss_epoch

        # load the last checkpoint with the best model
        self.policyNet.load_model(self.ES_CHKPT_PATH)
