import pickle
from typing import Dict
import torch
from torch import nn
import random
from collections import deque
import numpy as np

from tqdm import tqdm
from alphaslime.approx.dqn import DQN
from alphaslime.agents.agent import Agent
from alphaslime.epsilon.epsilon import Epsilon 
from alphaslime.store.config import Config
from alphaslime.store.constantConfig import Constants

class DQNAgent(Agent):
    """DQNAgent

    A DQN agent implementation
    """

    # def __init__(self, config:dict, seed=None) -> None:
    def __init__(self, const:Constants, config:Config, seed=None) -> None:
        super().__init__(const)
        if seed:
            torch.manual_seed(seed)

        self.epsilon = config.get('epsilon')
        self.gamma = torch.tensor(config.get('gamma')).float()
        self.BATCH_SIZE = config.get('batch_size')
        self.EXP_MEMORY_SIZE = config.get('exp_mem_size')
        self.lr = config.get('lr')
        self.TARGET_UPDATE = config.get('TARGET_UPDATE')
        self.epsilon_decay:Epsilon = config.get('epsilon_decay')
        self.q_type = config.get('q_type')
        # self.q_model = config.get('q_hat')

        # load model properities
        self.device = config.get('device')
        self.layer_sizes = config.get('q_layer_sizes')

        # --------------------------------------
        # self.q_model = config['q_hat']
        # self.env = config['env']
        # self.epsilon = config['epsilon']
        # self.gamma = torch.tensor(config['gamma']).float()
        # self.BATCH_SIZE = config['batch_size']
        # self.EXP_MEMORY_SIZE = config['exp_mem_size']
        # self.lr = config['lr']
        # self.TARGET_UPDATE = config['TARGET_UPDATE']
        # self.epsilon_decay:Epsilon = config['epsilon_decay']
        # --------------------------------------

        # instantsiat target model
        # self.q_type = config['q_type']
        # consider using a copy.deepcopy to create target net
        self.q_model = self.q_type(lr=self.lr, layer_sizes=self.layer_sizes).to(self.device)
        self.q_target = self.q_type(self.q_model.learning_rate, layer_sizes=self.q_model.layer_sizes).to(self.device)
        self.update_q_target_no_eval()

        self.n_actions = self.env.action_space.n

        # create replay buffer
        self.D = deque(maxlen=self.EXP_MEMORY_SIZE)

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=self.lr)

        # training data
        self.rewards = []
        self.loss_list = []
        self.epsilon_list = []
        self.avg_rewards = [] 

    def init_q_target(self):
        """Init q_target
        """
        self.q_target = self.q_type(self.q_model.learning_rate, layer_sizes=self.q_model.layer_sizes).to(self.device)
        self.update_q_target_no_eval()

    def clear_training_data(self):
        """Clear the training data

            Training data:
                - rewards
                - loss_list
                - epsilon_list
                - average scores
        """
        self.rewards = []
        self.loss_list = []
        self.epsilon_list = []
        self.avg_rewards = [] 
    

    def get_training_data(self) -> dict:
        """Return current recorded training data
            
            Data in form:
            training_data = {
                'avg_rewards': self.avg_rewards,
                'rewards': self.rewards,
                'epsilon': self.epsilon_list,
                'losses': self.loss_list
            }

        Returns:
            dict: Returned training data
        """
        training_data = {
            'avg_rewards': self.avg_rewards,
            'rewards': self.rewards,
            'epsilon': self.epsilon_list,
            'losses': self.loss_list
        }

        return training_data


    def clear_replay_memory(self):
        """[summary]

            clear experience reply memory
        """
        self.D = deque(maxlen=self.EXP_MEMORY_SIZE)

    def get_target_q_vals(self, obs):
        '''
            Return the target  q-values for given observation

            pause grad operation

            obs: torch shape(n_samples, n_features)
        '''

        with torch.no_grad():
            q_vals = self.q_target(obs)

        return q_vals

    def get_max_target_q_vals(self, obs):
        """return the maximum q_values from target model
        Args:
            obs (torch.Tensor): observed state
        """
        q_vals = self.get_target_q_vals(obs)
        q_max, max_indices = torch.max(q_vals, dim=1)

        return q_max
    
    # def train(self, train_config:Config, EPISODES, is_progress=False, threshold=195, is_threshold_stop=True, running_avg_len=100):
    def train(self, train_config:Config):
        """Train agent

        Args:
            EPISODES (int): Max number of episodes to train
            is_progress (bool, optional): Display training progress bar. Defaults to False.
            threshold (int, optional): Average reward threshold to reach. Defaults to 195.
            is_threshold_stop (bool, optional): Stop training once initial threshold is reached. Defaults to True.
            running_avg_len (int, optional): Number of episodes rewards to average for running mean. Defaults to 100.

        Returns:
            [type]: [description]
        """
        '''
            Train agent

            EPISODES: Total number of episodes to train

            is_progess: (total)
        '''

        # load configs
        EPISODES =  train_config.get('EPISODES')
        is_progress =  train_config.get('is_progress')
        threshold =  train_config.get('threshold')
        is_threshold_stop =  train_config.get('is_threshold_stop')
        running_avg_len =  train_config.get('running_avg_len')
        

        # self.init_replay_memory()
        # self.train_step_count = 128
        ranger = range(EPISODES)
        if is_progress:
            ranger = tqdm(ranger)

        rewards_deque = deque(maxlen=running_avg_len)
        is_solved = False   
        for episode in ranger:
            # update epsilon value
            # self.epsilon = self.epsilon_annealing(episode, max_eps_episode, min_eps)
            self.epsilon = self.epsilon_decay.get_epsilon(episode)

            t, episode_data = self.episode_train()

            # update target
            if episode % self.TARGET_UPDATE == 0:
                # self.update_q_target()
                self.update_q_target_no_eval()

            self.rewards.append(episode_data[0])
            self.loss_list.append(episode_data[1])
            self.epsilon_list.append(self.epsilon)   

            reward = episode_data[0]
            rewards_deque.append(reward)
            
            avg_reward= np.mean(rewards_deque)
            self.avg_rewards.append(avg_reward)
            

            
            if len(rewards_deque) == rewards_deque.maxlen:
                ### 195.0: for cartpole-v0 and 475 for v1
                if np.mean(rewards_deque) >= threshold:
                    if not is_solved: 
                        print('\n Environment solved in {:d} episodes!\tAverage Score: {:.2f}'. \
                            format(episode, np.mean(rewards_deque)))
                        is_solved = not is_solved
                    if is_threshold_stop:
                        break
        # return avg_scores_array

   

    def episode_train(self):
        '''
            Train one episode
        '''
        done = False
        obs = self.env.reset()
        rew = 0
        losses = 0
        # data: [total reward for episode , total loss for]
        t = 0
        action = self.get_action(obs)
        while not done:

            # perform one time step action
            done, reward, obs_next, action_next, other_data =  self.forward(obs, action)
            
            # update next observation
            obs = obs_next
            # update next action
            action = action_next

            # update reward and loss data
            rew += reward
            losses += other_data['loss']
        
            # increment time step
            t += 1

        return t, [rew, losses]

    def forward(self, obs, action):
        '''
            Perform one time step forward pass for 
            agent

            return done, reward, obs_next, action_next, other_data

            returns next action to be taken with obs_next
        '''
        done = None
        reward = None
        obs_next = None
        action_next = None
        other_data = {}
        loss = 0

        act = self.action_table[action.item()]
        #  take action, go to next time step
        # obs_next, reward, done, info = self.env.step(action.item())
        obs_next, reward, done, info = self.env.step(act)

        self.collect_experience([obs, action.item(), reward, obs_next, done])
        
        # get action to execute based on state
        action_next = self.get_action(obs_next)
        
        if len(self.D) > self.BATCH_SIZE:
            batch = self.sample_experience(self.BATCH_SIZE)
            loss = self.optimize_model(batch)
            

        other_data['loss'] = loss
        other_data['info'] = info

        return done, reward, obs_next, action_next, other_data

    

    def sample_experience(self, sample_size):
        mini_batch = random.sample(self.D, sample_size)

        obs_batch = torch.tensor([tple[0] for tple in mini_batch]).float()
        action_batch = torch.tensor([tple[1] for tple in mini_batch])
        reward_batch = torch.tensor([tple[2] for tple in mini_batch]).float()
        obs_next_batch = torch.tensor([tple[3] for tple in mini_batch]).float()
        done_batch = torch.tensor([tple[4] for tple in mini_batch]).int()

        return obs_batch, action_batch, reward_batch, obs_next_batch, done_batch

    def update_q_target(self):
        '''
            Update the target q-model parameters to match to the current
            q-model parameters

        '''
        self.q_target.load_state_dict(self.q_model.state_dict())
        self.q_target.eval()

    def update_q_target_no_eval(self):
        '''
            Update the target q-model parameters to match to the current
            q-model parameters

            Does not call the eval() for model

        '''
        self.q_target.load_state_dict(self.q_model.state_dict())
        # self.q_target.eval()

    def optimize_model(self, batch):
        '''
            Learn from minibatch and optimise/train the current q_model
        '''
        if len(self.D) < self.BATCH_SIZE:
            loss = None
            return loss

        # sample minibatch
        obs_batch, action_batch, reward_batch, obs_next_batch, done_batch = batch

        actions = action_batch.unsqueeze(1)

        # Compute prediction and loss
        q_values = self.q_model(obs_batch)
        Q_expected = q_values.gather(1, actions)

        target_max_q_values = self.get_max_target_q_vals(obs_next_batch)
        y = reward_batch + self.gamma*target_max_q_values*(1-done_batch)
        Q_target = y.unsqueeze(1)

        # Backpropagation
        self.optimizer.zero_grad()
        loss = self.loss(Q_expected, Q_target)
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def collect_experience(self, experience):
        self.D.append(experience)

    def get_action(self, obs):
        '''
            Returns the index of the action
            for the action to execute based on the 
            action table

            return action: (Tensor), returns index for action
        '''
        # TODO: use action table for action selection
        sample = torch.rand((1,))
        action = None
        if sample.item() < self.epsilon:
            action = torch.randint(0, self.n_actions, (1,))
        else:
            with torch.no_grad():
                # print(obs)
                state_torch = torch.from_numpy(obs).float()
                q_vals = self.q_model(state_torch)
                action = torch.argmax(q_vals)
        return action


    def save_model(self, PATH):
        """Save the current q_model

        PATH: Path to save model
        """
        torch.save(self.q_model.state_dict(), PATH)


    def load_model(self, PATH):
        """Load Q_model for inference

        Args:
            PATH ([str]): Path for model to be loaded
        """
        self.q_model.load_state_dict(torch.load(PATH))
        self.q_model.eval()


    def load_model_no_eval(self, PATH):
        """Load Q_model 
        
        This method does not call the torch eval()

        Args:
            PATH ([str]): Path for model to be loaded
        """
        self.q_model.load_state_dict(torch.load(PATH))
