import torch
from torch import nn

from dqn import DQN
import random
from collections import deque
import numpy as np

from tqdm import tqdm

class DQNAgent:
    '''
        https://blog.gofynd.com/building-a-deep-q-network-in-pytorch-fa1086aa5435
    '''

    def __init__(self, q_model, env, epsilon, gamma, batch_size, exp_mem_size, lr, seed=1423) -> None:

        torch.manual_seed(seed)
        self.device = q_model.device


        self.q_model = q_model
        self.q_target = DQN(self.q_model.learning_rate, layer_sizes=self.q_model.layer_sizes, device=self.q_model.device)
        self.update_q_target()
        # self.q_target.load_state_dict(self.q_model.state_dict())
        # self.q_target.eval()

        self.env = env
        self.n_actions = self.env.action_space.n
        self.epsilon = epsilon

        self.gamma = torch.tensor(gamma).float()
        self.steps_done = 0
        self.train_data = []

        # training data
        self.rewards = []
        self.loss_list = []
        self.epsilon_list = []

        self.target_update = 5
        self.target_update_counter = 0
        self.train_step_count = 0
        self.OPTIMIZE_COUNT = 4 

        self.EXP_MEMORY_SIZE = exp_mem_size
        self.BATCH_SIZE = batch_size

        # create replay buffer
        self.D = deque(maxlen=self.EXP_MEMORY_SIZE)


        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=lr)
        # self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=self.q_model.learning_rate)
        # self.optimizer = torch.optim.SGD(self.q_model.parameters(), lr=self.q_model.learning_rate)


    def init_replay_memory(self):
        """
            initialise replay memory
        """
        self.clear_replay_memory()
        counter = 0
        for _ in range(self.EXP_MEMORY_SIZE):
            # if is_mem_filled:
            #     break
            done = False
            obs = self.env.reset()
            # obs = torch.tensor([obs], device=self.device, dtype=torch.float64)
            while not done:
                # get action to execute based on state
                action = self.get_action(obs)

                #  take action, go to next time step
                obs_next, reward, done, info = self.env.step(action.item())
                self.collect_experience(
                    [obs, action.item(), reward, obs_next, done])

                obs = obs_next
                counter += 1
                if counter > self.EXP_MEMORY_SIZE:
                    break

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

    def train(self, EPISODES, is_progress=False):
        '''
            Train agent

            EPISODES: Total number of episodes to train
        '''
        # self.init_replay_memory()
        # self.train_step_count = 128
        ranger = range(EPISODES)
        if is_progress:
            ranger = tqdm(ranger)
            
        for episode in ranger:
            t, episode_data = self.episode_train()
            # update epsilon value
            # agent.epsilon = EPSILON_MIN + (EPSILON_START - EPSILON_MIN)*np.exp(-EPS_DECAY*episode)
            if self.epsilon > 0.05:
                self.epsilon -= (1 / 5000)

            self.rewards.append(episode_data[0])
            self.loss_list.append(episode_data[1])
            self.epsilon_list.append(self.epsilon)   
   

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

        #  take action, go to next time step
        obs_next, reward, done, info = self.env.step(action.item())

        self.collect_experience([obs, action.item(), reward, obs_next, done])
        
        # get action to execute based on state
        action_next = self.get_action(obs_next)

        self.train_step_count += 1

        if self.train_step_count > self.EXP_MEMORY_SIZE/2:
            self.train_step_count = 0
            for _ in range(self.OPTIMIZE_COUNT):
                if self.target_update_counter == self.target_update:
                    self.update_q_target()
                    self.target_update_counter = 0
                loss = self.optimize_model(self.BATCH_SIZE)

        other_data['loss'] = loss
        other_data['info'] = info

        return done, reward, obs_next, action_next, other_data

    

    def sample_experience(self, sample_size):
        # sample_size = self.BATCH_SIZE
        if len(self.D) < sample_size:
            sample_size = len(self.D)

        mini_batch = random.sample(self.D, sample_size)

        obs_batch = torch.tensor([tple[0] for tple in mini_batch]).float()
        action_batch = torch.tensor([tple[1] for tple in mini_batch]).float()
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

    def optimize_model(self, batch_size):
        '''
            Learn from minibatch and optimise/train the current q_model
        '''
        # sample minibatch
        obs_batch, action_batch, reward_batch, obs_next_batch, done_batch = self.sample_experience(
            batch_size)

        # Compute prediction and loss
        q_values = self.q_model(obs_batch)
        pred, _ = torch.max(q_values, axis=1)
        target_max_q_values = self.get_max_target_q_vals(obs_next_batch)
        y = reward_batch + self.gamma*target_max_q_values
        # y = reward_batch + self.gamma*target_max_q_values*(1-done_batch)

        loss = self.loss(pred, y)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.target_update_counter += 1


        return loss.item()


    def collect_experience(self, experience):
        self.D.append(experience)

    def get_action(self, obs):
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


    # def get_action(self, state):
    #     # We do not require gradient at this point, because this function will be used either
    #     # during experience collection or during inference
    #     with torch.no_grad():
    #         # Qp = self.q_net(torch.from_numpy(state).float().cuda())
    #         state_torch = torch.from_numpy(state)
    #         # print(state_torch)
    #         Qp = self.q_model(state_torch.float())
    #         # print(Qp)
    #     Q, A = torch.max(Qp, axis=0)
    #     # print('Q, A = {}, {}'.format(Q, A))
    #     A = A if torch.rand(1, ).item() > agent.epsilon else torch.randint(
    #         0, agent.n_actions, (1,))
    #     return A

