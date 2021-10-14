from collections import deque
import os
import numpy as np
import torch as T
import torch.nn as nn
from torch.nn.modules import loss
import torch.optim as optim
from torch.distributions.categorical import Categorical
from tqdm import tqdm

from alphaslime.agents.agent import Agent
from alphaslime.store.config import Config


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))



class PPOAgent(Agent):

    def __init__(self, CONSTANTS: Config, config:Config) -> None:
        super().__init__(CONSTANTS)

        # T.manual_seed(256)
        input_dims = config.get('input_dims')
        alpha = config.get('alpha')
        self.batch_size = config.get('batch_size')

        self.gamma = config.get('gamma')
        self.gae_lambda = config.get('gae_lambda')
        self.policy_clip = config.get('policy_clip')
        self.n_epochs = config.get('n_epochs')
        # after how many time steps, learning occurs
        self.STEP_UPDATE = config.get('STEP_UPDATE') 
        self.verbose = config.get('verbose') 

        # create actor and critic networks
        self.actor = ActorNetwork(self.n_actions, input_dims, alpha, fc1_dims=64, fc2_dims=64)
        self.critic = CriticNetwork(input_dims, alpha, fc1_dims=64, fc2_dims=64)
        # self.actor = ActorNetwork(self.n_actions, input_dims, alpha)
        # self.critic = CriticNetwork(input_dims, alpha)
        # create PPO memory
        self.memory = PPOMemory(self.batch_size)      
        
        # for param in self.critic.parameters():
        #     print(param)

        # training data
        self.rewards = []
        self.loss_list = []
        self.avg_rewards = [] 

        self.best_score = self.env.reward_range[0]  


    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_model(self):
        """Save actor and critic models
        """
        if self.verbose:
            print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_model(self):
        """Load actor and critic models
        """
        if self.verbose:
            print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def clear_training_data(self):
        """Clear the training data

            Training data:
                - rewards
                - loss_list
                - average scores
        """
        self.rewards = []
        self.loss_list = []
        self.avg_rewards = [] 
    

    def get_training_data(self) -> dict:
        """Return current recorded training data
            
            Data in form:
            training_data = {
                'avg_rewards': self.avg_rewards,
                'rewards': self.rewards,
                'losses': self.loss_list
            }

        Returns:
            dict: Returned training data
        """
        training_data = {
            'avg_rewards': self.avg_rewards,
            'rewards': self.rewards,
            'losses': self.loss_list
        }

        return training_data


    def train(self, train_config:Config):
        """Train agent

        Args:
            train_config (Config): Training configs
        """
        # load configs
        EPISODES =  train_config.get('EPISODES')
        is_progress =  train_config.get('is_progress')
        threshold =  train_config.get('threshold')
        is_threshold_stop =  train_config.get('is_threshold_stop')
        running_avg_len =  train_config.get('running_avg_len')

        ranger = range(EPISODES)
        if is_progress:
            ranger = tqdm(ranger)

        rewards_deque = deque(maxlen=running_avg_len)
        
        self.learn_iters = 0
        self.avg_reward = 0
        self.n_steps = 0
        is_solved = False

        learn_iters = 0
        avg_score = 0
        n_steps = 0
        for episode in ranger:

            observation = self.env.reset()
            done = False
            score = 0

            while not done:
                act_index, prob, val = self.get_action(observation)
                action = self.action_table[act_index]
                observation_, reward, done, info = self.env.step(action)
                n_steps += 1
                score += reward
                self.remember(observation, act_index, prob, val, reward, done)
                if n_steps % self.STEP_UPDATE == 0:
                    # self.learn()
                    self.optimize_model()
                    learn_iters += 1
                observation = observation_


            self.rewards.append(score)

            reward = score
            rewards_deque.append(reward)
            
            avg_reward = np.mean(rewards_deque)
            self.avg_rewards.append(avg_reward)

            if avg_reward > self.best_score:
                self.best_score = avg_reward
                self.save_model()

            if len(rewards_deque) == rewards_deque.maxlen:
                if np.mean(rewards_deque) >= threshold:
                    if not is_solved: 
                        print('\n Environment solved in {:d} episodes!\tAverage Score: {:.2f}'. \
                            format(episode, np.mean(rewards_deque)))
                        is_solved = not is_solved
                    if is_threshold_stop:
                        break
            if episode % 10 == 0:
                print('episode', episode, 'score %.1f' % score, 'avg score %.1f' % avg_reward,
                        'time_steps', n_steps, 'learning_steps', learn_iters)

    # def train(self, train_config:Config):
    #     """Train agent

    #     Args:
    #         train_config (Config): Training configs
    #     """
    #     # load configs
    #     EPISODES =  train_config.get('EPISODES')
    #     is_progress =  train_config.get('is_progress')
    #     threshold =  train_config.get('threshold')
    #     is_threshold_stop =  train_config.get('is_threshold_stop')
    #     running_avg_len =  train_config.get('running_avg_len')

    #     ranger = range(EPISODES)
    #     if is_progress:
    #         ranger = tqdm(ranger)

    #     rewards_deque = deque(maxlen=running_avg_len)
        
    #     self.learn_iters = 0
    #     self.avg_reward = 0
    #     self.n_steps = 0
    #     is_solved = False
    #     for episode in ranger:
    #         t, episode_data = self.episode_train()

    #         self.rewards.append(episode_data[0])
    #         self.loss_list.append(episode_data[1])

    #         reward = episode_data[0]
    #         rewards_deque.append(reward)
            
    #         avg_reward = np.mean(rewards_deque)
    #         self.avg_rewards.append(avg_reward)

    #         if avg_reward > self.best_score:
    #             self.best_score = avg_reward
    #             self.save_model()

    #         if len(rewards_deque) == rewards_deque.maxlen:
    #             if np.mean(rewards_deque) >= threshold:
    #                 if not is_solved: 
    #                     print('\n Environment solved in {:d} episodes!\tAverage Score: {:.2f}'. \
    #                         format(episode, np.mean(rewards_deque)))
    #                     is_solved = not is_solved
    #                 if is_threshold_stop:
    #                     break

    #         print('episode', episode, 'score %.1f' % reward, 'avg score %.1f' % avg_reward,
    #         'time_steps', self.n_steps, 'learning_steps', self.learn_iters)


    def episode_train(self):
        '''
            Train one episode
        '''
        done = False
        obs = self.env.reset()
        rew = 0
        self.batch_size_info = 3
        losses = T.zeros(size=(self.n_epochs, self.batch_size, self.batch_size_info), dtype=T.float32)


        # data: [total reward for episode , total loss for]
        t = 0
        act, prob, val = self.get_action(obs)
        action = self.action_table[act]
        while not done:

            # perform one time step action
            done, reward, obs_next, other_data =  self.forward(obs, [action, prob, val])
            
            # get next action info
            act_next, prob_next, val_next = self.get_action(obs_next)
            action_next = self.action_table[act_next]
            
            # update next observation
            obs = obs_next
            # update next action
            action = action_next
            prob = prob_next
            val = val_next

            # update reward and loss data
            rew += reward
            losses += other_data['loss']
        
            # increment time step
            t += 1

        return t, [rew, losses]


    def forward(self, obs, action_info:list):
        '''
            one time step train

            obs: state obeservation

            action_info (list): action, probabilty and value

            return:
            - done: boolean, True if episode complete

            - reward: int, reward gained from action

            - obs_next: next state observation after action executed

            - action_next: next action to execute bases on obs_next 

        '''
        # done = None
        # reward = None
        # obs_next = None
        other_data = {}
        loss = T.zeros(size=(self.n_epochs, self.batch_size, self.batch_size_info), dtype=T.float32)


        action, prob, val = action_info
        obs_next, reward, done, info = self.env.step(action)
        self.n_steps += 1
        self.remember(obs, action, prob, val, reward, done)
        if self.n_steps % self.STEP_UPDATE == 0:
            loss = self.optimize_model()
            self.learn_iters += 1

        # update other_data
        other_data['loss'] = loss
        other_data['info'] = info

        return done, reward, obs_next, other_data

    def get_action(self, observation):
        # return super().get_action(state)
        # convert state to a tensor
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                # print(total_loss.detach().item())
        self.memory.clear_memory()               



    def optimize_model(self):
        """Learn
        """
        # losses = T.zeros(size=(self.n_epochs, self.batch_size, self.batch_size_info), dtype=T.float32)

        for epoch in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for i_batch, batch in enumerate(batches):
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

                # update losses tensor
                # losses[epoch, i_batch] = T.Tensor([actor_loss.detach(), critic_loss.detach(), total_loss.detach()])
                # print(total_loss.detach().item())
        self.memory.clear_memory()
        # return losses