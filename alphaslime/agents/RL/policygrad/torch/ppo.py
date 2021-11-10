'''
    Based on code from:
    https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/torch/ppo_torch.py
'''

from collections import deque
import os
import numpy as np
import torch as T
from tqdm import tqdm
from alphaslime.agents.RL.policygrad.torch.actor import ActorNetwork
from alphaslime.agents.RL.policygrad.torch.critic import CriticNetwork
from alphaslime.agents.RL.policygrad.torch.memory import PPOMemory

from alphaslime.agents.agent import Agent
from alphaslime.store.config import Config




class PPOAgent(Agent):

    def __init__(self, CONSTANTS: Config, config:Config) -> None:
        super().__init__(CONSTANTS)

        # T.manual_seed(256)
        input_dims = config.get('input_dims')
        self.alpha = config.get('alpha')
        self.batch_size = config.get('batch_size')

        self.gamma = config.get('gamma')
        self.gae_lambda = config.get('gae_lambda')
        self.policy_clip = config.get('policy_clip')
        self.n_epochs = config.get('n_epochs')
        # after how many time steps, learning occurs
        self.STEP_UPDATE = config.get('STEP_UPDATE') 
        self.verbose = config.get('verbose') 

        self.MODEL_CHECKPOINT_PATH = config.get('model_chkpt_path')
        # create directory if not present
        if not os.path.exists(self.MODEL_CHECKPOINT_PATH):
            os.makedirs(self.MODEL_CHECKPOINT_PATH)

        act_layer_sizes = config.get('actor_dims')
        crit_layer_sizes = config.get('critic_dims')
        # create actor and critic networks
        self.actor = ActorNetwork(self.n_actions, act_layer_sizes, self.alpha)
        self.critic = CriticNetwork(crit_layer_sizes, self.alpha)
        # self.actor = ActorNetwork(self.n_actions, input_dims, self.alpha, fc1_dims=64, fc2_dims=64)
        # self.critic = CriticNetwork(input_dims, self.alpha, fc1_dims=64, fc2_dims=64)
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

    def save_model(self, base_path):
        """Save actor and critic models
        base_path (str): Base file path name
                            in the form of 
                            $PATH$/name_model
        """
        if self.verbose:
            print('... saving models ...')
        # self.actor.save_checkpoint()
        # self.critic.save_checkpoint()
        actor_path = base_path+'_actor.pt'
        critic_path = base_path+'_critic.pt'
        self.actor.save_model(actor_path)
        self.critic.save_model(critic_path)

    def load_model(self, paths:list):
        """Load actor and critic models

            paths (list): Path for actor and critic
                        actor_path = paths[0]
                        critic_path = paths[1]
        """
        if self.verbose:
            print('... loading models ...')
        # self.actor.load_checkpoint()
        # self.critic.load_checkpoint()
        self.actor.load_model(paths[0])
        self.critic.load_model(paths[1])

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
        # self.avg_reward = 0
        self.n_steps = 0
        is_solved = False
        best_score = train_config.get('best_score')
        if best_score is not None:
            self.best_score = best_score
        for episode in ranger:
            
            # train an episode
            t, score = self.episode_train()

            # append episode rewards to training data
            self.rewards.append(score)
            
            # determine running mean
            rewards_deque.append(score)
            avg_reward = np.mean(rewards_deque)
            # append mean to training data 
            self.avg_rewards.append(avg_reward)

            if avg_reward > self.best_score:
                self.best_score = avg_reward
                path = self.MODEL_CHECKPOINT_PATH + 'avg_rew_' + str(self.best_score)
                self.save_model(path)

            if len(rewards_deque) == rewards_deque.maxlen:
                # determine solved environment
                if avg_reward >= threshold:
                    if not is_solved: 
                        print('\n Environment solved in {:d} episodes!\tAverage Score: {:.2f}'. \
                            format(episode, avg_reward))
                        is_solved = not is_solved
                    if is_threshold_stop:
                        break
            if episode % 10 == 0 and self.verbose:
                print('episode', episode, 'score %.1f' % score, 'avg score %.1f' % avg_reward,
                        'time_steps', self.n_steps, 'learning_steps', self.learn_iters)


    def episode_train(self):
        observation = self.env.reset()
        done = False
        score = 0.0
        t = 0
        while not done:
            # make one step move
            act_index, prob, val = self.get_action(observation)
            action = self.action_table[act_index]
            observation_next, reward, done, info = self.env.step(action)
            
            # increment time step
            t += 1

            # update total number of steps taken during training
            self.n_steps += 1

            # update episode total reward
            score += reward

            # store experiences in memory
            self.remember(observation, act_index, prob, val, reward, done)

            # determine if model needs to be optimized
            if self.n_steps % self.STEP_UPDATE == 0:
                self.optimize_model()
                self.learn_iters += 1

            observation = observation_next

        return t, score

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


    # def forward(self, obs, action_info:list):
        #     '''
        #         one time step train

        #         obs: state obeservation

        #         action_info (list): action, probabilty and value

        #         return:
        #         - done: boolean, True if episode complete

        #         - reward: int, reward gained from action

        #         - obs_next: next state observation after action executed

        #         - action_next: next action to execute bases on obs_next 

        #     '''
        #     # done = None
        #     # reward = None
        #     # obs_next = None
        #     other_data = {}
        #     loss = T.zeros(size=(self.n_epochs, self.batch_size, self.batch_size_info), dtype=T.float32)


        #     action, prob, val = action_info
        #     obs_next, reward, done, info = self.env.step(action)
        #     self.n_steps += 1
        #     self.remember(obs, action, prob, val, reward, done)
        #     if self.n_steps % self.STEP_UPDATE == 0:
        #         loss = self.optimize_model()
        #         self.learn_iters += 1

        #     # update other_data
        #     other_data['loss'] = loss
        #     other_data['info'] = info

        #     return done, reward, obs_next, other_data
