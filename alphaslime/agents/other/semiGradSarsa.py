import numpy as np
import gym

from alphaslime.agents.baseline import BaselineAgent

# from alphaslime.agents.agent import Agent
from alphaslime.agents.greedyAgent import GreedyAgent

from alphaslime.envgame.slenv import SLenv

from alphaslime.approx.linearq import LinearQApprox


class SemiGradSarsa(GreedyAgent):
    '''
        Implement Episodic Semi-gradient Sarsa for Estimating
        state-action value function (q)
    '''

    def __init__(self, alpha=1/10, epsilon=0.1, gamma=0.9, d=12, env_id=None, opponent=None, weights=None, is_MA=True, SEED=None) -> None:
        '''
            alpha: alpha value (float)

            epsilon: epsilon value for epsilon-greedy (float)

            gamma: (float)

            d: dimension of expected observation state

            env_id: gym environment id

            opponent: opponent agent for multi-agent environments

            weights: pretrained weights for agent

            is_MA: (boolean), true if multiagent environemt 
        
        '''
        # q function approximator
        self.q_hat = LinearQApprox()
        self.d = d

        super().__init__(epsilon=epsilon, q_hat=self.q_hat, d=self.d, weights=weights)
        self.alpha = alpha
        # self.epsilon = epsilon
        self.gamma = gamma


        # environment
        if env_id is None:
            env_id="SlimeVolley-v0"
        if opponent is None:
            opponent=BaselineAgent()
        if is_MA:
            self.env = SLenv(opponent=opponent, env_id=env_id)
        else:
            self.env = gym.make(env_id)
        
        # seed environment
        if SEED is not None:
            self.env.seed(seed=SEED)


        # minimum epsilon value
        self.MINIMUM_EPSILON = 0.0
        # after how many episodes will we decay epsilon
        self.DECAY_EPSILON_EPISODE_TARGET = 10000
        # epsilon decay rate
        self.EPSILON_DECAY = 0.9
        self.EPSILON_DECAY_STATE = True

        # values for epislonn decay with reward threshold
        self.reward_threshold = 0
        self.reward_threshold_increment = 1

        # exponetial epsilon decay
        self.EPSILON_DECAY_BASE = 0.997

    
    def reward_threshold_updater(self):
        '''
            Returns an updated reward threshold

            reward_threshold: current reward_threshold

            return new reward_threshold
        '''
        return self.reward_threshold+self.reward_threshold_increment

    def decay_epsilon(self):
        '''
            return decayed epsilon value.

            Exponetial decay in the form of 

            epsilon = epsilon - max score
        '''

        # foo =  self.epsilon * self.EPSILON_DECAY
        bar = self.epsilon - 1/self.MAX_SCORE

        return bar

    def train(self, episodes, reset_data=True):
        '''
            Train Agent


            episodes: total number of episodes to play


            #TODO: save training data
        '''
        
        if reset_data:
            # reset weights
            self.w = np.zeros((self.d+1, self.max_actions))
            # self.epsilon = 1
            self.reward_threshold = 0

        # time step tracker per episode
        t = 0
        # episode reward tracker
        episode_reward_value = 0
        # max values for normalisation
        max_val_array = np.zeros((2))
        
        for episode in range(episodes):
            # reset environment
            obs = self.env.reset()
            action = self.get_action(obs)


            # ---------------------------------------------------
            # decay epsilon value

            # exponential decay
            # if self.EPSILON_DECAY_STATE:
            #     if episode%self.DECAY_EPSILON_EPISODE_TARGET == 0 and self.epsilon > self.MINIMUM_EPSILON:
            #         self.epsilon = self.epsilon*self.EPSILON_DECAY
            #         if self.epsilon < self.MINIMUM_EPSILON:
            #             self.epsilon = self.MINIMUM_EPSILON   

            # decay with reward threshold
            if self.EPSILON_DECAY_STATE:
                # if episode_reward_value >= self.reward_threshold:
                #     self.epsilon = self.decay_epsilon()
                #     self.reward_threshold = self.reward_threshold_updater()
                
                self.epsilon = np.power(self.EPSILON_DECAY_BASE, episode)

            # ---------------------------------------------------


            # ---------------------------------------------------
            # reset episode time and reward values
            t = 0
            episode_reward_value = 0
            # ---------------------------------------------------

            if episode%self.episode_printer == 0:
                print('Completed Episodes = {}'.format(episode))
            while t < self.T_MAX:
                # take action, go to next time step
                obs_next, reward, done, info = self.env.step(action)
                t += 1 # increment time step
                episode_reward_value += reward # incremen episode reward value

                action_index = self.action_table.index(action)
                # if at terminal state
                if done:
                    q_hat = self.q_hat.state_action_value(
                        state=obs, action=action, w=self.w[:, action_index])
                    grad_q_hat = self.q_hat.grad_q(
                        state=obs, action=action, w=self.w[:, action_index])
                    beta = reward - q_hat

                    # update weights
                    self.w[:, action_index] = self.w[:, action_index] + \
                        self.alpha*beta*grad_q_hat.reshape((-1))

                    # normalize weights
                    # max_val =  self.w[:, action_index].max(axis=0)
                    # if np.abs(max_val) > 0:
                    #     self.w[:, action_index] = self.w[:, action_index] / max_val

                    max_val = np.abs(self.w[:, action_index].max(axis=0))
                    min_val = np.abs(self.w[:, action_index].min(axis=0))
                    max_val_array[0] = max_val
                    max_val_array[1] = min_val

                    abs_max = np.max(max_val_array)
                    if abs_max > 0:
                        self.w[:, action_index] = self.w[:, action_index] / abs_max  

                    # go to next episode
                    break

                action_next = self.get_action(obs_next)

                # ---------------------------------------------------
                # update weight vector

                action_next_index = self.action_table.index(action_next)

                q_hat = self.q_hat.state_action_value(
                    state=obs, action=action, w=self.w[:, action_index])

                grad_q_hat = self.q_hat.grad_q(
                    state=obs, action=action, w=self.w[:, action_index])

                q_hat_next = self.q_hat.state_action_value(
                    state=obs_next, action=action_next, w=self.w[:, action_next_index])

                # q_hat_next = self.q_hat.state_action_value(
                #     state=obs_next, action=action_next, w=self.w[:, action_index])

                delta = (reward + self.gamma*q_hat_next - q_hat)
            
                self.w[:, action_index] = self.w[:, action_index] + \
                    self.alpha*delta*grad_q_hat.reshape((-1))

                # normalize weight
                max_val = np.abs(self.w[:, action_index].max(axis=0))
                min_val = np.abs(self.w[:, action_index].min(axis=0))
                max_val_array[0] = max_val
                max_val_array[1] = min_val

                abs_max = np.max(max_val_array)
                if abs_max > 0:
                    self.w[:, action_index] = self.w[:, action_index] / abs_max 

                # ---------------------------------------------------

                # ---------------------------------------------------
                # update state-obeservation and action-state
                obs = obs_next
                action = action_next
                # ---------------------------------------------------


