import numpy as np

from alphaslime.agents.greedyAgent import GreedyAgent


class SemiGradSarsa(GreedyAgent):
    '''
        Implement Episodic Semi-gradient Sarsa for Estimating
        state-action value function (q)
    '''

    # def __init__(self, alpha=1/10, epsilon=0.1, gamma=0.9, d=12, env_id=None, opponent=None, weights=None, is_MA=True, SEED=None, *args, **kwargs) -> None:
    def __init__(self, config:dict, verbose=False) -> None:
        '''
            config: dict, with configuration values

            verbose: boolean, if True, prints debug info to screen

            config: key-values

            - alpha: alpha value (float)

            - epsilon: epsilon value for epsilon-greedy (float)

            - gamma: (float)

            - d: dimension of expected observation state

            - opponent: opponent agent for multi-agent environments

            - weights: pretrained weights for agent

            - q_hat: QApprox, q function approximator
        
        '''

        super().__init__(config)
        self.alpha = config['alpha']
        # self.epsilon = epsilon
        self.gamma = config['gamma']
        
        self.verbose = verbose

        # environment
        # if env_id is None:
        #     env_id="SlimeVolley-v0"
        # if opponent is None:
        #     opponent=BaselineAgent()
        # if is_MA:
        #     self.env = SLenv(opponent=opponent, env_id=env_id)
        # else:
        #     self.env = gym.make(env_id)
        

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

        # create empty training data list
        self.train_data = []

    
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

    def forward(self, obs, action):
        '''
        one time step train

            obs: current state observation

            action: action to execute based on state

            return done, reward, obs_next, action_next, other_data

            other_data = {
                q_hat,
                q_hat_next
            }
        '''

        # done, reward, obs_next, action_next, other_data =  super().forward(obs, action)

        action_next = None
        q_hat_next = None

        #  take action, go to next time step
        obs_next, reward, done, info = self.env.step(action)

        action_index = self.action_table.index(action)
        # if at terminal state
        if done:
            q_hat = self.q_hat.state_action_value(
                state=obs, action=action, w=self.w[:, action_index])
            grad_q_hat = self.q_hat.grad_q(
                state=obs, action=action, w=self.w[:, action_index])
            beta = reward - q_hat

            # update weights
            update = self.alpha*beta*grad_q_hat.reshape((-1,)) 
            self.w[:, action_index] += update
                

            # normalize weights
            # abs_max = np.abs(self.w[:, action_index]).max(axis=0) 

            # if abs_max > 0:
            #     self.w[:, action_index] = self.w[:, action_index] / abs_max  
            # go to next episode
        else:
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


            delta = (reward + self.gamma*q_hat_next - q_hat)

            update = self.alpha*delta*grad_q_hat.reshape((-1,))
            self.w[:, action_index] += update


            # normalize weight
            # abs_max = np.abs(self.w[:, action_index]).max(axis=0) 
            # if abs_max > 0:
            #     self.w[:, action_index] = self.w[:, action_index] / abs_max 

            # ---------------------------------------------------

            # ---------------------------------------------------
            # update state-obeservation and action-state
            # obs = obs_next
            # action = action_next
            # ---------------------------------------------------
        
        other_data  = {
            'q_hat': q_hat,
            'q_hat_next': q_hat_next
        }
        

        return done, reward, obs_next, action_next, other_data


    def train(self, episodes, reset_data=True):
        '''
            Train Agent


            episodes: total number of episodes to play

            return self


            #TODO: save training data
        '''
        if reset_data:
            # reset weights
            # self.w = np.zeros((self.d+1, self.max_actions))
            # self.w = np.zeros((self.d, self.max_actions))
            self._reset()
            # self.epsilon = 1
            self.reward_threshold = 0

        # display weights
        # print('init: weight = {}'.format(self.w))
        # iterate episodes
        for episode in range(episodes):
            # training output display
            if episode%self.episode_printer == 0 and self.verbose:
                print('Completed Episodes = {}'.format(episode))

            # update epsilon decay value
            if self.EPSILON_DECAY_STATE:
                self.epsilon = np.power(self.EPSILON_DECAY_BASE, episode)
            
            # train episode
            t, episode_reward_value = self.episode_train()
            
            # display weights
            # print('weight = {}'.format(self.w))

            data = [t, episode_reward_value] 
            # append trained data to data list
            self.train_data.append(data)
        
        self.env.close()
        return self

