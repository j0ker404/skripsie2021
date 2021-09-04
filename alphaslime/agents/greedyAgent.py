'''
    Base implementation of epsilon greedy
    agent
'''


import numpy as np
from alphaslime.agents.agent import Agent
from alphaslime.approx.q import QApprox

class GreedyAgent(Agent):
    '''
        Epsilon greedy agent base class
        will implemented epsilon greedy algorithm for
        action selection

    '''

    def __init__(self, epsilon, q_hat:QApprox, d, weights=None, *args, **kwargs) -> None:
        '''
            epsilon: epsilon value

            q_hat: q-function approximator

            d: dimension of observation state space

            weights: weight vector for q-fuction approximator
                    ndarray: (d+1, n_actions)
        '''
        super().__init__(*args, **kwargs)
        
        # epsilon value used for exploration/exploitation
        self.epsilon = epsilon

        # q approximator
        self.q_hat = q_hat

        # set observation space dimensions
        self.d = d

        if weights is None:
            # configure weight to have extra dimension i.e. d+1
            # the extra dimension is used as a bias term/offset

            self.w = np.zeros((self.d+1, self.max_actions))
        else:
            # trained data
            self.w = weights

    
    def get_action(self, state):
        '''
            Get next action given current state
            for an  agent

            state: current observation

            epsilon-greedy implementation  

            return action: list(), len=3
        '''
        action_index = None

        # epsilon greedy alg
        prob = np.random.uniform()
        if prob < self.epsilon:
            # random action
            action_index = np.random.randint(low=0, high=self.max_actions)
        else:
            q_values = [self.q_hat.state_action_value(
                state=state, action=self.action_table[action_index], w=self.w[:, action_index]) for action_index in range(self.max_actions)]
            q_values = np.array(q_values)
            action_index = np.argmax(q_values)
        return self.action_table[action_index]


    
