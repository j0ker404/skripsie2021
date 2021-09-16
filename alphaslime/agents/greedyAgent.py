'''
    Base implementation of epsilon greedy
    agent
'''


import numpy as np
from alphaslime.agents.agent import Agent

class GreedyAgent(Agent):
    '''
        Epsilon greedy agent base class
        will implemented epsilon greedy algorithm for
        action selection

    '''

    def __init__(self, config:dict) -> None:
        '''
            epsilon, q_hat:QApprox, d, weights=None

            config: key-value

            - epsilon: epsilon value

            - q_hat: q-function approximator

            - d: dimension of observation state space

            - weights: weight vector for q-fuction approximator
                    ndarray: (d+1, n_actions)

            TODO: raise errors for non-init values 
        '''
        super().__init__(config)
        
        # epsilon value used for exploration/exploitation
        self.epsilon = config['epsilon']

        # q approximator
        self.q_hat = config['q_hat']

        # set observation space dimensions
        self.d = config['d']
        
        weights = config['weights']
        if weights is None:
            # configure weight to have extra dimension i.e. d+1
            # the extra dimension is used as a bias term/offset

            # self.w = np.zeros((self.d+1, self.max_actions))
            # self.w = np.zeros((self.d, self.max_actions))
            self._reset()
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
                state=state, action=self.action_table[action_index], w=self.w) for action_index in range(self.max_actions)]
            q_values = np.array(q_values)
            action_index = np.argmax(q_values)
        return self.action_table[action_index]

    
    def _reset(self):
        '''
            Reset data of agent


            # set weights to init value
        '''

        # self.w = np.zeros((self.d+1, self.max_actions))
        # self.w = np.zeros((self.d*self.max_actions+1, 1))
        # self.w = np.zeros((self.d*self.max_actions, 1))
        # 4096 = self.FEATURE_VECTOR_LENGTH , TODO add property
        self.w = np.zeros((4096*self.max_actions +1, 1))


    
