import numpy as np

class Agent:

    def __init__(self) -> None:
        # action space
        '''
            actions[0] -> forward
            actions[1] -> backward
            actions[2] -> jump

        '''
        # self.actions = np.array([0,0,0])
        self.actions = [0,0,0]

    def train(self):
        '''
            Train Agent
        '''
    
    def get_action(self, state):
        '''
            Get next action give current state  
        '''
        action = None

        return action