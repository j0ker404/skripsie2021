
class Agent:
    '''
        Base Agent class implementation

    '''
    def __init__(self) -> None:
        # action space
        '''
            actions[0] -> forward
            actions[1] -> backward
            actions[2] -> jump
        '''
        self.T_MAX =  3000
        self.MAX_SCORE = 5
        self.max_actions = 6

        # self.actions = np.array([0,0,0])
        self.actions = [0,0,0]
        self.action_table = [[0, 0, 0], # NOOP
                            [1, 0, 0], # LEFT (forward)
                            [1, 0, 1], # UPLEFT (forward jump)
                            [0, 0, 1], # UP (jump)
                            [0, 1, 1], # UPRIGHT (backward jump)
                            [0, 1, 0]] # RIGHT (backward)

    def train(self):
        '''
            Train Agent
        '''
        pass
    
    def get_action(self, state):
        '''
            Get next action give current state  
        '''
        action = None

        return action