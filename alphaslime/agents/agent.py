
class Agent:
    '''
        Base Agent class implementation

    '''
    def __init__(self, action_table=None, max_score=None, t_max=3000, episode_printer=100) -> None:
        # action space
        '''
            actions[0] -> forward
            actions[1] -> backward
            actions[2] -> jump
        '''
        self.T_MAX =  t_max
        self.MAX_SCORE = max_score
        self.episode_printer = episode_printer
        # self.max_actions = 6

        # self.actions = np.array([0,0,0])
        self.actions = [0,0,0]
        if action_table is None:
            # configure to use the normal thing
            self.action_table = [[0, 0, 0], # NOOP
                                [1, 0, 0], # LEFT (forward)
                                [1, 0, 1], # UPLEFT (forward jump)
                                [0, 0, 1], # UP (jump)
                                [0, 1, 1], # UPRIGHT (backward jump)
                                [0, 1, 0]] # RIGHT (backward)
        else:
            self.action_table = action_table
        self.max_actions = len(self.action_table)

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