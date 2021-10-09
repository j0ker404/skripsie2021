
class Agent:
    '''
        Base Agent class implementation

    '''
    # def __init__(self, config:dict, action_table=None, max_score=None, t_max=3000, episode_printer=100, env=None) -> None:
    def __init__(self, config:dict) -> None:
        '''
            action_table=None, max_score=None, t_max=3000, episode_printer=100, env=None
            config: key-values

            - action_table: list of actions
                - action is a boolean array
                - example, action_1 = [0,1,0]

            - max_score: float, max score that can be achieved for environment episode

            - t_max: int, max number of time steps per episode

            - episode_printer: int, used to determine the rate at which episode data is
                                logged(printed to screen)
            
            - env: gym environment
        
        '''
        # action space
        '''
            actions[0] -> forward
            actions[1] -> backward
            actions[2] -> jump
        '''
        # self.T_MAX =  t_max
        # self.MAX_SCORE = max_score
        # self.episode_printer = episode_printer
        # # configure environment
        # self.env = env

        # actions for slimeball
        # action_table = [[0, 0, 0], # NOOP
        #                 [1, 0, 0], # LEFT (forward)
        #                 [1, 0, 1], # UPLEFT (forward jump)
        #                 [0, 0, 1], # UP (jump)
        #                 [0, 1, 1], # UPRIGHT (backward jump)
        #                 [0, 1, 0]] # RIGHT (backward)

        # if action_table is None:
        #     # configure to use the normal thing
        #     self.action_table = [[0, 0, 0], # NOOP
        #                         [1, 0, 0], # LEFT (forward)
        #                         [1, 0, 1], # UPLEFT (forward jump)
        #                         [0, 0, 1], # UP (jump)
        #                         [0, 1, 1], # UPRIGHT (backward jump)
        #                         [0, 1, 0]] # RIGHT (backward)
        # else:
        #     self.action_table = action_table


        # self.T_MAX =  config['t_max']
        # self.MAX_SCORE = config['max_score']
        # self.episode_printer = config['episode_printer']
        # configure environment
        self.env = config['env']
        self.action_table = config['action_table']
        self.max_actions = len(self.action_table)


    def forward(self, obs, action):
        '''
            one time step train

            obs: state obeservation

            action: action to execute based on state

            return:
            - done: boolean, True if episode complete

            - reward: int, reward gained from action

            - obs_next: next state observation after action executed

            - action_next: next action to execute bases on obs_next 

        '''
        done = None
        reward = None
        obs_next = None
        action_next = None
        other_data = {}

        return done, reward, obs_next, action_next, other_data

    def episode_train(self):
        '''
            TODO: - log data to disk
            
            episode train

            return t, episode_reward_value
            - t: int, total number of time steps for episode
            - episode_reward_value: float, total reward value for episode
        '''
        # time step tracker per episode
        t = 0
        # episode reward tracker
        episode_reward_value = 0
        # done, is episode over
        done = False

        # reset environment for new episode
        obs = self.env.reset()
        # get action to execute based on state
        action = self.get_action(obs)

        while not done:
            '''
                other_data: dict | None
                    stores other data obtained from making an action
                    example a state-value function
            '''

            # go to next time step
            done, reward, obs_next, action_next, other_data = self.forward(obs, action)

            # increment episode reward counter
            episode_reward_value += reward

            # update observation, action for next time step
            obs = obs_next
            action = action_next

            # increment time step
            t += 1
        
        # return episode data

        return t, episode_reward_value

    def train(self):
        '''
            Train Agent
        '''
        pass
    
    def get_action(self, state):
        '''
            Get next action give current state  

            return:

                action_index: (int), Index of action to execute that corresponds
                                to an action in the Agent's Action table
        '''
        action = None

        return action
