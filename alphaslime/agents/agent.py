import pickle
from alphaslime.store.config import Config
class Agent:
    '''
        Base Agent class implementation

    '''
    # def __init__(self, config:dict, action_table=None, max_score=None, t_max=3000, episode_printer=100, env=None) -> None:
    def __init__(self, CONSTANTS:Config) -> None:
        '''
            CONSTANTS: key-values
            CONSTANTS is a Config instance that contains the constant information

            - action_table: list of actions
                - action is a boolean array
                - example, action_1 = [0,1,0]            
            - env: gym environment
        
        '''
        # action space
        '''
            actions[0] -> forward
            actions[1] -> backward
            actions[2] -> jump
        '''
    
        # configure environment
        self.env = CONSTANTS.get('env')
        self.action_table = CONSTANTS.get('action_table')
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

    def get_training_data(self) -> dict:
        """Return current recorded training data
            
            Data in form:
            training_data = {
                'train_1': train_1_value,
            }

        Returns:
            dict: Returned training data
        """
        pass


    def save_training_data(self, path):
        """Save the current recorded training data to disk

            Python Pickling is used to save the data that is stored in 
            a dictionary

            Data stored in a dictionary:

        Args:
            path (str): Path to save the data
        """
        training_data = self.get_training_data()
        with open(path, 'wb') as f:
            pickle.dump(training_data, f)