'''
    Selfplay Game environment wrapper

    Useful when trainng a single agent via selfplay

'''


from gym.core import Env
from alphaslime.agents.agent import Agent
from gym import Wrapper

class SPenv(Wrapper):

    def __init__(self, env: Env, opponent:Agent) -> None:
        super().__init__(env)
        # # opponent agent: left player
        self.opponent = opponent

    
    def reset(self):
        '''
            Reset game 
        '''
        self.obs_player =  self.env.reset()
        self.obs_opponent = self.obs_player

        return self.obs_player

    def step(self, action):
        '''
            Make action and return reward, next state, etc
        '''
        # if opponent is None, then do random move
        if self.opponent is None:
            action_opponent = self.action_space.sample()
        else:
            action_opponent = self.opponent.get_action(self.obs_opponent)
        self.obs_player, reward, done, info = self.env.step(action, action_opponent)
        self.obs_opponent = info['otherObs']
        

        return self.obs_player, reward, done, info

    def seed(self, seed):
        return self.env.seed(seed)



    
        



