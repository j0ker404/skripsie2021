'''
    Single Agent Game environment wrapper

    Useful when trainng a single agent against a baseline agent

    Can be useful for self play
'''


from gym.core import Env
import numpy as np
import gym
from ..agents.agent import Agent 
from gym import Wrapper

class SLenv(Wrapper):

    def __init__(self, env: Env, opponent:Agent) -> None:
        super().__init__(env)
        # # opponent agent: left player
        self.opponent = opponent

    # def __init__(self,opponent:Agent, env_id="SlimeVolley-v0") -> None:
        
        # # opponent agent: left player
        # self.opponent = opponent

        # # set environment
        # self.env = gym.make(env_id)
        

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
        action_opponent = self.opponent.get_action(self.obs_opponent)
        self.obs_player, reward, done, info = self.env.step(action, action_opponent)
        self.obs_opponent = info['otherObs']
        

        return self.obs_player, reward, done, info

    def seed(self, seed):
        return self.env.seed(seed)



    
        



