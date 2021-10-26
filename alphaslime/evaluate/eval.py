from collections import deque
import os
import pickle
import numpy as np
from torch.functional import Tensor
from alphaslime.agents.agent import Agent
import time
from tqdm import tqdm

class Evaluate:

    """General base class for
        evaulating agents
    """

    def __init__(self, env, base_dir_path,render=False, time_delay=0.03) -> None:
        self.RENDER = render
        # base directory to save data
        self.base_dir_path = base_dir_path
        self.env = env
        self.delay = time_delay

        # create dir if does not exist
        if not os.path.exists(self.base_dir_path):
            os.makedirs(self.base_dir_path)

    
    def evaluate_episode(self, idx):
        '''
            Evaluate one episode

            Episode terminates when either agent loses all five lives, 
            or after 3000 timesteps has passed.
            
            #TODO: save data to a file
                - save state
                - save actions
                - save rewards
                - save time step

            args:
                idx (int): episode count

            return agent right score,
            one can infer agent left score
        '''
        episode_reward = 0
        return episode_reward
        
    def evaluate(self, EPISODES, is_progress_bar=False, running_avg_len=100):
        """Evaluate agent performance for given number of episodes

        Args:
            EPISODES (int): Number of Episodes to run
            is_progress_bar (bool): Display progress bar. Default to False
        
        return:
            episodes_reward: (list), total reward per episode
            avg_rewards_array: (list), running reward average per episode 
        """
        rewards = []
        rewards_deque = deque(maxlen=running_avg_len)
        avg_rewards_array = [] 
        ranger = range(EPISODES)
        if is_progress_bar:
            ranger = tqdm(ranger)
        
        # evaluate episodes
        for episode in ranger:
            episode_reward = self.evaluate_episode(episode)
            rewards_deque.append(episode_reward)
            avg_score = np.mean(rewards_deque)
            # append average reward at current epsiode
            avg_rewards_array.append(avg_score)
            # append total episode reward
            rewards.append(episode_reward)

        self.env.close()

        return rewards, avg_rewards_array

    
    def save_episode(self, episode_index, episode_data):
        """Save episode data to disk

            Episode_data ~ [state_t, action_t, reward_t]

        Args:
            episode_index (int): episode index
            episode_data (list): episode data
        """

        eps_path = os.path.join(self.base_dir_path, 'episode_'+str(episode_index)+'.pkl')
        with open(eps_path, 'wb') as f:
            pickle.dump(episode_data, f)