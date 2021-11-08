"""
    Training configurations for
    PPO agent for cartpole
"""
import sys
import os

PACKAGE_PARENT = '../../../'
sys.path.append(PACKAGE_PARENT)

import gym
from alphaslime.store.constantConfig import Constants
from alphaslime.store.config import Config
import trained_agents.agents as TRAINED_AGENTS

# constant config
env_id = "CartPole-v1"
env = gym.make(env_id)

# actions for slimeball
action_table = [0, 1]

data_path = 'Expert_CP_Episodes/'


# eval config
EPISODES = 1000
is_progress_bar = True 
running_avg_len = 100
agent = TRAINED_AGENTS.agents_dict_cartpole()['PPO_Agent']

const = {
    'env': env,
    'action_table': action_table,
    'base_dir_path': data_path
}

eval_config = {
    'EPISODES': EPISODES,
    'is_progress_bar': is_progress_bar,
    'running_avg_len': running_avg_len,
    'agent': agent,
    'render': False,
    'time_delay': 0
}


CONST = Constants(const)


# create utility function for creating directors
def create_dirs():
    """Create required directors for
        configuration
    """
    # create base data dir
    if not os.path.exists(data_path):
        os.makedirs(data_path)