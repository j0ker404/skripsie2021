"""
    Training configurations for
    Baseline agent for single agent slimevolley ball
"""
import sys
import os

PACKAGE_PARENT = '../../../'
sys.path.append(PACKAGE_PARENT)

import gym
import slimevolleygym
from alphaslime.agents.baseline import BaselineAgent
from alphaslime.store.constantConfig import Constants
from alphaslime.store.config import Config

# constant config
env_id = "SlimeVolley-v0"
env = gym.make(env_id)

# actions for slimeball
action_table = [[0, 0, 0], # NOOP
                [1, 0, 0], # LEFT (forward)
                [1, 0, 1], # UPLEFT (forward jump)
                [0, 0, 1], # UP (jump)
                [0, 1, 1], # UPRIGHT (backward jump)
                [0, 1, 0]] # RIGHT (backward)

data_path = 'Expert_Episodes_data/'
data_path = 'Expert_Episodes/'


# eval config
EPISODES = 300
EPISODES = 100000
EPISODES = 10000
is_progress_bar = True 
running_avg_len = 100

const = {
    'env': env,
    'action_table': action_table,
    'base_dir_path': data_path
}

eval_config = {
    'EPISODES': EPISODES,
    'is_progress_bar': is_progress_bar,
    'running_avg_len': running_avg_len,
    'agent_type': BaselineAgent,
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