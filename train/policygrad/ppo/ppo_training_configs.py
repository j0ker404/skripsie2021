"""
    Training configurations for
    PPO agent for single agent slimevolley ball
"""
import sys
import os

PACKAGE_PARENT = '../../../'
sys.path.append(PACKAGE_PARENT)

import gym
from alphaslime.agents.RL.policygrad.torch.ppo import PPOAgent
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

# env.seed(256)
# random.seed(256)


# agent config
STEP_UPDATE = 4096
input_dims = env.observation_space.shape
gamma = 0.99
alpha = 0.0003
gae_lambda = 0.95
policy_clip = 0.2
batch_size = 64
n_epochs = 10


# training config
threshold = 195
is_threshold_stop = False
running_avg_len = 100
is_progress = True
EPISODES = 300
EPISODES = 1000


const = {
    'env': env,
    'action_table': action_table
}



agent_config = {
    'input_dims': input_dims,
    'gamma': gamma,
    'alpha':alpha,
    'gae_lambda': gae_lambda,
    'policy_clip': policy_clip,
    'batch_size': batch_size,
    'n_epochs': n_epochs,
    'STEP_UPDATE': STEP_UPDATE,
    'verbose': False
}


training_configs = {
    'agent_type': PPOAgent,
    'EPISODES': EPISODES,
    'is_progress': is_progress,
    'threshold': threshold, 
    'is_threshold_stop': is_threshold_stop,
    'running_avg_len': running_avg_len
}


CONST = Constants(const)
agent_hyper = Config(agent_config)
agent_training_configs = Config(training_configs)

figure_file = 'plots/cartpole.png'