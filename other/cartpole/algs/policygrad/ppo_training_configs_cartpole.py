"""
    Training configurations for
    PPO agent for CartPole-v1
"""

import sys
import os

PACKAGE_PARENT = '../../../../'
sys.path.append(PACKAGE_PARENT)

import gym
from alphaslime.agents.RL.policygrad.torch.ppo import PPOAgent
from alphaslime.store.constantConfig import Constants
from alphaslime.store.config import Config

# constant config
env_id = "CartPole-v1"
env = gym.make(env_id)

# actions for cartpole
action_table = [0,1]

# env.seed(256)
# random.seed(256)

data_path = 'PPO_cartpole_data_cont/'
data_path = 'PPO_cartpole_data/'
data_path = 'PPO_cartpole_data_1k/'

# agent config
STEP_UPDATE = 120
input_dims = env.observation_space.shape
gamma = 0.99
alpha = 0.0003
gae_lambda = 0.95
policy_clip = 0.2
batch_size = 5
n_epochs = 4

model_chkpt_path = data_path+'chkpts/'
act_dim_1 = 200
act_dim_2 = 200
actor_dims = [*input_dims, act_dim_1, act_dim_2]

crit_dim_1 = 64
crit_dim_2 = 64
critic_dims = [*input_dims, crit_dim_1, crit_dim_2]


# training config
threshold = 475
is_threshold_stop = False
running_avg_len = 100
is_progress = True
EPISODES = 300
EPISODES = 1000

# load prev trained models
trained_actor_path = ''
trained_critic_path = ''
trained_model_path = [trained_actor_path, trained_critic_path]
load_prev_trained = False
best_score = 0

const = {
    'env': env,
    'action_table': action_table,
    'PATH': data_path
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
    'verbose': True,
    'model_chkpt_path': model_chkpt_path,
    'actor_dims': actor_dims,
    'critic_dims': critic_dims
}


training_configs = {
    'agent_type': PPOAgent,
    'EPISODES': EPISODES,
    'is_progress': is_progress,
    'threshold': threshold, 
    'is_threshold_stop': is_threshold_stop,
    'running_avg_len': running_avg_len,
    'load_prev_trained': load_prev_trained,
    'trained_model_path': trained_model_path,
    'best_score': best_score
}


CONST = Constants(const)
agent_hyper = Config(agent_config)
agent_training_configs = Config(training_configs)

base_plot_path = 'plots/'
plot_path = data_path + base_plot_path
# figure_file = plot_path+'cartpole.png'

# create utility function for creating directors
def create_dirs():
    """Create required directors for
        configuration
    """
    # create base data dir
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    # create model checkpoint dir
    if not os.path.exists(model_chkpt_path):
        os.makedirs(model_chkpt_path)
    # create plot dir
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)