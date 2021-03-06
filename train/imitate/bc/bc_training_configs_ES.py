"""
    Training configurations for
    BC agent for single agent slimevolley ball
    with early stopping
"""
import sys
import os

PACKAGE_PARENT = '../../../'
sys.path.append(PACKAGE_PARENT)

import gym
import slimevolleygym
from alphaslime.store.constantConfig import Constants
from alphaslime.store.config import Config

from alphaslime.agents.imitate.torch.bcAgentES import BCAgentES
from alphaslime.agents.imitate.torch.bc import BCPolicyNet

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

expert_episodes_data_path = 'Expert_Episodes/'
model_data_path = 'BC_data_ES/'

# agent config
alpha = 0.0003
batch_size = 64
n_epochs = 200
input_dims = env.observation_space.shape
layer_dim_1 = 64
layer_dim_2 = 64
output_dims = len(action_table)

model_chkpt_path = model_data_path+'chkpts/'
bc_layer_sizes = [*input_dims, layer_dim_1, layer_dim_2, output_dims]


# training config
threshold = 0
is_threshold_stop = False
running_avg_len = 100
is_progress = True
EPISODES = 1000
# ES training configs
patience = 20
valid_episodes_data_path = 'Valid_Expert_Episodes/'
EPISODES_VALID = 250
ES_CHKPT_PATH = model_data_path + 'valid_chpt.pt'



# load prev trained model
trained_model_path = model_data_path + ''
load_prev_trained = False

const = {
    'env': env,
    'action_table': action_table,
    'PATH': model_data_path
}



agent_config = {
    'alpha':alpha,
    'batch_size': batch_size,
    'n_epochs': n_epochs,
    'verbose': True,
    'model_chkpt_path': model_chkpt_path,
    'policy_dims': bc_layer_sizes,
    'BCNetType': BCPolicyNet
}


training_configs = {
    'agent_type': BCAgentES,
    'EPISODES': EPISODES,
    'shuffle_batches': True,
    'expert_episodes_data_path': expert_episodes_data_path,
    'is_progress': is_progress,
    'threshold': threshold, 
    'is_threshold_stop': is_threshold_stop,
    'running_avg_len': running_avg_len,
    'load_prev_trained': load_prev_trained,
    'trained_model_path': trained_model_path,
    'patience': patience,
    'valid_episodes_data_path': valid_episodes_data_path,
    'EPISODES_VALID': EPISODES_VALID,
    'ES_CHKPT_PATH': ES_CHKPT_PATH,
}


CONST = Constants(const)
agent_hyper = Config(agent_config)
agent_training_configs = Config(training_configs)


# create utility function for creating directors
def create_dirs():
    """Create required directors for
        configuration
    """
    # create base data dir
    if not os.path.exists(model_data_path):
        os.makedirs(model_data_path)
    # create model checkpoint dir
    if not os.path.exists(model_chkpt_path):
        os.makedirs(model_chkpt_path)
    # create expert eps dir
    if not os.path.exists(expert_episodes_data_path):
        os.makedirs(expert_episodes_data_path)
    # create valid dir
    if not os.path.exists(valid_episodes_data_path):
        os.makedirs(valid_episodes_data_path)