"""
    Training configurations for
    PPO agent for single agent slimevolley ball
"""
import sys
import os

PACKAGE_PARENT = '../../../'
sys.path.append(PACKAGE_PARENT)

from alphaslime.agents.selfplay.policygrad.torch.ppoSP import PPO_SP
from alphaslime.store.constantConfig import Constants
from alphaslime.store.config import Config
import gym
import slimevolleygym

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

data_path = 'PPO_data/'

# agent config
STEP_UPDATE = 4096
input_dims = env.observation_space.shape
gamma = 0.99
alpha = 0.0003
gae_lambda = 0.95
policy_clip = 0.2
batch_size = 64
n_epochs = 10

model_chkpt_path = data_path+'chkpts/'
act_dim_1 = 64
act_dim_2 = 64
actor_dims = [*input_dims, act_dim_1, act_dim_2]

crit_dim_1 = 64
crit_dim_2 = 64
critic_dims = [*input_dims, crit_dim_1, crit_dim_2]


# training config
threshold = 0
is_threshold_stop = False
running_avg_len = 100
is_progress = True
EPISODES = 300
EPISODES = 1000

champ_dir = data_path + 'champ/'
champ_threshold = 0.5
champ_min_avg_rew = -5

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
    'agent_type': PPO_SP,
    'EPISODES': EPISODES,
    'is_progress': is_progress,
    'threshold': threshold, 
    'is_threshold_stop': is_threshold_stop,
    'running_avg_len': running_avg_len,
    'champ_dir': champ_dir,
    'champ_threshold': champ_threshold,
    'champ_min_avg_rew': champ_min_avg_rew
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
    # create champ dir
    if not os.path.exists(champ_dir):
        os.makedirs(champ_dir)