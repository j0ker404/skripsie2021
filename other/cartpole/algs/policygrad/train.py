"""
    Train PPO agent with cartpole
"""

import sys
import os

PACKAGE_PARENT = '../../../../'
# sys.path.append(PACKAGE_PARENT)

SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from alphaslime.trainer.trainerSA import TrainerSA as Trainer
import ppo_training_configs_cartpole as PPOCONFIGS
from alphaslime.trainer.datahelp.pg_agents import PGLearnFile


if __name__ == '__main__':

    # create required directories if not present
    PPOCONFIGS.create_dirs()

    # load configurations
    CONST = PPOCONFIGS.CONST
    agent_config = PPOCONFIGS.agent_hyper
    env = PPOCONFIGS.env
    agent_training_configs = PPOCONFIGS.agent_training_configs

    filenamer = PGLearnFile()
    print('start PPO cartpole training')
    trainer = Trainer(CONSTANTS=CONST)
    print('stop PPO cartpole training')

    filenames = trainer.train(training_config=agent_training_configs, agent_config=agent_config, fileNamer=filenamer)

    print(filenames)