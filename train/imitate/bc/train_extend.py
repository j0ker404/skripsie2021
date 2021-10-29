"""
    Train BC agent with baseline agent
"""

import sys
import os

PACKAGE_PARENT = '../../../'
# sys.path.append(PACKAGE_PARENT)

SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from alphaslime.trainer.trainerSA import TrainerSA as Trainer
import bc_training_configs_extend as BCCONFIGS
from alphaslime.trainer.datahelp.bc_agents import BCLearnFile


if __name__ == '__main__':

    # create required directories if not present
    BCCONFIGS.create_dirs()

    # load configurations
    CONST = BCCONFIGS.CONST
    agent_config = BCCONFIGS.agent_hyper
    env = BCCONFIGS.env
    agent_training_configs = BCCONFIGS.agent_training_configs

    filenamer = BCLearnFile()

    trainer = Trainer(CONSTANTS=CONST)
    print('Start Training')
    filenames = trainer.train(training_config=agent_training_configs, agent_config=agent_config, fileNamer=filenamer)
    print('End Training')

    print(filenames)