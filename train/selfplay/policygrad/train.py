"""
    PPO trainer main funciton
"""
import sys

PACKAGE_PARENT = '../../../'
sys.path.append(PACKAGE_PARENT)

from alphaslime.agents.RL.policygrad.torch.ppo import PPOAgent
from alphaslime.trainer.trainerSA import TrainerSA as Trainer
import pposp_configs as PPOCONFIGS
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

    trainer = Trainer(CONSTANTS=CONST)

    filenames = trainer.train(training_config=agent_training_configs, agent_config=agent_config, fileNamer=filenamer)

    print(filenames)