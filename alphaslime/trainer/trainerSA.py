from sys import path
from alphaslime.agents.agent import Agent
from alphaslime.store.config import Config
from alphaslime.trainer.datahelp.filename import FileName
from alphaslime.trainer.trainer import Trainer
from alphaslime.store.constantConfig import Constants


import os
import torch
import pickle

# TODO: use Config instances for all params when possible
class TrainerSA(Trainer):
    """ Trainer for a single agent 
        with a NN q-function approximator

    Args:
        Trainer ([type]): [description]


        path: (str) base path for saving data, in the form of './path_dir/' 
    """
    def __init__(self, CONSTANTS: Constants) -> None:
        super().__init__(CONSTANTS)
        # PATH: BASE PATH to save data
        self.BASE_PATH = self.CONSTANTS.get('PATH')
        # actions used with agent in the gym environment
        self.action_table = self.CONSTANTS.get('action_table')
        # gym environment
        self.env = self.CONSTANTS.get('env')
        # action and observation space sizes
        self.n_actions = self.env.action_space.n
        self.len_obs_space = self.env.observation_space.shape[0]


        # create directory if not present
        if not os.path.exists(self.BASE_PATH):
            os.makedirs(self.BASE_PATH)

        
        # if gpu is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # def train(self, hyperparams: dict):
    def train(self, training_config:Config, agent_config:Config, fileNamer:FileName, prefix=''):
        """Train agent

        Args:
            hyperparams (dict): hyper parameters for training agent

            training_config: Contains at least the entry for the 
                            'agent_type'

        return: filenames:list, list of saved file
        """
        
        # agent
        agent_type = training_config.get('agent_type') 
        agent = agent_type(self.CONSTANTS,agent_config)
        is_load_models = training_config.get('load_prev_trained')  
        if is_load_models:
            # load pretrained models
            path = training_config.get('trained_model_path')
            agent.load_model(path)
        # train agent
        agent.train(training_config)

        # return self.save_data(avg_rewards, agent, train, config)
        return self.save_data(agent, training_config, agent_config, fileNamer, prefix)

    def save_data(self, agent:Agent, training_config:Config, agent_config:Config, fileNamer:FileName, prefix=''):
        '''
            Save all training data to disk

            data saved:
                - (0) agent trained model
                - (1) agent training data
                - (2) agent configuration: agent_config:Config
                - (3) training configuration: training_config:Config
                - (4) CONSTANTS configuraion: CONSTANTS:Constants
            

            return: filenames:list, list of saved filenames. See data saved list for
                                    for the filename index order

        '''
        filenames = []

        file_info = fileNamer.gen_name(agent=agent, prefix=prefix)

        # save model
        path = self.BASE_PATH + file_info + '_model'
        filenames.append(path)
        agent.save_model(path)

        # save training data
        path = self.BASE_PATH  + file_info + '_data' + '.pkl'
        filenames.append(path)
        agent.save_training_data(path)


        # save agent config
        path = self.BASE_PATH  + file_info + '_agent_cfg' + '.pkl'
        filenames.append(path)
        agent_config.save(path)

        # save training config
        path = self.BASE_PATH  + file_info + '_train_cfg' + '.pkl'
        filenames.append(path)
        training_config.save(path)

        # save CONSTANTS
        path = self.BASE_PATH  + file_info + '_CONSTANTS' + '.pkl'
        filenames.append(path)
        self.CONSTANTS.save(path)

        return filenames
