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

        # self.q_type = q_type
        # self.agent = agent

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


        return: filenames:list, list of saved file
        """
        
        # agent
        agent_type = training_config.get('agent_type') 
        agent = agent_type(self.CONSTANTS,agent_config)
        # train agent
        # avg_rewards = agent.train(EPISODES, is_progress=is_progress, threshold=reward_threshold, is_threshold_stop=reward_is_threshold_stop)
        # avg_rewards = agent.train(training_config)
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
        # learning_rate = agent_config.get('lr')
        # gamma = agent_config.get('gamma')
        # avg_reward = agent.avg_rewards[-1]

        # file_info = "gamma_{}_lr_rate_{}_reward_{}".format(str(gamma), str(learning_rate), str(avg_reward))
        file_info = fileNamer.gen_name(agent=agent, prefix=prefix)

        # save model
        path = self.BASE_PATH + file_info + '_model' + '.pt'
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


    # def save_data(self, avg_rewards, agent:Agent, training_config:Config, agent_config:Config):
    #     '''
    #         Save training data to disk

    #         return: filenames:list, list of saved filenames

    #         TODO: There is an error pickling Wrappers of Env,
    #             Currently just saving string version of env object
    #     '''
    #     filenames = []
    #     learning_rate = agent.q_model.learning_rate
    #     gamma = agent.gamma.item()
    #     avg_reward = avg_rewards[-1]

    #     model_info = "gamma_{}_lr_rate_{}_reward_{}".format(str(gamma), str(learning_rate), str(avg_reward))

    #     path = self.BASE_PATH + model_info + '_model' + '.pt'
    #     filenames.append(path)
    #     # save model
    #     agent.save_q_model(path)

    #     # save training data
    #     training_data = {
    #         'avg_rewards': avg_rewards,
    #         'rewards': agent.rewards,
    #         'epsilon': agent.epsilon_list,
    #         'losses': agent.loss_list
    #     }
    #     path = self.BASE_PATH  + model_info + '_data' + '.pkl'
    #     filenames.append(path)
    #     with open(path, 'wb') as f:
    #         pickle.dump(training_data, f)


    #     # save hyperparams
    #     path = self.BASE_PATH  + model_info + '_hyper' + '.pkl'
    #     filenames.append(path)
    #     with open(path, 'wb') as f:
    #         pickle.dump(hyperparams, f)

    #     #NOTE that we convert env obj to string in place for pickling

    #     # save agent config
    #     path = self.BASE_PATH  + model_info + '_agent_cfg' + '.pkl'
    #     filenames.append(path)
    #     # print(agent_config)
    #     # agent_config.pop('env')
    #     agent_config['env'] = str(agent_config['env'])
    #     # print(agent_config)
    #     with open(path, 'wb') as f:
    #         pickle.dump(agent_config, f)

    #     # # save CONSTANTS
    #     path = self.BASE_PATH  + model_info + '_CONSTANTS' + '.pkl'
    #     filenames.append(path)
    #     # # self.CONSTANTS.pop('env')
    #     self.CONSTANTS['env'] = str(self.CONSTANTS['env'])
    #     with open(path, 'wb') as f:
    #         pickle.dump(self.CONSTANTS, f)

    #     return filenames

