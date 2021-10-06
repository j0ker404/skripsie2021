from typing import Dict
from alphaslime.agents.agent import Agent
from alphaslime.trainer.trainer import Trainer

import os
import torch
import pickle


class TrainerSA(Trainer):
    """ Trainer for a single agent 
        with a NN q-function approximator

    Args:
        Trainer ([type]): [description]


        path: (str) base path for saving data, in the form of './path/' 
    """
    def __init__(self, CONSTANTS: dict) -> None:
        super().__init__(CONSTANTS)
        # PATH: BASE PATH to save data
        self.BASE_PATH = self.CONSTANTS['PATH']
        # actions used with agent in the gym environment
        self.action_table = self.CONSTANTS['action_table']
        # gym environment
        self.env = self.CONSTANTS['env']
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
    
    def train(self, hyperparams: dict):
        """Train agent

        Args:
            hyperparams (dict): hyper parameters for training agent


        return: filenames:list, list of saved file
        """
        gamma = hyperparams['gamma']
        epsilon = hyperparams['epsilon']
        learning_rate = hyperparams['learning_rate']
        EPISODES = hyperparams['EPISODES']
        MINI_BATCH_SIZE = hyperparams['MINI_BATCH_SIZE']
        MEMORY_SIZE = hyperparams['MEMORY_SIZE']
        TARGET_UPDATE = hyperparams['TARGET_UPDATE']
        epsilon_decay_model = hyperparams['epsilon_decay']
        reward_threshold = hyperparams['threshold']
        is_progress = hyperparams['is_progress']
        layer_sizes = hyperparams['layer_sizes']
        q_type = hyperparams['q_type']
        agent_type = hyperparams['agent_type']

        # q function approximator
        q_hat = q_type(lr=learning_rate, layer_sizes=layer_sizes, device=self.device).to(self.device)

        # set config file for agent
        config = {
            'lr': learning_rate,
            'gamma': gamma,
            'epsilon': epsilon,
            'action_table': self.action_table,
            'env': self.env,
            'q_hat': q_hat,
            'batch_size': MINI_BATCH_SIZE,
            'exp_mem_size': MEMORY_SIZE,
            'TARGET_UPDATE': TARGET_UPDATE,
            'epsilon_decay': epsilon_decay_model,
            'q_type': q_type,
        }

        # agent 
        agent = agent_type(config)
        # train agent
        avg_rewards = agent.train(EPISODES, is_progress=is_progress, threshold=reward_threshold)

        return self.save_data(avg_rewards, agent, hyperparams)

    def save_data(self, avg_rewards, agent:Agent, hyperparams:dict, agent_config:dict):
        '''
            Save training data to disk

            return: filenames:list, list of saved filenames
        '''
        filenames = []
        learning_rate = agent.q_model.learning_rate
        gamma = agent.gamma.item()
        avg_reward = avg_rewards[-1]

        model_info = "gamma_{}_lr_rate_{}_reward_{}".format(str(gamma), str(learning_rate), str(avg_reward))

        path = self.BASE_PATH + model_info + '_model' + '.pt'
        filenames.append(path)
        # save model
        agent.save_q_model(path)

        # save training data
        training_data = {
            'avg_rewards': avg_rewards,
            'rewards': agent.rewards,
            'epsilon': agent.epsilon_list,
            'losses': agent.loss_list
        }
        path = self.BASE_PATH  + model_info + '_data' + '.pkl'
        filenames.append(path)
        with open(path, 'wb') as f:
            pickle.dump(training_data, f)


        # save hyperparams
        path = self.BASE_PATH  + model_info + '_hyper' + '.pkl'
        filenames.append(path)
        with open(path, 'wb') as f:
            pickle.dump(hyperparams, f)

        # save agent config
        path = self.BASE_PATH  + model_info + '_agent_cfg' + '.pkl'
        filenames.append(path)
        with open(path, 'wb') as f:
            pickle.dump(agent_config, f)

        # save CONSTANTS
        path = self.BASE_PATH  + model_info + '_CONSTANTS' + '.pkl'
        filenames.append(path)
        with open(path, 'wb') as f:
            pickle.dump(self.CONSTANTS, f)

        return filenames

