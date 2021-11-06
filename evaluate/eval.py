import sys
import os
PACKAGE_PARENT = '../'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


from alphaslime.evaluate.eval_agents import EvaluateGameMA
from alphaslime.agents.RL.policygrad.torch.ppo import PPOAgent
from alphaslime.agents.imitate.torch.bcAgent import BCAgent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gym
import slimevolleygym
import pickle

# constant config
env_id = "SlimeVolley-v0"
env = gym.make(env_id)


def load_agents():
    """Load trained agents

    Returns:
        agents: List of trained agents
    """
    # store are trained agents in a list
    agents = []

    # load PPO agent trained with baseline
    print('loading: ppo_agent')
    import trained_agents.ppo.slime.ppo_training_configs_cont as PPOCONFIGS
    CONST = PPOCONFIGS.CONST
    agent_config = PPOCONFIGS.agent_config
    act_path = './trained_agents/ppo/slime/gamma_0.99_alpha_0.0003_reward_-0.22_model_actor.pt'
    crt_path = './trained_agents/ppo/slime/gamma_0.99_alpha_0.0003_reward_-0.22_model_critic.pt'
    paths = [act_path, crt_path]
    ppo_agent = PPOAgent(CONSTANTS=CONST, config=agent_config)
    ppo_agent.load_model(paths)
    print('-'*10)

    # load Self-play PPO agent trained
    print('loading: sp_ppo_agent')
    import trained_agents.selfplay.slime.no_boots.pposp_configs_cont as SP_PPOCONFIGS
    CONST = SP_PPOCONFIGS.CONST
    agent_config = SP_PPOCONFIGS.agent_config
    act_path = './trained_agents/selfplay/slime/no_boots/gamma_0.99_alpha_0.0003_reward_4.58_model_actor.pt'
    crt_path = './trained_agents/selfplay/slime/no_boots/gamma_0.99_alpha_0.0003_reward_4.58_model_critic.pt'
    paths = [act_path, crt_path]
    sp_ppo_agent = PPOAgent(CONSTANTS=CONST, config=agent_config)
    sp_ppo_agent.load_model(paths)
    print('-'*10)

    # load bootstrapped PPO agent trained
    print('loading: boots_ppo_agent')
    import trained_agents.selfplay.slime.boots.pposp_configs_boots as BOOTS_PPOCONFIGS
    CONST = BOOTS_PPOCONFIGS.CONST
    agent_config = BOOTS_PPOCONFIGS.agent_config
    act_path = './trained_agents/selfplay/slime/boots/gamma_0.99_alpha_0.0003_reward_4.61_model_actor.pt'
    crt_path = './trained_agents/selfplay/slime/boots/gamma_0.99_alpha_0.0003_reward_4.61_model_critic.pt'
    paths = [act_path, crt_path]
    boots_ppo_agent = PPOAgent(CONSTANTS=CONST, config=agent_config)
    boots_ppo_agent.load_model(paths)
    print('-'*10)

    # load BC agent trained
    print('loading: bc_agent')
    import trained_agents.imitate.slime.bc_training_configs_extend as BCCONFIGS
    CONST = BCCONFIGS.CONST
    agent_config = BCCONFIGS.agent_config
    path = './trained_agents/imitate/slime/alpha_0.0003_loss_7061._model_bc.pt'
    bc_agent = BCAgent(CONSTANTS=CONST, config=agent_config)
    bc_agent.load_model(path)
    print('-'*10)


    # store agents in list
    agents.append(['PPO_Agent', ppo_agent])
    agents.append(['Self-play_PPO_Agent', sp_ppo_agent])
    agents.append(['Bootstrapped_Self-play_PPO_Agent', boots_ppo_agent])
    agents.append(['BC_Agent', bc_agent])

    return agents


if __name__ == '__main__':
    eps = 1000
    base_dir_path = "./evaluate_data/"
    RENDER = False
    running_avg_len = 100
    match_data = {}

    match_data_filename = 'match_data.pkl'
    if not os.path.exists(base_dir_path):
        os.makedirs(base_dir_path)

    agents = load_agents()
    # Agent VS Agents
    eps = 1000
    base_dir_path = "./evaluate_data/"
    RENDER = False
    running_avg_len = 100
    match_data = {}
    for agent_right_name, agent_right in agents:
        for agent_left_name, agent_left in agents:

            gym_evaluator = EvaluateGameMA(agent_right, agent_left, env, base_dir_path, render=RENDER, time_delay=0)

            # evaulate agent
            rewards, avg_rewards_array = gym_evaluator.evaluate(eps, is_progress_bar=True, running_avg_len=running_avg_len)

            data_name = agent_right_name + '_vs_' + agent_left_name
            match_data[data_name] = [rewards, avg_rewards_array]
    


    match_data_filename = 'match_data.pkl'

    # save file
    if not os.path.exists(base_dir_path):
        os.makedirs(base_dir_path)


    eval_path = os.path.join(base_dir_path, match_data_filename)
    with open(eval_path, 'wb') as f:
        pickle.dump(match_data, f)   
