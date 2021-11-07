"""
    Trained agents for Slime Volleyball environment
"""

import sys
import os
PACKAGE_PARENT = '../'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from alphaslime.agents.RL.policygrad.torch.ppo import PPOAgent
from alphaslime.agents.imitate.torch.bcAgent import BCAgent
from alphaslime.agents.baseline import BaselineAgent
import gym
import slimevolleygym


def load_agents():
    """Load trained agents

    Returns:
        agents: List of trained agents
    """
    # store are trained agents in a list
    agents = []

    # import os
    # cwd = os.getcwd()
    # print(cwd)
    # print(SCRIPT_DIR)

    # load PPO agent trained with baseline
    print('loading: ppo_agent')
    import trained_agents.ppo.slime.ppo_training_configs_cont as PPOCONFIGS
    CONST = PPOCONFIGS.CONST
    agent_config = PPOCONFIGS.agent_config
    act_path =  os.path.join(SCRIPT_DIR,'ppo/slime/gamma_0.99_alpha_0.0003_reward_-0.22_model_actor.pt')
    crt_path =  os.path.join(SCRIPT_DIR,'ppo/slime/gamma_0.99_alpha_0.0003_reward_-0.22_model_critic.pt')
    paths = [act_path, crt_path]
    ppo_agent = PPOAgent(CONSTANTS=CONST, config=agent_config)
    ppo_agent.load_model(paths)
    print('-'*10)

    # load Self-play PPO agent trained
    print('loading: sp_ppo_agent')
    import trained_agents.selfplay.slime.no_boots.pposp_configs_cont as SP_PPOCONFIGS
    CONST = SP_PPOCONFIGS.CONST
    agent_config = SP_PPOCONFIGS.agent_config
    act_path = os.path.join(SCRIPT_DIR,'selfplay/slime/no_boots/gamma_0.99_alpha_0.0003_reward_4.58_model_actor.pt')
    crt_path = os.path.join(SCRIPT_DIR,'selfplay/slime/no_boots/gamma_0.99_alpha_0.0003_reward_4.58_model_critic.pt')
    paths = [act_path, crt_path]  
    sp_ppo_agent = PPOAgent(CONSTANTS=CONST, config=agent_config)
    sp_ppo_agent.load_model(paths)
    print('-'*10)

    # load bootstrapped PPO agent trained
    print('loading: boots_ppo_agent')
    import trained_agents.selfplay.slime.boots.pposp_configs_boots as BOOTS_PPOCONFIGS
    CONST = BOOTS_PPOCONFIGS.CONST
    agent_config = BOOTS_PPOCONFIGS.agent_config
    act_path = os.path.join(SCRIPT_DIR,'selfplay/slime/boots/gamma_0.99_alpha_0.0003_reward_4.61_model_actor.pt')
    crt_path = os.path.join(SCRIPT_DIR,'selfplay/slime/boots/gamma_0.99_alpha_0.0003_reward_4.61_model_critic.pt')
    paths = [act_path, crt_path]
    boots_ppo_agent = PPOAgent(CONSTANTS=CONST, config=agent_config)
    boots_ppo_agent.load_model(paths)
    print('-'*10)

    # load BC agent trained
    print('loading: bc_agent')
    import trained_agents.imitate.slime.bc_training_configs_extend as BCCONFIGS
    CONST = BCCONFIGS.CONST
    agent_config = BCCONFIGS.agent_config
    path = os.path.join(SCRIPT_DIR,'imitate/slime/alpha_0.0003_loss_7061._model_bc.pt')
    bc_agent = BCAgent(CONSTANTS=CONST, config=agent_config)
    bc_agent.load_model(path)
    print('-'*10)


    # store agents in list
    agents.append(['PPO_Agent', ppo_agent])
    agents.append(['Self-play_PPO_Agent', sp_ppo_agent])
    agents.append(['Bootstrapped_Self-play_PPO_Agent', boots_ppo_agent])
    agents.append(['BC_Agent', bc_agent])

    return agents


def agents_dict():
    """Return dictionary of trained agents
    """

    agents = {}
    agents_list = load_agents()
    for name, agent in agents_list:
        agents[name] = agent

    return agents 

def load_baseline():
    """Load baseline agent

    Returns:
        Agent: Baselinea agent
    """
    env_id = "SlimeVolley-v0"
    env = gym.make(env_id)

    # actions for slimeball
    action_table = [[0, 0, 0], # NOOP
                    [1, 0, 0], # LEFT (forward)
                    [1, 0, 1], # UPLEFT (forward jump)
                    [0, 0, 1], # UP (jump)
                    [0, 1, 1], # UPRIGHT (backward jump)
                    [0, 1, 0]] # RIGHT (backward)
    const = {
        'env': env,
        'action_table': action_table,
    }

    baseline = BaselineAgent(const)

    return baseline
