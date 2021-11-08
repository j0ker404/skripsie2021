"""
    Get expert trajectories from baseline agent
"""

import sys
import os

PACKAGE_PARENT = '../../../'
# sys.path.append(PACKAGE_PARENT)

SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from alphaslime.evaluate.eval_agents import EvaluateGameSA
import valid_eps_baseline_configs as BASECONFIGS



if __name__ == '__main__':

    # create required directories if not present
    BASECONFIGS.create_dirs()

    # load configurations
    CONST = BASECONFIGS.CONST
    env = BASECONFIGS.env
    
    agent_type = BASECONFIGS.eval_config['agent_type']
    render = BASECONFIGS.eval_config['render']
    time_delay = BASECONFIGS.eval_config['time_delay']

    agent = agent_type(CONST)
    base_dir_path = CONST.get('base_dir_path')

    evaluater = EvaluateGameSA(agent,env, base_dir_path, render, time_delay)
    EPISODES = BASECONFIGS.eval_config['EPISODES']
    is_progress_bar = BASECONFIGS.eval_config['is_progress_bar']
    running_avg_len = BASECONFIGS.eval_config['running_avg_len']
    print('Start generating expert trajectories...')
    evaluater.evaluate(EPISODES, is_progress_bar, running_avg_len)
    print('Stop generating expert trajectories...')

