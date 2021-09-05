import sys
import os

PACKAGE_PARENT = '../../'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from alphaslime.evaluate.eval_agents import EvaluateGameSA
from alphaslime.agents.other.semiGradSarsa import SemiGradSarsa
from alphaslime.approx.linearq import LinearQApprox

from pandas import read_csv
import numpy as np

import gym

if __name__ == '__main__':

    # df = read_csv('./train/weights_sarsa.csv', header=None)
    df = read_csv('./other/cartpole/train/sarsa/weight.csv', header=None)
    weights = df.values
    print(weights)
    env_id = "CartPole-v1"

    seed = 42
    np.random.seed(seed)    

    alpha = 0.95 # step size
    epsilon = 1
    gamma = 0.995
    training_episodes = 300
    observation_dimension=4
    action_table = [0, 1]
    # q function approximator
    q_hat = LinearQApprox()
    
    env_id = 'CartPole-v1'
    env = gym.make(env_id)
    env.seed(seed)


    
    config = {
        'alpha': 0.95,
        'gamma': gamma,
        'epsilon': epsilon,
        'action_table': action_table,
        'd': observation_dimension,
        't_max': 3000,
        'max_score': 200,
        'episode_printer': 100,
        'env': env,
        'weights':None,
        'q_hat': q_hat
    }

    agent = SemiGradSarsa(config)
    agent.action_table =  [0, 1]
    agent.max_actions = len(agent.action_table)
    agent.MAX_SCORE = 200

    base_dir = './'
    RENDER = False
    eval_game = EvaluateGameSA(agent=agent, base_dir_path=base_dir, render=RENDER, env=env)

    N = 100
    # run N episodes
    agent_score = np.zeros((N,2))
    print('Start running episodes..')
    for n in range(N):

        score = eval_game.evaluate_episode()
        agent_score[n] = score


    print('*'*10)
    print('Agent right total score = {}'.format(agent_score.sum()))
    print('*'*10)

    print('*'*10)
    print('Agent(player) score: \n')
    for episode, score in enumerate(agent_score):
        print('Episode {}\t: {}'.format(episode, score))
    print('*'*10)

    print('Average score per episode = {}'.format(np.mean(agent_score)))