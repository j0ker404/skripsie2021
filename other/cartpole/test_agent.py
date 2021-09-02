import sys
import os

PACKAGE_PARENT = '../../'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from alphaslime.evaluate.eval_agents import EvaluateGameSA
from alphaslime.agents.other.semiGradSarsa import SemiGradSarsa

from pandas import read_csv
import numpy as np




if __name__ == '__main__':

    # df = read_csv('./train/weights_sarsa.csv', header=None)
    df = read_csv('./other/cartpole/train/sarsa/weight.csv', header=None)
    weights = df.values
    print(weights)

    # agent_right = SemiGradSarsa(epsilon=0.0, weights=weights)
    # agent = SemiGradSarsa(epsilon=0.0, weights=weights)
    agent = SemiGradSarsa(epsilon=0.0, weights=weights, is_MA=False, env_id='CartPole-v1')
    agent.action_table =  [0, 1]
    agent.max_actions = len(agent.action_table)
    agent.MAX_SCORE = 195

    base_dir = './'
    RENDER = False
    env_id = "CartPole-v0"
    eval_game = EvaluateGameSA(agent=agent, base_dir_path=base_dir, render=RENDER, env_id=env_id)

    N = 10
    # run N episodes
    agent_score = np.zeros((N,))
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