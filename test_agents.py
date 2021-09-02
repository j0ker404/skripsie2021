from alphaslime.evaluate.eval_agents import EvaluateGame
from alphaslime.agents.baseline import BaselineAgent
from alphaslime.agents.other.semiGradSarsa import SemiGradSarsa
from alphaslime.agents.selfplay.sarsa import SarsaSP  

from pandas import read_csv
import numpy as np

if __name__ == '__main__':

    # df = read_csv('./train/weights_sarsa.csv', header=None)
    df = read_csv('./train/selfplay/sarsa/weight.csv', header=None)
    weights = df.values
    print(weights)

    # agent_right = SemiGradSarsa(epsilon=0.0, weights=weights)
    agent_right = SarsaSP(epsilon=0.0, weights=weights)
    agent_left = BaselineAgent()
    base_dir = './'
    RENDER = True
    env_id = "SlimeVolley-v0"
    # env_id = "SlimeVolleyPixel-v0"
    eval_game = EvaluateGame(agent_right=agent_right, agent_left=agent_left, base_dir_path=base_dir, render=RENDER, env_id=env_id)

    N = 50
    # run N episodes
    agent_right_score = np.zeros((N,))
    agent_left_score = np.zeros((N,))
    print('Start running episodes..')
    for n in range(N):
        # print('-'*99)
        # print('Evaulating episode {}'.format(n))
        score = eval_game.evaluate_episode()
        agent_right_score[n] = score
        agent_left_score[n] = -score
        # print('Agent right  score = {}'.format(score))
        # print('Agent left  score = {}'.format(-score))
        # print('-'*99)
    

    print('*'*10)
    print('Agent right total score = {}'.format(agent_right_score.sum()))
    print('Agent left total score = {}'.format(agent_left_score.sum()))
    print('*'*10)

    print('*'*10)
    print('Agent(player) score: \n')
    for episode, score in enumerate(agent_right_score):
        print('Episode {}\t: {}'.format(episode, score))
    print('*'*10)