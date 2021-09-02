from alphaslime.evaluate.eval_agents import EvaluateGameSA
from alphaslime.agents.other.semiGradSarsa import SemiGradSarsa

from pandas import read_csv
import numpy as np

if __name__ == '__main__':

    # df = read_csv('./train/weights_sarsa.csv', header=None)
    df = read_csv('./other/train/sarsa/weight.csv', header=None)
    weights = df.values
    print(weights)

    # agent_right = SemiGradSarsa(epsilon=0.0, weights=weights)
    agent = SemiGradSarsa(epsilon=0.0, weights=weights)
    base_dir = './'
    RENDER = True
    env_id = "SlimeVolley-v0"
    # env_id = "SlimeVolleyPixel-v0"
    eval_game = EvaluateGameSA(agent==agent, base_dir_path=base_dir, render=RENDER, env_id=env_id)

    N = 50
    # run N episodes
    agent_right_score = np.zeros((N,))
    print('Start running episodes..')
    for n in range(N):

        score = eval_game.evaluate_episode()
        agent_right_score[n] = score


    print('*'*10)
    print('Agent right total score = {}'.format(agent_right_score.sum()))
    print('*'*10)

    print('*'*10)
    print('Agent(player) score: \n')
    for episode, score in enumerate(agent_right_score):
        print('Episode {}\t: {}'.format(episode, score))
    print('*'*10)