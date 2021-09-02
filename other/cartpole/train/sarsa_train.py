import sys
import os
import csv

PACKAGE_PARENT = '../../../'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from alphaslime.agents.other.semiGradSarsa import SemiGradSarsa

import numpy as np


if __name__ == '__main__':
    alpha = 0.6 # step size
    epsilon = 1
    gamma = 0.995
    training_episodes = 1000
    observation_dimension=4
    action_table = [0, 1]

    # TODO: simplyfy config
    agent = SemiGradSarsa(alpha=alpha, epsilon=epsilon, gamma=gamma, d=observation_dimension, is_MA=False, env_id='CartPole-v1')
    agent.action_table = action_table
    agent.max_actions = len(agent.action_table)
    agent.MAX_SCORE = 195
    # change action space

    agent.train(training_episodes)

    weights = agent.w
    print('-'*99)
    print('final training data')
    print('self.w = {}'.format(
        agent.w.shape))

    print('epsilon = {}'.format(agent.epsilon))
    print('-'*99)

    with open('./other/cartpole/train/sarsa/weight.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        # writer.writerow(header)

        # write multiple rows
        writer.writerows(weights)

