import sys
import os
import csv

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from alphaslime.agents.selfplay.semiGradSarsa import SemiGradSarsa
# import alphaslime.agents.selfplay
import numpy as np


if __name__ == '__main__':
    alpha = 0.5/8 # step size
    epsilon = 1
    gamma = 0.999
    training_episodes = 10000*2
    agent = SemiGradSarsa(alpha=alpha, epsilon=epsilon, gamma=gamma)
    agent.train(training_episodes)

    weights = agent.w
    print('-'*99)
    print('final training data')
    print('self.w = {}'.format(
        agent.w.shape))

    print('epsilon = {}'.format(agent.epsilon))
    print('-'*99)

    with open('./train/weights_sarsa.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        # writer.writerow(header)

        # write multiple rows
        writer.writerows(weights)

