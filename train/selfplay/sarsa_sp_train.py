import sys
import os
import csv

PACKAGE_PARENT = '../../'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from alphaslime.approx.linearq import LinearQApprox
from alphaslime.agents.selfplay.sarsa import SarsaSP


import numpy as np


if __name__ == '__main__':
    alpha = 3e-4 # step size
    epsilon = 1
    gamma = 0.99
    training_episodes = 100
    champion_trails = 10

    q_hat = LinearQApprox()
    agent = SarsaSP(alpha=alpha, epsilon=epsilon, gamma=gamma, q_hat=q_hat)
    agent.train(episodes=training_episodes, num_champions_train=champion_trails)

    weights = agent.w
    print('-'*99)
    print('final training data')
    print('self.w = {}'.format(
        agent.w.shape))

    print('epsilon = {}'.format(agent.epsilon))
    print('-'*99)

    with open('./train/selfplay/sarsa/weight.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        # writer.writerow(header)

        # write multiple rows
        writer.writerows(weights)

