import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from alphaslime.agents.selfplay.semiGradSarsa import SemiGradSarsa
# import alphaslime.agents.selfplay

if __name__ == '__main__':
    alpha = 1/10 # step size
    epsilon = 1
    training_episodes = 10
    agent = SemiGradSarsa(alpha=alpha, epsilon=epsilon)
    agent.train(training_episodes)