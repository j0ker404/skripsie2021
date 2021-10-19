"""Utility ploting functions
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve_plus_score(x, scores, figure_file, running_avg_len=100, title='Running average'):
    """Plot the total episode rewards as well as the running episode average

        Plot will be saved to disk
    Args:
        x ([int]): episodes
        scores ([float]): reward per episode
        figure_file ([string]): Path to save file
    """
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-running_avg_len):(i+1)])
    plt.plot(x, scores, label='Total Reward per episode')
    plt.plot(x, running_avg, label='Running total reward average over {} episodes'.format(running_avg_len))
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward per episode')
    plt.legend(bbox_to_anchor=(1.05, 1)) 
    plt.title(title)
    plt.savefig(figure_file)
    plt.show()
