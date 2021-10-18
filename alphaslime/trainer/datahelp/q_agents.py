from torch.functional import Tensor
from alphaslime.trainer.datahelp.filename import FileName
from alphaslime.agents.agent import Agent
import torch

class QLearnFile(FileName):
    """Class that generates the base file name


    Args:
        FileName (FileName): Base Class
    """

    def __init__(self) -> None:
        super().__init__()

    def gen_name(self, agent:Agent, prefix=''):
        gamma = agent.gamma
        if type(gamma) == torch.Tensor:
            gamma = gamma.item()
        learning_rate = agent.lr

        if type(learning_rate) == torch.Tensor:
            learning_rate = learning_rate.item()

        avg_reward = agent.avg_rewards[-1]
        file_info = "gamma_{:.5}_lr_rate_{:.8}_reward_{:.5}".format(str(gamma), str(learning_rate), str(avg_reward)) 
        return file_info
