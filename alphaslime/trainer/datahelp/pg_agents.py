from alphaslime.trainer.datahelp.filename import FileName
from alphaslime.agents.agent import Agent
import torch

class PGLearnFile(FileName):
    """Class that generates the base file name

        Filenamer for Policy Gradient agents

    Args:
        FileName (FileName): Base Class
    """

    def __init__(self) -> None:
        super().__init__()

    def gen_name(self, agent:Agent, prefix=''):
        gamma = agent.gamma
        if type(gamma) == torch.Tensor:
            gamma = gamma.item()
        alpha = agent.alpha

        if type(alpha) == torch.Tensor:
            alpha = alpha.item()

        avg_reward = agent.avg_rewards[-1]
        file_info = "gamma_{}_alpha_{}_reward_{}".format(str(gamma), str(alpha), str(avg_reward)) 
        return file_info