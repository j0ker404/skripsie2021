from alphaslime.trainer.datahelp.filename import FileName
from alphaslime.agents.agent import Agent
import torch

class BCLearnFile(FileName):
    """Class that generates the base file name

        Filenamer for BC agents

    Args:
        FileName (FileName): Base Class
    """

    def __init__(self) -> None:
        super().__init__()

    def gen_name(self, agent:Agent, prefix=''):
        alpha = agent.alpha

        if type(alpha) == torch.Tensor:
            alpha = alpha.item()

        min_loss = agent.min_loss_per_eps
        file_info = "alpha_{:.8}_loss_{:.5}".format(str(alpha), str(min_loss)) 
        return file_info