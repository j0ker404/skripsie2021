"""In this class we will load champion
agents from disk and be able to sample
champions

Based on the champion sampling discussed in:
Bansal et al., Emergent complexity via multi-agent competition, ICLR, 2018.
"""

from alphaslime.agents.agent import Agent
from alphaslime.store.config import Config


class Champions:

    def __init__(self, champ_dir:str) -> None:
        """Constructor

        Args:
            champ_dir (str): Directory that contains the champions
                            in the form of './path_dir/'
        """
        self.champ_dir = champ_dir
        self.champ_counter = 0

    def sample(self) -> str:
        """Get the model path of the 
            sampled champion

        Returns:
            str: Path of the champion's model
        """
        pass

    def append(self, agent:Agent):
        """Append agent to champion list 
        and save champion to disk

        Args:
            agent (Agent): [description]
        """
        pass