from alphaslime.agents.agent import Agent
from champion import Champions

import os
import fnmatch
import random
class PPOChampions(Champions):

    def __init__(self, champ_dir: str) -> None:
        super().__init__(champ_dir)
        self.file_start = 'champ_'
        self.base_path = self.champ_dir + self.file_start
    

    def sample(self) -> list:
        """Get the model path of the 
            sampled champion

        Returns:
            list: Paths of the champion's model
        """
        ran_index = random.randint(0, self.champ_counter-1)
        pattern = self.base_path + str(ran_index)
        paths = PPOChampions.find(pattern, self.champ_dir)
        if len(paths) == 0:
            return None
        return paths

    def append(self, agent:Agent):
        """Append agent to champion list 
        and save champion to disk

        Args:
            agent (Agent): [description]
        """
        base_path = self.base_path + str(self.champ_counter)
        agent.save_model(base_path)
        self.champ_counter += 1

    # https://stackoverflow.com/questions/1724693/find-a-file-in-python
    def find_all(name, path):
        result = []
        for root, dirs, files in os.walk(path):
            if name in files:
                result.append(os.path.join(root, name))
        return result

    def find(pattern, path):
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
        return result