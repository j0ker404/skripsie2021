from alphaslime.store.config import Config
from alphaslime.store.constantConfig import Constants
from alphaslime.trainer.datahelp.filename import FileName

class Trainer:
    """ Base class for training an agent in
        an enviroment
    """
    def __init__(self, CONSTANTS:Constants) -> None:
        self.CONSTANTS = CONSTANTS

    def train(self, training_config:Config, agent_config:Config, fileNamer:FileName, prefix=''):
        """Train agent with the given hyperparameters

        Args:
            hyperparams (dict): hyperparamters for training
        """
        pass

    def save_data(self):
        """Save current training data
        """
        pass