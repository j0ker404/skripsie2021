from alphaslime.store.constantConfig import Constants

class Trainer:
    """ Base class for training an agent in
        an enviroment
    """
    def __init__(self, CONSTANTS:Constants) -> None:
        self.CONSTANTS = CONSTANTS

    def train(self, hyperparams:dict):
        """Train agent with the given hyperparameters

        Args:
            hyperparams (dict): hyperparamters for training
        """
        pass

    def save_data(self):
        """Save current training data
        """
        pass