import numpy as np
from alphaslime.epsilon.epsilon import Epsilon


class ExponentialDecay(Epsilon):
    '''
        Exponential decay epsilon
    '''
    def __init__(self, min_epsilon, max_epsilon, decay_rate) -> None:
        super().__init__()
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.decay_rate = decay_rate

    def get_epsilon(self, episode):
        """Return exponetial decayed epsilon value

        Args:
            episode (int): current episode iteration

        Returns:
            float: decayed epsilon value
        """
        epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-self.decay_rate*episode)   
        return epsilon