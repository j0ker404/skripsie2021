from alphaslime.epsilon.epsilon import Epsilon


class LinearDecay(Epsilon):
    """Linear Epsilon decay 

        Implemenation of a linear epsilon decay
    """
    def __init__(self, min_eps, max_episode) -> None:
        super().__init__()
        self.min_eps  = min_eps
        self.max_episode = max_episode

    def get_epsilon(self, episode):
        slope = (self.min_eps - 1.0) / self.max_episode
        ret_eps = max(slope * episode + 1.0, self.min_eps)
        return ret_eps    