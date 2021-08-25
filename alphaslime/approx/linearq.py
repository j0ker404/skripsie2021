import numpy as np
from q import QApprox

class LinearQApprox(QApprox):

    def __init__(self) -> None:
        super().__init__()

    def state_action_value(self,state,action,w:np.ndarray):
        '''
            Return the approximate state-action value
            using linear combination
        '''
        q_hat = None
        x = self.get_feature_vector(state=state, action=action)
        q_hat = w.dot(x)
        return q_hat

    def grad_q(self,state,action,w):
        return self.get_feature_vector(state=state, action=action)

    def get_feature_vector(self, state, action):
        '''
            from state-action formulate x(s,a)
        '''
        pass