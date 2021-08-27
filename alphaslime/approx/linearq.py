import numpy as np
from alphaslime.approx.q import QApprox

class LinearQApprox(QApprox):
    '''
        Linear q-function approximator
        using linear combinations with a weight vector 
        and feature vector x(s,a)
    '''
    def __init__(self) -> None:
        super().__init__()

    def state_action_value(self,state,action,w:np.ndarray):
        '''
            Return the approximate state-action value
            using linear combination

            q_hat = w.T dot x(s,a)

            state: list

            action: list

            w: ndarray (d, 1)

            return q_hat: shape(1,) 
        '''
        q_hat = None
        x = self.get_feature_vector(state=state, action=action)
        q_hat = w.dot(x)
        return q_hat

    def grad_q(self,state,action,w):
        '''
            Return the gradient of the q function

            return grad q: shape(d, 1)
        '''
        return self.get_feature_vector(state=state, action=action)

    def get_feature_vector(self, state, action):
        '''
            from state-action formulate x(s,a)

            state: list

            action: list

            return ndarray (d, 1)
        '''
        

        return np.array(state).reshape((-1,1))