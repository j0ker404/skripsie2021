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

            w: ndarray (d+1,)

            return q_hat: shape(1,) 
        '''
        q_hat = None
        x = self.get_feature_vector(state=state, action=action)
        w = w.reshape((-1,1))
        # print('w \n{}'.format(w))
        # print('x \n{}'.format(x))
        q_hat = (w.T.dot(x)).reshape((1,))
        return q_hat

    def grad_q(self,state,action,w):
        '''
            Return the gradient of the q function

            return grad q: shape(d+1, 1)
        '''
        return self.get_feature_vector(state=state, action=action)

    def get_feature_vector(self, state, action):
        '''
            from state-action formulate x(s,a)

            state: list

            action: list

            return ndarray (d+1, 1)
        '''
        
        feature_vector = np.array(state).reshape((-1,1))
        # add bias term
        feature_vector = np.vstack((feature_vector, np.array([1])))
        # print('feature vector = \n{}'.format(feature_vector))
        return feature_vector