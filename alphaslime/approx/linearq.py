import numpy as np
from alphaslime.approx.q import QApprox
from alphaslime.approx.tile import IHT
import alphaslime.approx.tile as tile

class LinearQApprox(QApprox):
    '''
        Linear q-function approximator
        using linear combinations with a weight vector 
        and feature vector x(s,a)
    '''
    def __init__(self, action_table=None) -> None:
        super().__init__()
        self.action_table = action_table
        self.MAX_ACTIONS = len(self.action_table)

        # tiling data
        self.FEATURE_VECTOR_LENGTH = 4096
        self.iht=IHT(self.FEATURE_VECTOR_LENGTH)

    
    def _get_tile_indices(self, state):
        '''
            convert state space to feature vector x(s,a) with tile coding
        '''
        return tile.tiles(self.iht,8,[8*state[0]/(0.5+1.2),8*state[1]/(0.07+0.07)])

    def state_action_value(self,state,action,w:np.ndarray):
        '''
            Return the approximate state-action value
            using linear combination

            q_hat = w.T dot x(s,a)

            state: list

            action: list

            w: ndarray (n_max_actions*d+1,1)

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

            return ndarray (n_max_actions*d+1, 1)
        '''
        # https://danieltakeshi.github.io/2016/10/31/going-deeper-into-reinforcement-learning-understanding-q-learning-and-linear-function-approximation/
        # '''
        action_index = self.action_table.index(action)
        d = len(state)
        # print('d = {}'.format(d))
        feature_vector = np.zeros((d*self.MAX_ACTIONS, ))
        base_index = d*action_index
        for i, feature in enumerate(state):
            feature_vector[base_index + i] = feature 
        
        feature_vector = feature_vector.reshape((-1,1))
        # feature_vector = np.array(state).reshape((-1,1))
        # add bias term
        feature_vector = np.vstack((feature_vector, np.array([1])))
        # print('feature vector = \n{}'.format(feature_vector))
        
        # '''
        '''
            # so we are making a feature vector from tile coding

            action_index = self.action_table.index(action)
            # create empty obs feature vector
            feature_vector = np.zeros((self.FEATURE_VECTOR_LENGTH*self.MAX_ACTIONS, ))
            obs_tile_indices = self._get_tile_indices(state)
            obs_tiles = np.zeros((self.FEATURE_VECTOR_LENGTH,))
            obs_tiles[obs_tile_indices] = 1
            base_index = self.FEATURE_VECTOR_LENGTH*action_index
            indices = np.arange(base_index, base_index+self.FEATURE_VECTOR_LENGTH,step=1)
            # set elements of feature vectors
            feature_vector[indices] = obs_tiles

            feature_vector = feature_vector.reshape(-1,1)
            feature_vector = np.vstack((feature_vector, np.array([1])))
        
        '''
        return feature_vector