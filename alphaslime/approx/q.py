class QApprox:
    '''
        Base class for implementing
        a q function approximator
    '''
    def __init__(self) -> None:
        pass

    def state_action_value(self,state,action,w):
        pass

    def grad_q(self,state,action,w):
        pass