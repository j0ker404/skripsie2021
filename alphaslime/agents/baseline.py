from alphaslime.agents.agent import Agent
import slimevolleygym

class BaselineAgent(Agent):
    '''
        Implement baseline agent

        for now use the baseline provided by the 
        Slimeball gym environment
    '''

    def __init__(self) -> None:
        super().__init__()
        self.policy = slimevolleygym.BaselinePolicy() 

    def get_action(self, state):
        # action =  super().get_action(state)
        self.actions  = self.policy.predict(state)

        return self.actions


