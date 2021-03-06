from alphaslime.agents.agent import Agent
import slimevolleygym

class BaselineAgent(Agent):
    '''
        Implement baseline agent

        for now use the baseline provided by the 
        Slimeball gym environment
    '''

    def __init__(self, config:dict) -> None:
        super().__init__(config)
        self.policy = slimevolleygym.BaselinePolicy() 

    def get_action(self, state):
        """Return index of action that matches the 

        Args:
            state (list): state observed

        Returns:
            action_index: index of action that matches action_table
        """
        # action =  super().get_action(state)
        self.actions  = self.policy.predict(state)
        try:
            action_index = self.action_table.index(self.actions)
        except ValueError:
            # if no valid action, then action must be
            # a no move i.o.w action 0
            action_index = 0
        return action_index


