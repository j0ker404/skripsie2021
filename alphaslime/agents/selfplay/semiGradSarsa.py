import numpy as np
from alphaslime.agents.baseline import BaselineAgent

from alphaslime.agents.agent import Agent
from alphaslime.envgame.slenv import SLenv

from alphaslime.approx.linearq import LinearQApprox

class SemiGradSarsa(Agent):
    '''
        Implement Episodic Semi-gradient Sarsa for Estimating
        state-action value function (q)
    '''

    def __init__(self, alpha, epsilon, gamma=0.9, d=12, env_id="SlimeVolley-v0", opponent=BaselineAgent()) -> None:
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        # environment
        self.env = SLenv(opponent=opponent, env_id=env_id)

        # weight vector
        self.w = np.zeros((d, 1))

        # q function approximator
        self.q_hat = LinearQApprox()


    def train(self, episodes):
        '''
            Train Agent


            episodes: total number of episodes to play



            #TODO: save training data
        '''

        # time step tracker
        t = 0

        for episode in range(episodes):
            # reset environment
            obs = self.env.reset()
            action = self.get_action(obs)
            
            while t < self.T_MAX:
                # take action, go to next time step
                obs_next, reward, done, info = self.env.step(action)
                t += 1
                
                # if at terminal state
                if done:
                    q_hat = self.q_hat.state_action_value(state=obs, action=action, w=self.w)
                    grad_q_hat = self.q_hat.grad_q(state=obs, action=action, w=self.w)
                    beta = reward *  q_hat

                    # update weights
                    self.w = self.w + self.alpha*beta*grad_q_hat
                    
                    # go to next episode
                    break
                
                action_next = self.get_action(obs_next)

                # ---------------------------------------------------
                # update weight vector

                q_hat = self.q_hat.state_action_value(state=obs, action=action, w=self.w)
                grad_q_hat = self.q_hat.grad_q(state=obs, action=action, w=self.w)
                q_hat_next = self.q_hat.state_action_value(state=obs_next, action=action_next, w=self.w)

                delta = (reward + self.gamma*q_hat_next - q_hat)
                self.w = self.w + self.alpha*delta*grad_q_hat
                # ---------------------------------------------------

                obs = obs_next
                action = action_next


    def get_action(self, state):
        '''
            Get next action give current state

            state: current observation

            TODO: implement epsilon-greedy  
        '''
        action = None

        return action