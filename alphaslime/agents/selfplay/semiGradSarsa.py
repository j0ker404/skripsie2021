from typing import AsyncIterator
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

        # weight vectors
        '''
            We have 6 weight vectors corresponds to each availabe move

            each coloumn corresponds to a weight vector
        '''
        self.w = np.zeros((d, self.max_actions))

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
                
                action_index = self.action_table.index(action)
                # if at terminal state
                if done:
                    print('done')
                    q_hat = self.q_hat.state_action_value(state=obs, action=action, w=self.w[:,action_index])
                    grad_q_hat = self.q_hat.grad_q(state=obs, action=action, w=self.w[:,action_index])
                    beta = reward *  q_hat

                    # update weights
                    self.w[:,action_index] = self.w[:,action_index] + self.alpha*beta*grad_q_hat.reshape((-1))
                    
                    # go to next episode
                    break
                
                action_next = self.get_action(obs_next)

                # ---------------------------------------------------
                # update weight vector

                action_next_index = self.action_table.index(action_next)

                q_hat = self.q_hat.state_action_value(state=obs, action=action, w=self.w[:,action_index])
                grad_q_hat = self.q_hat.grad_q(state=obs, action=action, w=self.w[:,action_index])
                q_hat_next = self.q_hat.state_action_value(state=obs_next, action=action_next, w=self.w[:,action_next_index])

                delta = (reward + self.gamma*q_hat_next - q_hat)
                print('-'*99)
                print('self.w[:,action_index] = {}'.format(self.w[:,action_index]))
                print('grad_q_hat = {}'.format(grad_q_hat))
                self.w[:,action_index] = self.w[:,action_index] + self.alpha*delta*grad_q_hat.reshape((-1))
                print('self.w[:,action_index] = {}'.format(self.w[:,action_index]))
                print('-'*99)
                # ---------------------------------------------------

                obs = obs_next
                action = action_next

    


    def get_action(self, state):
        '''
            Get next action given current state
            for an  agent

            state: current observation

            TODO: implement epsilon-greedy  
        '''
        action = [0,0,0]

        return action