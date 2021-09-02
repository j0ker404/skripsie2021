from alphaslime.approx.q import QApprox
import numpy as np

from alphaslime.envgame.slenv import SLenv
from alphaslime.agents.greedyAgent import GreedyAgent
from alphaslime.approx.linearq import LinearQApprox



class SarsaSP(GreedyAgent):
    '''
        Semi gradient Sarsa agent trained with self-play
    '''


    def __init__(self, epsilon, q_hat: QApprox = LinearQApprox(), alpha=1/10, gamma=0.9, d=12, weights=None, env_id="SlimeVolley-v0") -> None:
        super().__init__(epsilon, q_hat, d, weights=weights)

        # original values
        self.INIT_EPSILON = self.epsilon

        # alg values
        self.alpha = alpha
        self.gamma = gamma

        # Threshold: avg_score that agent must beat 
        # current opponent to become new champion
        # arbitary values chosen
        self.THRESHOLD = 0.01

        # environment variables
        self.env_id = env_id

        # epsilon decay variables
        self.EPSILON_DECAY_STATE = True
        self.EPSILON_DECAY_BASE = 0.997
        


    
    def train(self, episodes, num_champions_train, champions_dir=None):
        '''
            Train agent

            episodes: number of episiodes to use per training of
            champion

            champions_train: number of champion trails

            TODO:   - save champions to disk
                    - log data
        '''

        if champions_dir is None:
            # champions list
            champions = []
        else:
            # TODO: load champions from data
            pass

        # load random champion for intial purpuse
        champion = GreedyAgent(self.epsilon, self.q_hat, self.d, weights=self.w)
        champions.append(champion)

        # configure initial gym environment
        self.env = SLenv(opponent=champion, env_id=self.env_id)
        # time step tracker per episode
        t = 0
        # episode reward tracker
        episode_reward_value = 0
        max_val_array = np.zeros((2,))

        avg_score_episode = 0

        for i in range(num_champions_train):

            # reset episode avgerage score
            avg_score_episode = avg_score_episode/episodes

            # reset epsilon value
            self.epsilon = self.INIT_EPSILON

            print('Running champion trail = {} ...'.format(i))

            # train new agent
            for episode in range(episodes):

                # update champions list
                if avg_score_episode > self.THRESHOLD:
                    champion = GreedyAgent(self.epsilon, self.q_hat, self.d, weights=self.w)
                    champions.append(champion)
                    avg_score_episode = 0

                # use latest champion for training new agent
                champion = champions[-1]
                # change opponent in environment
                self.env.opponent = champion
                
                
                

                # reset environment
                obs = self.env.reset()
                action = self.get_action(obs)

                # decay epsilon value
                if self.EPSILON_DECAY_STATE:
                    self.epsilon = np.power(self.EPSILON_DECAY_BASE, episode)

                # ---------------------------------------------------
                # reset episode time and reward values
                # use for logging data to disk
                t = 0
                episode_reward_value = 0
                # ---------------------------------------------------


                # step through episode
                while t < self.T_MAX:
                    # take action, go to next time step
                    obs_next, reward, done, info = self.env.step(action)
                    t += 1 # increment time step
                    episode_reward_value += reward # increment episode reward value
                    avg_score_episode += reward

                    action_index = self.action_table.index(action)
                    # if at terminal state
                    if done:
                        q_hat = self.q_hat.state_action_value(
                            state=obs, action=action, w=self.w[:, action_index])
                        grad_q_hat = self.q_hat.grad_q(
                            state=obs, action=action, w=self.w[:, action_index])
                        beta = reward - q_hat

                        # update weights
                        self.w[:, action_index] = self.w[:, action_index] + \
                            self.alpha*beta*grad_q_hat.reshape((-1))

                        # normalize weights
                        # max_val =  self.w[:, action_index].max(axis=0)
                        # if np.abs(max_val) > 0:
                        #     self.w[:, action_index] = self.w[:, action_index] / max_val

                        max_val = np.abs(self.w[:, action_index].max(axis=0))
                        min_val = np.abs(self.w[:, action_index].min(axis=0))
                        max_val_array[0] = max_val
                        max_val_array[1] = min_val

                        abs_max = np.max(max_val_array)
                        if abs_max > 0:
                            self.w[:, action_index] = self.w[:, action_index] / abs_max  

                        # go to next episode
                        break

                    action_next = self.get_action(obs_next)

                    # ---------------------------------------------------
                    # update weight vector

                    action_next_index = self.action_table.index(action_next)

                    q_hat = self.q_hat.state_action_value(
                        state=obs, action=action, w=self.w[:, action_index])

                    grad_q_hat = self.q_hat.grad_q(
                        state=obs, action=action, w=self.w[:, action_index])

                    q_hat_next = self.q_hat.state_action_value(
                        state=obs_next, action=action_next, w=self.w[:, action_next_index])

                    # q_hat_next = self.q_hat.state_action_value(
                    #     state=obs_next, action=action_next, w=self.w[:, action_index])

                    delta = (reward + self.gamma*q_hat_next - q_hat)
                
                    self.w[:, action_index] = self.w[:, action_index] + \
                        self.alpha*delta*grad_q_hat.reshape((-1))

                    # normalize weight
                    max_val = np.abs(self.w[:, action_index].max(axis=0))
                    min_val = np.abs(self.w[:, action_index].min(axis=0))
                    max_val_array[0] = max_val
                    max_val_array[1] = min_val

                    abs_max = np.max(max_val_array)
                    if abs_max > 0:
                        self.w[:, action_index] = self.w[:, action_index] / abs_max 

                    # ---------------------------------------------------

                    # ---------------------------------------------------
                    # update state-obeservation and action-state
                    obs = obs_next
                    action = action_next
                    # ---------------------------------------------------


    
