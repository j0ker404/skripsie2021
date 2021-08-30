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

    def __init__(self, alpha=1/10, epsilon=0.1, gamma=0.9, d=12, env_id="SlimeVolley-v0", opponent=BaselineAgent(), weights=None) -> None:
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.d = d

        # environment
        self.env = SLenv(opponent=opponent, env_id=env_id)

        # weight vectors
        '''
            We have 6 weight vectors corresponds to each availabe move

            each coloumn corresponds to a weight vector
        '''
        if weights is None:
            self.w = np.zeros((self.d+1, self.max_actions))
        else:
            # trained data
            self.w = weights
            # self.w[0, :] = np.array([-0.43381092296440843,0.14643595992553798,-0.7503064937995332,-1.771893129468569,-2.441086658877392,0.5423854146843167])
            # self.w[1, :] = np.array([-0.18142370284631687,-0.46506407491986257,-0.1535308285633933,-1.1199714758032322,-0.7422237628889041,0.493490999432343])
            # self.w[2, :] = np.array([-0.6064144971066551,0.8477886290641731,0.7604241022272246,0.4156987946593265,0.34850646352823056,-0.5557242498693779])
            # self.w[3, :] = np.array([-0.7072932288057565,-0.6318120125164647,1.0,-0.4048581698796228,0.01888388098937407,-0.9578669209596591])
            # self.w[4, :] = np.array([0.010342215037551275,1.0,0.7757298575787244,-1.565178944758415,-1.5447566291306032,1.0])
            # self.w[5, :] = np.array([-0.40062686098537786,-0.661341296641912,0.5210636678797306,-2.9450752173285757,-2.0045244776855786,-0.15005546961103222])
            # self.w[6, :] = np.array([-0.49878639170857053,0.5488407784187483,0.22836609381175033,-0.06482834533394519,-2.137854653348676,-0.8237478411195932])
            # self.w[7, :] = np.array([-0.001867011661956708,0.5733635662089257,-0.45079189222857863,-1.6534268871448607,1.0,-0.4144349902490697])
            # self.w[8, :] = np.array([-0.011511417957906767,-0.6172188005214232,0.24447182755350455,-0.3317708781314478,-0.6443588153600264,-1.249483089374404])
            # self.w[9, :] = np.array([-0.1442560610483229,-0.1559355164475284,0.4090758366696198,1.0,-0.8007519118349908,-0.16017739197149322])
            # self.w[10, :] = np.array([-0.5445310147948608,-0.15199609674460796,-0.9510043652849349,-1.4829634642289298,0.07742060317128256,-0.5244880268768782])
            # self.w[11, :] = np.array([1.0,-0.6890063377248142,0.23733286565378933,-0.29016568030965645,-0.2574847711411783,-0.7835321590820489])
            # self.w[12, :] = np.array([-0.49890928889848013,-0.7016640684774706,-0.06727119763119215,-0.47519760953144796,-0.9247822559125775,0.16526102661814832])

        # q function approximator
        self.q_hat = LinearQApprox()

        # minimum epsilon value
        self.MINIMUM_EPSILON = 0.0
        # after how many episodes will we decay epsilon
        self.DECAY_EPSILON_EPISODE_TARGET = 10000
        # epsilon decay rate
        self.EPSILON_DECAY = 0.9
        self.EPSILON_DECAY_STATE = True

        # values for epislonn decay with reward threshold
        self.reward_threshold = 0
        self.reward_threshold_increment = 1
    
    def reward_threshold_updater(self):
        '''
            Returns an updated reward threshold

            reward_threshold: current reward_threshold

            return new reward_threshold
        '''
        return self.reward_threshold+self.reward_threshold_increment

    def decay_epsilon(self):
        '''
            return decayed epsilon value.

            Exponetial decay in the form of 

            epsilon = epsilon - max score
        '''

        # foo =  self.epsilon * self.EPSILON_DECAY
        bar = self.epsilon - 1/self.MAX_SCORE

        return bar

    def train(self, episodes, reset_data=True):
        '''
            Train Agent


            episodes: total number of episodes to play


            #TODO: save training data
        '''
        
        if reset_data:
            # reset weights
            self.w = np.zeros((self.d+1, self.max_actions))
            # self.epsilon = 1
            self.reward_threshold = 0

        # time step tracker per episode
        t = 0
        # episode reward tracker
        episode_reward_value = 0
        for episode in range(episodes):
            # reset environment
            obs = self.env.reset()
            action = self.get_action(obs)


            # ---------------------------------------------------
            # decay epsilon value

            # exponential decay
            # if self.EPSILON_DECAY_STATE:
            #     if episode%self.DECAY_EPSILON_EPISODE_TARGET == 0 and self.epsilon > self.MINIMUM_EPSILON:
            #         self.epsilon = self.epsilon*self.EPSILON_DECAY
            #         if self.epsilon < self.MINIMUM_EPSILON:
            #             self.epsilon = self.MINIMUM_EPSILON   

            # decay with reward threshold
            if self.EPSILON_DECAY_STATE:
                if episode_reward_value >= self.reward_threshold:
                    self.epsilon = self.decay_epsilon()
                    self.reward_threshold = self.reward_threshold_updater()

            # ---------------------------------------------------


            # ---------------------------------------------------
            # reset episode time and reward values
            t = 0
            episode_reward_value = 0
            # ---------------------------------------------------

            if episode%1000 == 0:
                print('Completed Episodes = {}'.format(episode))
            while t < self.T_MAX:
                # take action, go to next time step
                obs_next, reward, done, info = self.env.step(action)
                t += 1 # increment time step
                episode_reward_value += reward # incremen episode reward value

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
                    max_val =  self.w[:, action_index].max(axis=0)
                    if np.abs(max_val) > 0:
                        self.w[:, action_index] = self.w[:, action_index] / max_val

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
                max_val = self.w[:, action_index].max(axis=0)
                if np.abs(max_val) > 0:
                    self.w[:, action_index] = self.w[:, action_index] / max_val  

                # ---------------------------------------------------

                # ---------------------------------------------------
                # update state-obeservation and action-state
                obs = obs_next
                action = action_next
                # ---------------------------------------------------

    def get_action(self, state):
        '''
            Get next action given current state
            for an  agent

            state: current observation

            epsilon-greedy implementation  
        '''
        action_index = None

        # epsilon greedy alg
        prob = np.random.uniform()
        if prob < self.epsilon:
            # random action
            action_index = np.random.randint(low=0, high=self.max_actions)
        else:
            q_values = [self.q_hat.state_action_value(
                state=state, action=self.action_table[action_index], w=self.w[:, action_index]) for action_index in range(self.max_actions)]
            q_values = np.array(q_values)
            action_index = np.argmax(q_values)
        return self.action_table[action_index]
