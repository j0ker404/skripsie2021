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

    def __init__(self, alpha=1/10, epsilon=0.1, gamma=0.9, d=12, env_id="SlimeVolley-v0", opponent=BaselineAgent()) -> None:
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
        self.w = np.zeros((self.d+1, self.max_actions))
        # trained data
        self.w[0, :] = np.array([-0.08096563793105337,0.5932391754317804,-1.2201428723125307,-1.921199260255042,-3.154315931320239,-0.18057306111337598])
        self.w[1, :] = np.array([0.06788054329721156,0.7913320444289345,-0.4882389816252477,-0.8581545046760796,-0.9085363557309376,-0.059579768685018304])
        self.w[2, :] = np.array([-0.5360218106590677,-0.4529623360630911,1.0,-3.565721017635909,-4.509855045998199,1.0])
        self.w[3, :] = np.array([1.0,-1.3552734650414175,-0.09572247119357247,-8.744323996614133,-2.17576574649541,-0.3978010046096702])
        self.w[4, :] = np.array([-0.01604645754233,0.298407718564675,-0.4191551764356536,0.889391063135986,0.5425023891514924,0.19494554429735902])
        self.w[5, :] = np.array([-0.12617165506007874,-0.18616986863116858,-0.8319429692408365,-1.054002303169712,-3.517995195423102,-0.021840176628467968])
        self.w[6, :] = np.array([0.08706785018159437,-0.2520340503362638,-1.8134939666754204,1.0,-1.680991432568234,-0.35517836524660745])
        self.w[7, :] = np.array([-0.7292724929709715,-1.3173727638491806,-1.4317499704230099,-5.357873028890994,-2.8670131092535036,-0.24277372840984815])
        self.w[8, :] = np.array([-0.14257044537691826,-0.16513961220972304,-0.08374093959397749,0.1554394015975789,-1.1930254588270894,-0.164704737968246])
        self.w[9, :] = np.array([-0.12486914197209592,0.27108919711745516,-0.3184515499359992,-0.7896130882540389,-1.0013746835938981,0.040370071150576406])
        self.w[10, :] = np.array([-0.05960914069619926,0.482711265013648,-0.8781867587617771,-0.962817595736845,0.18077630651358667,0.3252841224420629])
        self.w[11, :] = np.array([-0.017579313712547128,1.0,0.7213540360720776,-5.843265247583098,1.0,-0.8713264413246551])
        self.w[12, :] = np.array([-0.14251878526092623,0.3436441494792104,-0.9882959133499729,-0.6718638381576495,-2.655654606563139,-0.1485374564477367])


        # q function approximator
        self.q_hat = LinearQApprox()

        # minimum epsilon value
        self.MINIMUM_EPSILON = 0.1
        # after how many episodes will we decay epsilon
        self.DECAY_EPSILON_EPISODE_TARGET = 100
        # epsilon decay rate
        self.EPSILON_DECAY = 0.9

    def train(self, episodes):
        '''
            Train Agent


            episodes: total number of episodes to play


            #TODO: save training data
        '''
        
        # reset weights
        self.w = np.zeros((self.d+1, self.max_actions))
        self.epsilon = 1

        # time step tracker
        t = 0

        for episode in range(episodes):
            # reset environment
            obs = self.env.reset()
            action = self.get_action(obs)

            # print data to screen every 100 episode
            # if (episode) % 100 == 0:
            #     print('-'*99)
            #     print('self.w = {}'.format(
            #         self.w))
            #     print('-'*99)

            # ---------------------------------------------------
            # decay epsilon value
            if episode%self.DECAY_EPSILON_EPISODE_TARGET == 0 and self.epsilon > self.MINIMUM_EPSILON:
                # print('dsfdf')
                self.epsilon = self.epsilon*self.EPSILON_DECAY
                if self.epsilon < self.MINIMUM_EPSILON:
                    self.epsilon = self.MINIMUM_EPSILON     
            # ---------------------------------------------------

            while t < self.T_MAX:
                # take action, go to next time step
                obs_next, reward, done, info = self.env.step(action)
                t += 1

                action_index = self.action_table.index(action)
                # if at terminal state
                if done:
                    q_hat = self.q_hat.state_action_value(
                        state=obs, action=action, w=self.w[:, action_index])
                    grad_q_hat = self.q_hat.grad_q(
                        state=obs, action=action, w=self.w[:, action_index])
                    beta = reward * q_hat

                    # update weights
                    self.w[:, action_index] = self.w[:, action_index] + \
                        self.alpha*beta*grad_q_hat.reshape((-1))

                    # normalize weights
                    # self.w = self.w / self.w.max(axis=0)

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

                delta = (reward + self.gamma*q_hat_next - q_hat)
                # print('-'*99)
                # print('self.w[:,action_index] = {}'.format(self.w[:,action_index]))
                # print('grad_q_hat = {}'.format(grad_q_hat))
                self.w[:, action_index] = self.w[:, action_index] + \
                    self.alpha*delta*grad_q_hat.reshape((-1))

                # normalize weight
                max_val = self.w[:, action_index].max(axis=0)
                if max_val > 0:
                    self.w[:, action_index] = self.w[:, action_index] / max_val  

                # print('self.w[:,action_index] = {}'.format(self.w[:,action_index]))
                # print('-'*99)
                # ---------------------------------------------------

                # ---------------------------------------------------
                # update state-obeservation and action-state
                obs = obs_next
                action = action_next
                # ---------------------------------------------------

        # print('-'*99)
        # print('final training data')
        # print('self.w = {}'.format(
        #     self.w))

        # print('epsilon = {}'.format(self.epsilon))
        # print('-'*99)



    def get_action(self, state):
        '''
            Get next action given current state
            for an  agent

            state: current observation

            TODO: implement epsilon-greedy  
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
