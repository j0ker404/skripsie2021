'''
    Evaluate agents

    Agent1: Right character
    Agent2: Left character



    Both state and pixel observations are presented assuming the agent is playing on the right side of the screen.


'''
import gym
from alphaslime.agents.agent import Agent
import time

class EvaluateGameMA:
    '''
        Multi-agent evalutaion

        Class that evaluates the performance of the 
        trained agents for the l gym environment

        #TODO: add functionality for human controlled agent
    '''
    def __init__(self, agent_right:Agent, agent_left:Agent, base_dir_path, env_id="SlimeVolley-v0", render=False, time_delay=0.03) -> None:
        self.agent_right = agent_right
        self.agent_left = agent_left
        self.RENDER = render
        # base directory to save data
        self.base_dir_path = base_dir_path
        self.env = gym.make(env_id)
        self.delay = time_delay
    
    def evaluate_episode(self):
        '''
            Evaluate one episode

            Episode terminates when either agent loses all five lives, 
            or after 3000 timesteps has passed.
            
            #TODO: save data to a file
                - save state
                - save actions
                - save rewards
                - save time step

            
            return agent right score,
            one can infer agent left score
        '''
        obs1 = self.env.reset()
        obs2 = obs1 # both sides always see the same initial observation.

        done = False
        total_reward = 0

        # time step counter
        t = 0
        # start episode
        while not done:

            action_right = self.agent_right.get_action(obs1)
            action_left = self.agent_left.get_action(obs2)
            # print('action_right = {}'.format(action_right))
            # print('action_left = {}'.format(action_left))

            # print('t = {}'.format(t))
            # print('obs1 = {}\n'.format(obs1))

            # go to next time step
            obs1, reward, done, info = self.env.step(action_right, action_left) # extra argument
            obs2 = info['otherObs'] #  opponent's observations

            total_reward += reward

            # increment time step
            t += 1
            if self.RENDER:
                # render game to screen
                self.env.render()
                # sleep
                time.sleep(self.delay)

        # print("agent right's score:", total_reward)
        # print("agent left's score:", -total_reward)

        # return score

        return total_reward


class EvaluateGameSA:
    '''
        Single-agent evalutaion

        Class that evaluates the performance of the 
        trained agents for gym environment

        #TODO: add functionality for human controlled agent
    '''
    def __init__(self, agent:Agent,  base_dir_path, env_id="SlimeVolley-v0", render=False, time_delay=0.03) -> None:
        self.agent = agent
        self.RENDER = render
        # base directory to save data
        self.base_dir_path = base_dir_path
        self.env = gym.make(env_id)
        self.delay = time_delay
    
    def evaluate_episode(self):
        '''
            Evaluate one episode

            Episode terminates when either agent loses all five lives, 
            or after 3000 timesteps has passed.
            
            #TODO: save data to a file
                - save state
                - save actions
                - save rewards
                - save time step

            
            return agent right score,
            one can infer agent left score
        '''
        obs1 = self.env.reset()
        obs2 = obs1 # both sides always see the same initial observation.

        done = False
        total_reward = 0

        # time step counter
        t = 0
        # start episode
        while not done:

            action = self.agent.get_action(obs1)

            # go to next time step
            obs1, reward, done, info = self.env.step(action) # extra argument

            total_reward += reward

            # increment time step
            t += 1
            if self.RENDER:
                # render game to screen
                self.env.render()
                # sleep
                time.sleep(self.delay)

        # print("agent right's score:", total_reward)
        # print("agent left's score:", -total_reward)

        # return score

        return total_reward



