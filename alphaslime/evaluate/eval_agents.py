'''
    Evaluate agents

    Agent1: Right character
    Agent2: Left character



    Both state and pixel observations are presented assuming the agent is playing on the right side of the screen.


'''
from collections import deque
import gym
import numpy as np
from torch.functional import Tensor
from alphaslime.agents.agent import Agent
import time
from tqdm import tqdm

class EvaluateGameMA:
    '''
        Multi-agent evalutaion

        Class that evaluates the performance of the 
        trained agents for the l gym environment

        #TODO: add functionality for human controlled agent
    '''
    def __init__(self, agent_right:Agent, agent_left:Agent, base_dir_path, env, render=False, time_delay=0.03) -> None:
        self.agent_right = agent_right
        self.agent_left = agent_left
        self.RENDER = render
        # base directory to save data
        self.base_dir_path = base_dir_path
        self.env = env
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
    def __init__(self, agent:Agent, env, base_dir_path, render=False, time_delay=0.03) -> None:
        """Contructor

        Args:
            agent (Agent): Initilised Agent to interact with environment
            env (Gym.env): Evironment that agent interacts with
            base_dir_path (str): Path to save data
            render (bool, optional): Determine if agent-environment interaction must be drawn to screen. Defaults to False.
            time_delay (float, optional): Time delay between each frame of Rendered interaction. Defaults to 0.03.
        """
        self.agent = agent
        self.RENDER = render
        # base directory to save data
        self.base_dir_path = base_dir_path
        self.env = env
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

        done = False
        total_reward = 0

        # time step counter
        t = 0
        # start episode
        while not done:

            action_index = self.agent.get_action(obs1)
            if type(action_index) == Tensor:
                action = self.agent.action_table[action_index.item()]
            else:
                action = self.agent.action_table[action_index] 
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
        return total_reward

    def evaluate(self, EPISODES, is_progress_bar=False, running_avg_len=100):
        """Evaluate agent performance for given number of episodes

        Args:
            EPISODES (int): Number of Episodes to run
            is_progress_bar (bool): Display progress bar. Default to False
        
        return:
            episodes_reward: (list), total reward per episode
            avg_rewards_array: (list), running reward average per episode 
        """
        rewards = []
        rewards_deque = deque(maxlen=running_avg_len)
        avg_rewards_array = [] 
        ranger = range(EPISODES)
        if is_progress_bar:
            ranger = tqdm(ranger)
        
        # evaluate episodes
        for episode in ranger:
            episode_reward = self.evaluate_episode()
            rewards_deque.append(episode_reward)
            avg_score = np.mean(rewards_deque)
            # append average reward at current epsiode
            avg_rewards_array.append(avg_score)
            # append total episode reward
            rewards.append(episode_reward)

        self.env.close()

        return rewards, avg_rewards_array



