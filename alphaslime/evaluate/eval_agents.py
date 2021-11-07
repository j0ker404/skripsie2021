'''
    Evaluate agents

    Agent1: Right character
    Agent2: Left character



    Both state and pixel observations are presented assuming the agent is playing on the right side of the screen.


'''
from torch.functional import Tensor
from alphaslime.agents.agent import Agent
import time


from alphaslime.evaluate.eval import Evaluate

# global variables for human control
manualAction = [0, 0, 0] # forward, backward, jump
otherManualAction = [0, 0, 0]
manualMode = False
otherManualMode = False

class EvaluateGameMA(Evaluate):
    '''
        Multi-agent evalutaion

        Class that evaluates the performance of the 
        trained agents for the l gym environment

        #TODO: add functionality for human controlled agent
    '''
    def __init__(self, agent_right:Agent, agent_left:Agent, env, base_dir_path, render=False, time_delay=0.03) -> None:
        super().__init__(env, base_dir_path, render=render, time_delay=time_delay)
        self.agent_right = agent_right
        self.agent_left = agent_left

    
    def evaluate_episode(self, idx, save=True):
        '''
            Evaluate one episode

            Episode terminates when either agent loses all five lives, 
            or after 3000 timesteps has passed.
            
            args:
                idx (int): Current episode index

            #TODO: save data to a file
                - save state
                - save actions
                - save rewards
                - save time step

            
            return agent right score,
            one can infer agent left score
        '''

        if self.RENDER:
            from pyglet.window import key
            from time import sleep

        global manualMode, manualAction, otherManualMode, otherManualAction
        # taken from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
        def key_press(k, mod):
            global manualMode, manualAction, otherManualMode, otherManualAction
            if k == key.LEFT:  manualAction[0] = 1
            if k == key.RIGHT: manualAction[1] = 1
            if k == key.UP:    manualAction[2] = 1
            if (k == key.LEFT or k == key.RIGHT or k == key.UP): manualMode = True

            if k == key.D:     otherManualAction[0] = 1
            if k == key.A:     otherManualAction[1] = 1
            if k == key.W:     otherManualAction[2] = 1
            if (k == key.D or k == key.A or k == key.W): otherManualMode = True

        def key_release(k, mod):
            global manualMode, manualAction, otherManualMode, otherManualAction
            if k == key.LEFT:  manualAction[0] = 0
            if k == key.RIGHT: manualAction[1] = 0
            if k == key.UP:    manualAction[2] = 0
            if k == key.D:     otherManualAction[0] = 0
            if k == key.A:     otherManualAction[1] = 0
            if k == key.W:     otherManualAction[2] = 0

        if self.RENDER:
            self.env.render()
            self.env.viewer.window.on_key_press = key_press
            self.env.viewer.window.on_key_release = key_release

        obs1 = self.env.reset()
        obs2 = obs1 # both sides always see the same initial observation.

        done = False
        total_reward = 0

        # time step counter
        t = 0
        # start episode
        while not done:

            action_data_right = self.agent_right.get_action(obs1)
            action_data_left = self.agent_left.get_action(obs2)
            # print('action_right = {}'.format(action_right))
            # print('action_left = {}'.format(action_left))

            # print('t = {}'.format(t))
            # print('obs1 = {}\n'.format(obs1))

            try:
                # Assume that action data is a
                # type of list
                # assume that action_index is the 
                # first element
                action_index_right, *other_action_data_right = action_data_right
            except:
                # action data is a single element
                # thus action data is the action index
                action_index_right = action_data_right

            if type(action_index_right) == Tensor:
                action_right = self.agent_right.action_table[action_index_right.item()]
            else:
                action_right = self.agent_right.action_table[action_index_right] 
            try:
                # Assume that action data is a
                # type of list
                # assume that action_index is the 
                # first element
                action_index_left, *other_action_data_left = action_data_left
            except:
                # action data is a single element
                # thus action data is the action index
                action_index_left = action_data_left

            if type(action_index_left) == Tensor:
                action_left = self.agent_left.action_table[action_index_left.item()]
            else:
                action_left = self.agent_left.action_table[action_index_left] 

            # human action
            if manualMode: # override with keyboard
                action_right = manualAction

            if otherManualMode:
                action_left = otherManualAction

            # go to next time step
            obs1, reward, done, info = self.env.step(action_right, action_left) # extra argument
            obs2 = info['otherObs'] #  opponent's observations

            total_reward += reward
            if reward > 0 or reward < 0:
                manualMode = False
                otherManualMode = False

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


class EvaluateGameSA(Evaluate):
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
        super().__init__(env, base_dir_path, render=render, time_delay=time_delay)
        self.agent = agent

    
    def evaluate_episode(self, idx, save=True):
        '''
            Evaluate one episode

            Episode terminates when either agent loses all five lives, 
            or after 3000 timesteps has passed.
        
            args:
                idx (int): episode count
            
            return agent right score,
            one can infer agent left score
        '''

        if self.RENDER:
            from pyglet.window import key
            from time import sleep

        # taken from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
        global manualMode, manualAction
        def key_press(k, mod):
            global manualMode, manualAction
            if k == key.LEFT:  manualAction[0] = 1
            if k == key.RIGHT: manualAction[1] = 1
            if k == key.UP:    manualAction[2] = 1
            if (k == key.LEFT or k == key.RIGHT or k == key.UP): manualMode = True

        def key_release(k, mod):
            global manualMode, manualAction
            if k == key.LEFT:  manualAction[0] = 0
            if k == key.RIGHT: manualAction[1] = 0
            if k == key.UP:    manualAction[2] = 0


        if self.RENDER:
            self.env.render()
            self.env.viewer.window.on_key_press = key_press
            self.env.viewer.window.on_key_release = key_release

        obs1 = self.env.reset()

        done = False
        total_reward = 0
        episode_data = []
        # time step counter
        t = 0
        # start episode
        while not done:
            state_t = obs1
            action_data = self.agent.get_action(obs1)
            try:
                # Assume that action data is a
                # type of list
                # assume that action_index is the 
                # first element
                action_index, *other_action_data = action_data
            except:
                # action data is a single element
                # thus action data is the action index
                action_index = action_data

            if type(action_index) == Tensor:
                action = self.agent.action_table[action_index.item()]
            else:
                action = self.agent.action_table[action_index] 

            # human action
            if manualMode: # override with keyboard
                action = manualAction

            # go to next time step
            obs1, reward, done, info = self.env.step(action) # extra argument

            total_reward += reward
            if reward > 0 or reward < 0:
                manualMode = False


            # increment time step
            t += 1
            if self.RENDER:
                # render game to screen
                self.env.render()
                # sleep
                time.sleep(self.delay)
            episode_time_step_data = [state_t, action_index, reward]
            episode_data.append(episode_time_step_data)

        # save episode data
        if save:
            self.save_episode(idx, episode_data)
        return total_reward