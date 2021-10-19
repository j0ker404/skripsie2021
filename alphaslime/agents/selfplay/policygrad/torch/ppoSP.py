from collections import deque
import numpy as np
from tqdm import tqdm
from alphaslime.agents.RL.policygrad.torch.ppo import PPOAgent
from alphaslime.envgame.spenv import SPenv
from alphaslime.store.config import Config
from alphaslime.trainer.selfplay.champion import Champions

import copy

from alphaslime.trainer.selfplay.ppoChamps import PPOChampions

class PPO_SP(PPOAgent):
    """PPO agent that implements training
        method via self play

    Args:
        PPOAgent (PPOAgent): Base class
    """

    def __init__(self, CONSTANTS: Config, config: Config) -> None:
        super().__init__(CONSTANTS, config)
        self.env_orginal = copy.deepcopy(self.env)
        self.CONSTANTS = CONSTANTS
        self.config = config

    def train_selfplay(self, train_config: Config):
        # load configs
        EPISODES =  train_config.get('EPISODES')
        is_progress =  train_config.get('is_progress')
        threshold =  train_config.get('threshold')
        is_threshold_stop =  train_config.get('is_threshold_stop')
        running_avg_len =  train_config.get('running_avg_len')
        agent_class = train_config.get('agent_type')

        ranger = range(EPISODES)
        if is_progress:
            ranger = tqdm(ranger)

        rewards_deque = deque(maxlen=running_avg_len)
        
        self.learn_iters = 0
        # self.avg_reward = 0
        self.n_steps = 0
        is_solved = False
        # load champ list
        champ_dir = train_config.get('champ_dir')
        champions = PPOChampions(champ_dir)
        champ_threshold = train_config.get('champ_threshold')
        champ_min_avg_reward = train_config.get('champ_min_avg_rew')
        champ_prev_avg_score = champ_min_avg_reward
        for episode in ranger:
            # load new opponent
            opponent_path = champions.sample()
            if opponent_path is None:
                opponent = None
            else:
                opponent = agent_class(self.CONSTANTS, self.config)
                opponent.load_model(opponent_path)

            # add new opponent to environment
            self.env = SPenv(env=self.env_orginal, opponent=opponent)
            # train an episode
            t, score = self.episode_train()

            # append episode rewards to training data
            self.rewards.append(score)
            
            # determine running mean
            rewards_deque.append(score)
            avg_reward = np.mean(rewards_deque)
            # append mean to training data 
            self.avg_rewards.append(avg_reward)
            
            # check point agent
            if avg_reward > self.best_score:
                self.best_score = avg_reward
                path = self.MODEL_CHECKPOINT_PATH + 'avg_rew_' + str(self.best_score)
                self.save_model(path)

            # threshold against prevoius agent
            if avg_reward-champ_prev_avg_score > champ_threshold:
                champ_prev_avg_score = avg_reward

            if len(rewards_deque) == rewards_deque.maxlen:
                # determine solved environment
                if avg_reward >= threshold:
                    if not is_solved: 
                        print('\n Environment solved in {:d} episodes!\tAverage Score: {:.2f}'. \
                            format(episode, avg_reward))
                        is_solved = not is_solved
                    if is_threshold_stop:
                        break
            if episode % 10 == 0 and self.verbose:
                print('episode', episode, 'score %.1f' % score, 'avg score %.1f' % avg_reward,
                        'time_steps', self.n_steps, 'learning_steps', self.learn_iters)
