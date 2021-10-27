import os
import pickle
from torch.utils.data import Dataset

class EpisodeDataset(Dataset):
    """
        Episode Dataset loader

        A single episode consist of many time steps.

        Each time step consits of a [state, action, reward]
    """
    # def __init__(self, episode_labels, episode_dir, prefix='episode_', file_ext='.pkl', transform=None, target_transform=None):
    def __init__(self, episode_path, transform=None, target_transform=None):
        self.episode_path = episode_path
        self.transform = transform
        self.target_transform = target_transform
        with open(episode_path, 'rb') as f:
            self.episode = pickle.load(f)

    def __len__(self):
        return len(self.episode)

    def __getitem__(self, idx):
        """Episode consists of time steps that

            Episode ~ [state, action, reward]
        Args:
            idx (int): Time index of episode

        Returns:
            state_t, actions_t_index, reward_t: (State_t, Action_t, Reward_t) tuple
        """
        state_t, actions_t_index, reward_t = self.episode[idx]
        if self.transform:
            state_t = self.transform(state_t)
        if self.target_transform:
            actions_t_index = self.target_transform(actions_t_index)
            reward_t = self.target_transform(reward_t)
        return state_t, actions_t_index, reward_t


class EpisodesDataset(Dataset):
    """Dataset for storing episodes paths.


    Args:
        Dataset (Dataset): baseclass
    """

    def __init__(self, num_eps, episodes_dir, prefix='episode_', file_ext='.pkl', transform=None):
        self.num_eps = num_eps
        self.episodes_dir = episodes_dir
        self.prefix = prefix
        self.file_ext = file_ext

        self.transform = transform


    def __len__(self):
        return self.num_eps

    def __getitem__(self, idx):
        """Episode consists of time steps that

            Episode ~ [state, action, reward]
        Args:
            idx (int): Time index of episode

        Returns:
            path to episode (str): Path to episode trajectory data
        """
        eps_path = os.path.join(self.episodes_dir, self.prefix+str(idx)+self.file_ext)
        if self.transform:
            eps_path = self.transform(eps_path)
        return eps_path


import random
from torch.utils.data import DataLoader
if __name__ == '__main__':

    EPS = 6
    eps_dir = 'eps/'
    if not os.path.exists(eps_dir):
        os.makedirs(eps_dir)
    # print('Generating data')
    # for episode in range(EPS):
    #     episode_data = []
    #     T = random.randint(1, 10)
    #     for t in range(T):
    #         state = [random.random() for _ in range(12)]
    #         action = random.randint(0,5)
    #         reward = random.randint(-5,5)
    #         episode_time_step_data = [state, action, reward]
    #         episode_data.append(episode_time_step_data)
    #     eps_path = os.path.join(eps_dir, 'episode_'+str(episode)+'.pkl')
    #     with open(eps_path, 'wb') as f:
    #         pickle.dump(episode_data, f)

    # loading data
    # episode_labels = [i for i in range(EPS)]
    num_eps = 6
    episode_dir = eps_dir
    expert_episodes_data = EpisodesDataset(num_eps, episode_dir)
    expert_episodes_dataloader = DataLoader(expert_episodes_data, batch_size=2, shuffle=True)

    for batch, X in enumerate(expert_episodes_dataloader):
        print('-'*5)
        print('batch: {}'.format(batch))
        print('X = {}'.format(X))
        for episode_path in X:
            print('*'*5)
            episode_data = EpisodeDataset(episode_path)
            # iterate through each time step of episode
            for state_t, actions_t_index, reward_t in episode_data:
                print('@'*5)
                print('len(State_t) = {}'.format(state_t))
                print('actions_t_index = {}'.format(actions_t_index))
                print('reward_t = {}'.format(reward_t))
                print('@'*5)
            print('*'*5)
        print('-'*5)


    # prefix='episode_'
    # file_ext='.pkl'
    # for idx in range(EPS):
    #     episode_path = os.path.join(eps_dir, prefix+str(episode_labels[idx])+file_ext)
    #     episode = None
    #     with open(episode_path, 'rb') as f:
    #         episode = pickle.load(f)
    #     print(len(episode))
    #     for t in episode:
    #         print('-'*5)
    #         print(t)
    #         print('-'*5)
            
        # states, actions_index, rewards = episode

    # for batch, (X, y) in enumerate(train_dataloader):
    #     print('-'*5)
    #     # for item in X:
    #     #     print(item)
    #     states, actions_index, rewards = X
    #     # print(states)
    #     print(actions_index)
    #     print('-'*5)

