import pickle

import numpy as np
import torch

from utils import convert_to_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset(torch.utils.data.Dataset):
    """Dataset class."""

    def __init__(self, config=None, data=None, path=None):
        print(config)
        self.shuffle = config['shuffle']
        self.horizon = config['horizon']
        self.store_gpu = config['store_gpu']
        self.config = config

        if data is not None:
            self._init_from_data(data)
        elif path is not None:
            self._init_from_path(path)
        else:
            raise ValueError("Either 'data' or 'path' must be provided.")

        self.zeros = np.zeros(
            config['state_dim'] ** 2 + 3 * config['action_dim'] + 1
        )
        self.zeros = convert_to_tensor(self.zeros, store_gpu=self.store_gpu)

    def _init_from_data(self, data):
        self.dataset = {
            'query_states': convert_to_tensor(data['query_states'], store_gpu=self.store_gpu),
            'optimal_actions': convert_to_tensor(data['optimal_actions'], store_gpu=self.store_gpu),
            'context_states': convert_to_tensor(data['context_states'], store_gpu=self.store_gpu),
            'context_actions': convert_to_tensor(data['context_actions'], store_gpu=self.store_gpu),
            'context_next_states': convert_to_tensor(data['context_next_states'], store_gpu=self.store_gpu),
            'context_rewards': convert_to_tensor(data['context_rewards'], store_gpu=self.store_gpu),
            'context': convert_to_tensor(data['context'], store_gpu=self.store_gpu),
        }

    def _init_from_path(self, path):
        # if path is not a list
        if not isinstance(path, list):
            path = [path]

        self.trajs = []
        for p in path:
            with open(p, 'rb') as f:
                self.trajs += pickle.load(f)
            
        context_states = []
        context_actions = []
        context_next_states = []
        context_rewards = []
        context = []
        query_states = []
        optimal_actions = []
        true_context = []

        for traj in self.trajs:

            context_states.append(traj['context_states'])
            context_actions.append(traj['context_actions'])
            context_next_states.append(traj['context_next_states'])
            context_rewards.append(traj['context_rewards'])
            context.append(traj['context'])

            query_states.append(traj['query_state'])
            optimal_actions.append(traj['optimal_action'])
            true_context.append(traj['true_context'])

        context_states = np.array(context_states)
        context_actions = np.array(context_actions)
        context_next_states = np.array(context_next_states)
        context_rewards = np.array(context_rewards)

        context = np.array(context)
        
        if len(context_rewards.shape) < 3:
            context_rewards = context_rewards[:, :, None]
        query_states = np.array(query_states)
        optimal_actions = np.array(optimal_actions)
        true_context = np.array(true_context)

        self.dataset = {
            'query_states': convert_to_tensor(query_states, store_gpu=self.store_gpu),
            'optimal_actions': convert_to_tensor(optimal_actions, store_gpu=self.store_gpu),
            'context_states': convert_to_tensor(context_states, store_gpu=self.store_gpu),
            'context_actions': convert_to_tensor(context_actions, store_gpu=self.store_gpu),
            'context_next_states': convert_to_tensor(context_next_states, store_gpu=self.store_gpu),
            'context_rewards': convert_to_tensor(context_rewards, store_gpu=self.store_gpu),
            'context': convert_to_tensor(context, store_gpu=self.store_gpu),
            'true_context': convert_to_tensor(true_context, store_gpu=self.store_gpu),
        }

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset['query_states'])

    def __getitem__(self, index):
        'Generates one sample of data'
        res = {
            'context_states': self.dataset['context_states'][index],
            'context_actions': self.dataset['context_actions'][index],
            'context_next_states': self.dataset['context_next_states'][index],
            'context_rewards': self.dataset['context_rewards'][index],
            'context': self.dataset['context'][index],
            'query_states': self.dataset['query_states'][index],
            'optimal_actions': self.dataset['optimal_actions'][index],
            'true_context': self.dataset['true_context'][index],
            'zeros': self.zeros,
        }

        if self.shuffle:
            perm = torch.randperm(self.horizon)
            res['context_states'] = res['context_states'][perm]
            res['context_actions'] = res['context_actions'][perm]
            res['context_next_states'] = res['context_next_states'][perm]
            res['context_rewards'] = res['context_rewards'][perm]
            res['context'] = res['context'][perm]
            res['true_context'] = res['true_context'][perm]

        return res


class ImageDataset(Dataset):
    """"Dataset class for image-based data."""

    def __init__(self, paths, config, transform):
        config['store_gpu'] = False
        super().__init__(paths, config)
        self.transform = transform
        self.config = config

        context_filepaths = []
        query_images = []

        for traj in self.trajs:
            context_filepaths.append(traj['context_images'])
            query_image = self.transform(traj['query_image']).float()
            query_images.append(query_image)

        self.dataset.update({
            'context_filepaths': context_filepaths,
            'query_images': torch.stack(query_images),
        })

    def __getitem__(self, index):
        'Generates one sample of data'
        filepath = self.dataset['context_filepaths'][index]
        context_images = np.load(filepath)
        context_images = [self.transform(images) for images in context_images]
        context_images = torch.stack(context_images).float()

        query_images = self.dataset['query_images'][index]

        res = {
            'context_images': context_images,#.to(device),
            'context_states': self.dataset['context_states'][index],
            'context_actions': self.dataset['context_actions'][index],
            'context_next_states': self.dataset['context_next_states'][index],
            'context_rewards': self.dataset['context_rewards'][index],
            'query_images': query_images,#.to(device),
            'query_states': self.dataset['query_states'][index],
            'optimal_actions': self.dataset['optimal_actions'][index],
            'zeros': self.zeros,
        }

        if self.shuffle:
            perm = torch.randperm(self.horizon)
            res['context_images'] = res['context_images'][perm]
            res['context_states'] = res['context_states'][perm]
            res['context_actions'] = res['context_actions'][perm]
            res['context_next_states'] = res['context_next_states'][perm]
            res['context_rewards'] = res['context_rewards'][perm]

        return res
