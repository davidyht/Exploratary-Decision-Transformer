import argparse
import os
import pickle

import matplotlib.pyplot as plt
import torch
from IPython import embed

import common_args
from evals import eval_bandit, eval_cgbandit, eval_base
from envs import bandit_env, cg_bandit
from ctrls.ctrl_bandit import BanditTransformerController
from net import Transformer, ImageTransformer
from utils import (
    build_cgbandit_data_filename,
    build_cgbandit_model_filename,
    build_bandit_data_filename,
    build_bandit_model_filename,
    build_linear_bandit_data_filename,
    build_linear_bandit_model_filename,
    build_darkroom_data_filename,
    build_darkroom_model_filename,
    build_miniworld_data_filename,
    build_miniworld_model_filename,
)
import numpy as np
import scipy
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    common_args.add_eval_args(parser)
    parser.add_argument('--seed', type=int, default=0)

    args = vars(parser.parse_args())
    print("Args: ", args)

    n_envs = args['envs']
    n_hists = args['hists']
    n_samples = args['samples']
    H = args['H']
    dim = args['dim']
    state_dim = dim
    action_dim = dim
    n_embd = args['embd']
    n_head = args['head']
    n_layer = args['layer']
    lr = args['lr']
    epoch = args['epoch']
    shuffle = args['shuffle']
    dropout = args['dropout']
    var = args['var']
    cov = args['cov']
    test_cov = args['test_cov']
    envname = args['env']
    horizon = args['hor']
    n_eval = args['n_eval']
    seed = args['seed']
    lin_d = args['lin_d']
    
    tmp_seed = seed
    if seed == -1:
        tmp_seed = 0

    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tmp_seed)
    np.random.seed(tmp_seed)

    if test_cov < 0:
        test_cov = cov
    if horizon < 0:
        horizon = H

    model_config = {
        'shuffle': shuffle,
        'lr': lr,
        'dropout': dropout,
        'n_embd': n_embd,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_envs': n_envs,
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
        'seed': seed,
    }
    if envname == 'cgbandit':
        state_dim = 1

        model_config.update({'var': var, 'cov': cov})
        filename = build_cgbandit_model_filename(envname, model_config)
        bandit_type = 'uniform'
    elif envname == 'bandit':
        state_dim = 1

        model_config.update({'var': var, 'cov': cov})
        filename = build_bandit_model_filename(envname, model_config)
        bandit_type = 'uniform'
    elif envname == 'bandit_bernoulli':
        state_dim = 1

        model_config.update({'var': var, 'cov': cov})
        filename = build_bandit_model_filename(envname, model_config)
        bandit_type = 'bernoulli'
    elif envname == 'linear_bandit':
        state_dim = 1

        model_config.update({'lin_d': lin_d, 'var': var, 'cov': cov})
        filename = build_linear_bandit_model_filename(envname, model_config)
    elif envname.startswith('darkroom'):
        state_dim = 2
        action_dim = 5

        filename = build_darkroom_model_filename(envname, model_config)
    elif envname == 'miniworld':
        state_dim = 2
        action_dim = 4

        filename = build_miniworld_model_filename(envname, model_config)
    else:
        raise NotImplementedError

    config = {
        'horizon': H,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'dropout': dropout,
        'test': True,
    }

    # Load network from saved file.
    # By default, load the final file, otherwise load specified epoch.
    if envname == 'miniworld':
        config.update({'image_size': 25})
        model = ImageTransformer(config).to(device)
    else:
        model = Transformer(config).to(device)
    
    tmp_filename = filename
    if epoch < 0:
        model_path = f'models/{tmp_filename}.pt'
    else:
        model_path = f'models/{tmp_filename}_epoch{epoch}.pt'
    
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    env1 = bandit_env.sample(5, 500, 0.1)
    states = []
    actions = []
    next_states = []
    rewards = []
    query_states = [1]
    zeros = torch.zeros(horizon, action_dim + state_dim * 3 + 1)
    for i in range(horizon):
        state = env1.reset()
        states[i] = torch.tensor(state)

        controller = BanditTransformerController(model, device, du = action_dim)
        action = controller.act(states, actions, rewards, next_states, query_states, zeros)
        next_state, reward = env1.step(action)

        actions[i] = torch.tensor(action)
        next_states[i] = torch.tensor(next_state)
        rewards[i] = torch.tensor(reward)

    print("rewards: ", rewards)