import argparse
import os
import pickle

import matplotlib.pyplot as plt
import torch
import common_args
from evals import eval_bandit
from net import Transformer, Context_extractor, pretrain_transformer
from utils import (
    build_data_filename,
    build_model_filename,
)
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    common_args.add_train_args(parser)
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
    w = args['exploration_rate']
    
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

    dpt_config = {
        'class': 'dpt',
        'shuffle': shuffle,
        'lr': lr,
        'dropout': dropout,
        'n_embd': 32,
        'n_layer': 4,
        'n_head': 4,
        'n_envs': n_envs,
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
        'seed': seed,
    }

    ppt_config = {
        'class': 'ppt',
        'exploration_rate': w,
        'shuffle': shuffle,
        'lr': lr,
        'dropout': dropout,
        'n_embd': 32,
        'n_layer': 4,
        'n_head': 4,
        'n_envs': n_envs,
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
        'seed': seed,
    }

    if envname == 'bandit':
        state_dim = 1

        dpt_config.update({'var': var, 'cov': cov})
        ppt_config.update({'var': var, 'cov': cov})
        filename1 = build_model_filename(envname, dpt_config)
        filename2 = build_model_filename(envname, ppt_config)

        bandit_type = 'uniform'
    # elif envname == 'bandit_bernoulli':
    #     state_dim = 1

    #     model_config.update({'var': var, 'cov': cov})
    #     filename = build_model_filename(envname, model_config)
    #     bandit_type = 'bernoulli'
    else:
        raise NotImplementedError

    config1 = {
        'horizon': H,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': 4,
        'n_embd': 32,
        'n_head': 4,
        'dropout': dropout,
        'test': True,
    }
    config2 = {
        'horizon': H,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': 4,
        'n_embd': 32,
        'n_head': 4,
        'dropout': dropout,
        'test': True,
    }

    # Load network from saved file.
    # By default, load the final file, otherwise load specified epoch.

    model_pre = pretrain_transformer(config1).to(device)
    model_ctx = Context_extractor(config2).to(device)
    model_act = Transformer(config2).to(device)
    
    tmp_filename1 = filename1
    tmp_filename2 = filename2

    if epoch < 0:
        model_path_pre = f'models/{tmp_filename1}.pt'
        model_path_act = f'models/{tmp_filename2}_model_act.pt'
        model_path_ctx = f'models/{tmp_filename2}_model_ctx.pt'
    else:
        model_path_pre = f'models/{tmp_filename1}_epoch{epoch}.pt'
        model_path_act = f'models/{tmp_filename2}_model_act_epoch{epoch}.pt'
        model_path_ctx = f'models/{tmp_filename2}_model_ctx_epoch{epoch}.pt'

    checkpoint_pre = torch.load(model_path_pre, map_location=device)
    checkpoint_act = torch.load(model_path_act, map_location=device)
    checkpoint_ctx = torch.load(model_path_ctx, map_location=device)

    model_pre.load_state_dict(checkpoint_pre)
    model_pre.eval()
    model_act.load_state_dict(checkpoint_act)
    model_act.eval()
    model_ctx.load_state_dict(checkpoint_ctx)
    model_ctx.eval()

    dataset_config = {
        'horizon': horizon,
        'dim': dim,
    }

    if envname in ['bandit', 'bandit_bernoulli']:
        dataset_config.update({'var': var, 'cov': cov, 'type': 'uniform'})
        eval_filepath = build_data_filename(
            envname, n_eval, dataset_config, mode=2)
        save_filename = f'{filename2}_testcov{test_cov}_hor{horizon}.pkl'

    else:
        raise ValueError(f'Environment {envname} not supported')


    with open(eval_filepath, 'rb') as f:
        eval_trajs = pickle.load(f)

    n_eval = min(n_eval, len(eval_trajs))


    evals_filename = f"evals_epoch{epoch}"
    if not os.path.exists(f'figs/{evals_filename}'):
        os.makedirs(f'figs/{evals_filename}', exist_ok=True)
    if not os.path.exists(f'figs/{evals_filename}/bar'):
        os.makedirs(f'figs/{evals_filename}/bar', exist_ok=True)
    if not os.path.exists(f'figs/{evals_filename}/online'):
        os.makedirs(f'figs/{evals_filename}/online', exist_ok=True)
    if not os.path.exists(f'figs/{evals_filename}/graph'):
        os.makedirs(f'figs/{evals_filename}/graph', exist_ok=True)
    if not os.path.exists(f'figs/{evals_filename}/online_sample'):
        os.makedirs(f'figs/{evals_filename}/online_sample', exist_ok=True)

    # Online and offline evaluation.
    if envname == 'bandit':
        simulated_var = 1.0
        config = {
            'horizon': horizon,
            'var': simulated_var,
            'n_eval': n_eval,
            'exploration_rate': w,
            # 'bandit_type': bandit_type,
        }

        save_filename = f'{save_filename}_simvar{simulated_var}'
        
        model_list = [(model_pre, 'dpt'), ([model_act, model_ctx], 'ppt')]
        eval_bandit.online(eval_trajs, model_list=model_list, **config, evals_filename = evals_filename, save_filename = save_filename)
        plt.savefig(f'figs/{evals_filename}/online/{save_filename}.png')
        plt.clf()
        plt.cla()
        plt.close()

        # means = 0.1 * np.random.randint(1, 9, size=(10,3))
        # for j in range(10):
        #     eval_bandit.online_sample(
        #         model=[model_act, model_ctx], model0=model_pre, means = means[j], horizon = horizon, var = simulated_var)
        #     plt.savefig(f'figs/{evals_filename}/online_sample/{save_filename}_{means[j]}.png')
        #     plt.clf()
        #     plt.cla()
        #     plt.close()
        
        # eval_bandit.offline(eval_trajs, [model1, model2], **config)
        # plt.savefig(f'figs/{evals_filename}/bar/{save_filename}_bar.png')
        # plt.clf()

        # eval_bandit.offline_graph(eval_trajs, model, **config)
        # plt.savefig(f'figs/{evals_filename}/graph/{save_filename}_graph.png')
        # plt.clf()
        
    else:
        raise NotImplementedError