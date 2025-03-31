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
    state_dim = 1
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

    exploration_rate_list = []

    if not exploration_rate_list:  # 检查列表是否为空
        print("Exploration rate list is empty. Skipping PPT-related logic.")
    else:
        ppt_configs = {}
        ppt_filenames = {}
        ppt_models = {}
        for w in exploration_rate_list:
            ppt_config = {
                        'class': 'ppt',
                        'exploration_rate': w,
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
            if envname == 'bandit':
                ppt_config.update({'var': var, 'cov': cov})
                ppt_filenames[w] = build_model_filename(envname, ppt_config)

            ppt_configs[w] = ppt_config
            
            ppt_config_model = {
                'horizon': H,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'n_layer': n_layer,
                'n_embd': n_embd,
                'n_head': n_head,
                'dropout': dropout,
                'test': True,
            }
            model_ctx = Context_extractor(ppt_config_model).to(device)
            model_act = Transformer(ppt_config_model).to(device)

            tmp_filename_ppt = ppt_filenames[w]
            if epoch < 0:
                model_path_act = f'models/{tmp_filename_ppt}_model_act.pt'
                model_path_ctx = f'models/{tmp_filename_ppt}_model_ctx.pt'
            else:
                model_path_act = f'models/{tmp_filename_ppt}_model_act_epoch{epoch}.pt'
                model_path_ctx = f'models/{tmp_filename_ppt}_model_ctx_epoch{epoch}.pt'
            
            checkpoint_act = torch.load(model_path_act, map_location=device)
            checkpoint_ctx = torch.load(model_path_ctx, map_location=device)

            model_act.load_state_dict(checkpoint_act)
            model_act.eval()
            model_ctx.load_state_dict(checkpoint_ctx)
            model_ctx.eval()

            ppt_models[w] = (model_act, model_ctx)

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
  
    if envname == 'bandit':
        state_dim = 1

        dpt_config.update({'var': var, 'cov': cov})
        dpt_filename = build_model_filename(envname, dpt_config)
        bandit_type = 'uniform'
    else:
        raise NotImplementedError

    dpt_config_model = {
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

    dpt_model = pretrain_transformer(dpt_config_model).to(device)
    
    tmp_filename_dpt = dpt_filename

    if epoch < 0:
        model_path_pre = f'models/{tmp_filename_dpt}.pt'
    else:
        model_path_pre = f'models/{tmp_filename_dpt}_epoch{epoch}.pt'

    checkpoint_pre = torch.load(model_path_pre, map_location=device)

    dpt_model.load_state_dict(checkpoint_pre)
    dpt_model.eval()

    model_list = [(dpt_model, 'dpt')]
    if exploration_rate_list:
        for w in exploration_rate_list:
            model_list.append((ppt_models[w], f'ppt_{w}'))


    dataset_config = {
        'horizon': horizon,
        'dim': dim,
    }

    if envname in ['bandit', 'bandit_bernoulli']:
        dataset_config.update({'var': var, 'cov': cov, 'type': 'uniform'})
        eval_filepath = build_data_filename(
            envname, n_eval, dataset_config, mode=2)
        save_filename = f'{dpt_filename}_testcov{test_cov}_hor{horizon}.pkl'

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
        regret_values = {}

        for simulated_var in [0.3, 0.5, 0.8]:
            config = {
                'horizon': horizon,
                'var': simulated_var,
                'n_eval': n_eval,
                # 'bandit_type': bandit_type,
            }

            save_filename_online = f'{save_filename}_simvar{simulated_var}'

            regret_means = eval_bandit.online(eval_trajs, model_list=model_list, **config)

            regret_values[simulated_var] = regret_means

            plt.savefig(f'figs/{evals_filename}/online/{save_filename_online}.png')
            plt.clf()
            plt.cla()
            plt.close()

        fig, ax = plt.subplots(figsize=(12, 8))

        sim_vars = list(regret_values.keys())
        keys = list(next(iter(regret_values.values())).keys())

        bar_width = 0.1  # Decrease the width of the bar
        alpha = 0.7  # Increase the transparency

        offset = np.arange(len(sim_vars))

        for i, key in enumerate(keys):
            values = [regret_values[sim_var][key] for sim_var in sim_vars]
            ax.bar(offset + i * bar_width, values, bar_width, alpha=alpha, label=f'{key}')

        # Highlight the best algorithm for each sim_var
        for j, sim_var in enumerate(sim_vars):
            best_key = min(regret_values[sim_var], key=regret_values[sim_var].get)
            best_value = regret_values[sim_var][best_key]
            best_index = keys.index(best_key)
            bars = ax.bar(offset[j] + best_index * bar_width, best_value, bar_width, alpha=1.0, edgecolor='black', linewidth=2)
            for bar in bars:
                bar.set_facecolor(ax.patches[best_index].get_facecolor())

        ax.set_xticks(offset + bar_width * (len(keys) - 1) / 2)
        ax.set_xticklabels(sim_vars)

        ax.set_xlabel('Sim Variance')
        ax.set_ylabel('Cumulative Regret')
        ax.set_title('Cumulative Regret for Different Sim Variances')
        ax.legend()

        plt.savefig(f'figs/{evals_filename}/suboptimality_comparison.png')


        # means = np.random.randint(50, 55, (10, dim)) * 0.01

        # for j in range(10):

        #     eval_bandit.online_sample(
        #         model_list = model_list, means = means[j], horizon = horizon, var = 0.3,  type='easy')
        #     plt.savefig(f'figs/{evals_filename}/online_sample/{save_filename_online}_{means[j]}.png')
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