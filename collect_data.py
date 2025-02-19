import argparse
import os
import pickle
import random

import gymnasium as gym
import numpy as np

import common_args
from envs import bandit_env
from utils import build_data_filename


def rollin_bandit(env, cov, exp=False, orig=False):
    H = env.H
    opt_a_index = env.opt_a_index
    xs, us, xps, rs = [], [], [], []
    if cov is None:
        cov = np.random.choice([0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    
    if not exp:
        alpha = np.ones(env.dim)
        probs = np.random.dirichlet(alpha)
        probs2 = np.zeros(env.dim)
        rand_index = np.random.choice(np.arange(env.dim))
        probs2[rand_index] = 1.0
        probs = (1 - cov) * probs + cov * probs2
    else:
        alpha = np.ones(env.dim)
        probs = np.random.dirichlet(alpha)
        probs2 = np.zeros(env.dim)
        probs2[opt_a_index] = 1.0
        probs = (1 - cov) * probs + cov * probs2

    for h in range(H):
        x = np.array([1])
        u = np.zeros(env.dim)
        i = np.random.choice(np.arange(env.dim), p=probs)
        u[i] = 1.0
        xp, r = env.transit(x, u)

        xs.append(x)
        us.append(u)
        xps.append(xp)
        rs.append(r)

    xs, us, xps, rs = np.array(xs), np.array(us), np.array(xps), np.array(rs)
    ns = np.cumsum(us, axis=0)
    c = np.zeros((H, env.dim)) # Context prediction is not provided in offline dataset.
    
    return xs, us, xps, rs, c

def generate_bandit_histories_from_envs(envs, n_hists, n_samples, cov, type):
    trajs = []
    for env in envs:
        for j in range(n_hists):
            (
                context_states,
                context_actions,
                context_next_states,
                context_rewards,
                context,
            ) = rollin_bandit(env, cov=cov)
            for k in range(n_samples):
                query_state = np.array([1])
                optimal_action = np.array(env.opt_a)

                traj = {
                    'query_state': query_state,
                    'optimal_action': optimal_action,
                    'context_states': context_states,
                    'context_actions': context_actions,
                    'context_next_states': context_next_states,
                    'context_rewards': context_rewards,
                    'context': context,
                    'means': env.means,
                }
                trajs.append(traj)
    return trajs

def generate_bandit_histories(n_envs, dim, horizon, var, **kwargs):
    envs = [bandit_env.sample(dim, horizon, var)
            for _ in range(n_envs)]
    trajs = generate_bandit_histories_from_envs(envs, **kwargs)
    return trajs

# random data collection
def collect_data():
    if __name__ == '__main__':
        if not os.path.exists('datasets'):
            os.makedirs('datasets', exist_ok=True)

        np.random.seed(42)
        random.seed(42)

        parser = argparse.ArgumentParser()
        common_args.add_dataset_args(parser)
        args = vars(parser.parse_args())
        print("Args: ", args)

        env = args['env']
        n_envs = args['envs']
        n_eval_envs = args['envs_eval']
        n_hists = args['hists']
        n_samples = args['samples']
        horizon = args['H']
        dim = args['dim']
        var = args['var']
        cov = args['cov']
        rdm_fix_ratio = args['rdm_fix_ratio']

        n_train_envs = int(.8 * n_envs)

        config = {
            'n_hists': n_hists,
            'n_samples': n_samples,
            'horizon': horizon,
        }

        all_trajs = []
        eval_trajs = []
        
        n_envs_ratio1 = int(n_envs * rdm_fix_ratio[0])
        n_envs_ratio2 = int(n_envs * rdm_fix_ratio[1])

        if env == "bandit":
            config.update({'dim': dim, 'var': var, 'cov': cov, 'type': 'uniform'})
            trajs1 = generate_bandit_histories(n_envs_ratio1, **config)
            trajs = generate_bandit_histories(int(n_eval_envs), **config)
            config.update({'cov': 1.0})
            trajs2 = generate_bandit_histories(n_envs_ratio2, **config)

            config.update({'cov': cov, 'type': 'uniform'})

        all_trajs.extend(trajs2)
        all_trajs.extend(trajs1)
        eval_trajs.extend(trajs)

        random.shuffle(all_trajs)

        train_trajs = all_trajs[:n_train_envs]
        test_trajs = all_trajs[n_train_envs:]

        train_filepath = build_data_filename(env, n_envs, config, mode=0)
        with open(train_filepath, 'wb') as file:
            pickle.dump(train_trajs, file)
        print(f"Training data saved to {train_filepath}.")

        test_filepath = build_data_filename(env, n_envs, config, mode=1)
        with open(test_filepath, 'wb') as file:
            pickle.dump(test_trajs, file)
        print(f"Testing data saved to {test_filepath}.")

        eval_filepath = build_data_filename(env, n_eval_envs, config, mode=2)
        with open(eval_filepath, 'wb') as file:
            pickle.dump(eval_trajs, file)
        print(f"Evaluating data saved to {eval_filepath}.")



collect_data()
