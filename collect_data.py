import argparse
import os
import pickle
import random

import gymnasium as gym
import numpy as np

import common_args
from envs import cg_bandit, bandit_env
from utils import build_data_filename, build_merge_data_filename


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
    ms = np.zeros((H, env.dim))
    for h in range(H):
        ms[h, :] = rs[h] * us[h, :]
    ms = np.cumsum(ms, axis=0)
    for h in range(H):
        ms[h, :] = ms[h, :] / (ns[h] + 1e-6)
    
    c = ms
    
    return xs, us, xps, rs, c

def rollin_cgbandit(env, cov, exp=True, orig=False):
    H = env.H
    T = env.cg_time
    pre_opt_a_index = env.pre_opta_index
    post_opt_a_index = env.post_opta_index
    xs, us, xps, rs = [], [], [], []
    if cov is None:
        cov = np.random.choice([0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    
    if not exp:
        alpha = np.ones(env.dim)
        probs = np.random.dirichlet(alpha)
        probs2 = np.zeros(env.dim)
        rand_index = np.random.choice(np.arange(env.dim))
        probs2[rand_index] = 1.0
        pre_probs = (1 - cov) * probs + cov * probs2
        post_probs = pre_probs
    else:
        alpha = np.ones(env.dim)
        probs = np.random.dirichlet(alpha)
        pre_probs2 = np.zeros(env.dim)
        pre_probs2[pre_opt_a_index] = 1.0
        post_probs2 = np.zeros(env.dim)
        post_probs2[post_opt_a_index] = 1.0
        pre_probs = (1 - cov) * probs + cov * pre_probs2
        post_probs = (1 - cov) * probs + cov * post_probs2

    for h in range(T):
        x = np.array([1])
        u = np.zeros(env.dim)
        i = np.random.choice(np.arange(env.dim), p=pre_probs)
        u[i] = 1.0
        xp, r = env.transit(u)

        xs.append(x)
        us.append(u)
        xps.append(xp)
        rs.append(r)

    for h in range(T, H):
        x = np.array([1])
        u = np.zeros(env.dim)
        i = np.random.choice(np.arange(env.dim), p=post_probs)
        u[i] = 1.0
        xp, r = env.transit(u)

        xs.append(x)
        us.append(u)
        xps.append(xp)
        rs.append(r)

    xs, us, xps, rs = np.array(xs), np.array(us), np.array(xps), np.array(rs)
    ns1 = np.cumsum(us[:T], axis=0)
    ns2 = np.cumsum(us[T:], axis=0)
    ms1 = np.zeros((T, env.dim))
    ms2 = np.zeros((H - T, env.dim))

    for h in range(T):
        ms1[h, :] = rs[h] * us[h, :]
    for h in range(T, H):
        ms2[h - T, :] = rs[h] * us[h, :]
    ms1 = np.cumsum(ms1, axis=0)
    ms2 = np.cumsum(ms2, axis=0)
    for h in range(T):
        ms1[h, :] = ms1[h, :] / (ns1[h] + 1e-6)
    for h in range(T, H):
        ms2[h - T, :] = ms2[h - T, :] / (ns2[h - T] + 1e-6)
    ms = np.concatenate((ms1, ms2), axis=0)
    cs = ms
    return xs, us, xps, rs, cs

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
                optimal_action = np.concatenate((env.opt_a, env.opt_a), axis=0)

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

def generate_cgbandit_histories_from_envs(envs, n_hists, n_samples, cov, type):
    trajs = []
    for env in envs:
        for j in range(n_hists):
            (
                context_states,
                context_actions,
                context_next_states,
                context_rewards,
                context,
            ) = rollin_cgbandit(env, cov=cov)
            for k in range(n_samples):
                query_state = np.array([1])
                optimal_action = np.concatenate((env.pre_opt_a, env.post_opt_a), axis=0)

                traj = {
                    'query_state': query_state,
                    'optimal_action': optimal_action,
                    'context_states': context_states,
                    'context_actions': context_actions,
                    'context_next_states': context_next_states,
                    'context_rewards': context_rewards,
                    'context': context,
                    'means': env.means,
                    'cg_time': env.cg_time,
                }
                trajs.append(traj)
    return trajs

def generate_cgbandit_histories(n_envs, dim, horizon, var, **kwargs):
    envs = [cg_bandit.sample(dim, horizon, var)
            for _ in range(n_envs)]
    trajs = generate_cgbandit_histories_from_envs(envs, **kwargs)
    return trajs

def generate_bandit_histories(n_envs, dim, horizon, var, **kwargs):
    envs = [bandit_env.sample(dim, horizon, var)
            for _ in range(n_envs)]
    trajs = generate_bandit_histories_from_envs(envs, **kwargs)
    return trajs

# random data collection
def collect_data():
    if __name__ == '__main__':
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

        elif env == "cgbandit":
            config.update({'dim': dim, 'var': var, 'cov': cov, 'type': 'uniform'})
            trajs1 = generate_cgbandit_histories(n_envs_ratio1, **config)
            trajs = generate_cgbandit_histories(int(n_eval_envs), **config)
            config.update({'cov': 1.0})
            trajs2 = generate_cgbandit_histories(n_envs_ratio2, **config)

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

# Collect data with both fix-arm and random-arm trajectories
def collect_merge_data():
    if __name__ == '__main__':
        np.random.seed(0)
        random.seed(0)

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
        env_names = args['env_names']
        ratio = args['ratio']
        rdm_fix_ratio = args['rdm_fix_ratio']

        n_train_envs = int(.8 * n_envs)

        config = {
            'n_hists': n_hists,
            'n_samples': n_samples,
            'horizon': horizon,
        }

        if len(set(env_names)) != len(env_names):
            raise ValueError("env_names should not contain repeated elements.")
        valid_env_names = ["bandit", "cgbandit"]
        for env_name in env_names:
            if env_name not in valid_env_names:
                raise ValueError(f"Invalid environment name: {env_name}.")

        all_trajs = []
        eval_trajs = []
        
        for env_name, ratio in zip(env_names, ratio):
            n_envs_ratio1 = int(ratio * n_envs * rdm_fix_ratio[0])
            n_envs_ratio2 = int(ratio * n_envs * rdm_fix_ratio[1])
            if env_name == "bandit":
                config.update({'dim': dim, 'var': var, 'cov': cov, 'type': 'uniform'})
                trajs1 = generate_bandit_histories(n_envs_ratio1, **config)
                eval_trajs = generate_bandit_histories(int(n_eval_envs * ratio), **config)
                config.update({'cov': 1.0})
                trajs2 = generate_bandit_histories(n_envs_ratio2, **config)

                config.update({'cov': cov, 'type': 'uniform'})

            elif env_name == "cgbandit":
                config.update({'dim': dim, 'var': var, 'cov': cov, 'type': 'uniform'})
                trajs1 = generate_cgbandit_histories(n_envs_ratio1, **config)
                trajs = generate_cgbandit_histories(int(n_eval_envs * ratio), **config)
                config.update({'cov': 1.0})
                trajs2 = generate_cgbandit_histories(n_envs_ratio2, **config)

                config.update({'cov': cov, 'type': 'uniform'})
            all_trajs.extend(trajs2)
            all_trajs.extend(trajs1)
            eval_trajs.extend(trajs)

        random.shuffle(all_trajs)

        train_trajs = all_trajs[:n_train_envs]
        test_trajs = all_trajs[n_train_envs:]

        train_filepath = build_merge_data_filename(env_names, n_envs, config, mode=0)
        with open(train_filepath, 'wb') as file:
            pickle.dump(train_trajs, file)
        print(f"Training data saved to {train_filepath}.")

        test_filepath = build_merge_data_filename(env_names, n_envs, config, mode=1)
        with open(test_filepath, 'wb') as file:
            pickle.dump(test_trajs, file)
        print(f"Testing data saved to {test_filepath}.")

        eval_filepath = build_merge_data_filename(env_names, n_eval_envs, config, mode=2)
        with open(eval_filepath, 'wb') as file:
            pickle.dump(eval_trajs, file)
        print(f"Evaluating data saved to {eval_filepath}.")


collect_data()
