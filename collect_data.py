import argparse
import os
import pickle
import random

import gymnasium as gym
import numpy as np

import common_args
from envs import bandit_env, darkroom_env
from utils import build_data_filename, build_darkroom_data_filename


def rollin_bandit(env, cov, exp=False, orig=False, style='default'):
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

    if style == 'convergent':
        # when set to 'convergent', optimal arm is always picked after h_cvg steps
        h_cvg = np.random.randint(1, H)
        for h in range(h_cvg):
            x = np.array([1])
            u = np.zeros(env.dim)
            i = np.random.choice(np.arange(env.dim), p=probs)
            u[i] = 1.0
            xp, r = env.transit(x, u)

            xs.append(x)
            us.append(u)
            xps.append(xp)
            rs.append(r)
        for h in range(h_cvg, H):
            x = np.array([1])
            u = np.zeros(env.dim)
            i = opt_a_index
            u[i] = 1.0
            xp, r = env.transit(x, u)
            xs.append(x)
            us.append(u)
            xps.append(xp)
            rs.append(r)
    else:
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

    # c = np.zeros((H, env.dim)) # Context prediction is not provided in offline dataset.
    num = np.cumsum(us, axis=0)
    cum_r = np.einsum('i,ij->ij', rs, us)
    cum_r = np.cumsum(cum_r, axis=0)
    c = cum_r / (num + 1e-8)
    
    return xs, us, xps, rs, c

def rollin_mdp(env, rollin_type):
    states = []
    actions = []
    next_states = []
    rewards = []

    state = env.reset()
    for _ in range(env.horizon):
        if rollin_type == 'uniform':
            state = env.sample_state()
            action = env.sample_action()
        elif rollin_type == 'expert':
            action = env.opt_action(state)
        else:
            raise NotImplementedError
        next_state, reward = env.transit(state, action)

        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        state = next_state

    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    rewards = np.array(rewards)
    contexts = np.zeros((env.horizon, env.state_dim * env.dim)) # Context prediction is not provided in offline dataset.

    return states, actions, next_states, rewards, contexts


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
                    'true_context': env.means,
                }
                trajs.append(traj)
    return trajs

def generate_mdp_histories_from_envs(envs, n_hists, n_samples, rollin_type):
    trajs = []
    for env in envs:
        for j in range(n_hists):
            (
                context_states,
                context_actions,
                context_next_states,
                context_rewards,
                contexts,
            ) = rollin_mdp(env, rollin_type=rollin_type)
            for k in range(n_samples):
                query_state = env.sample_state()
                optimal_action = env.opt_action(query_state)

                traj = {
                    'query_state': query_state,
                    'optimal_action': optimal_action,
                    'context_states': context_states,
                    'context_actions': context_actions,
                    'context_next_states': context_next_states,
                    'context_rewards': context_rewards,
                    'context': contexts,
                    'true_context': env.means,
                    # 'goal': env.goal,
                }

                # Add perm_index for DarkroomEnvPermuted
                if hasattr(env, 'perm_index'):
                    traj['perm_index'] = env.perm_index

                trajs.append(traj)
    return trajs


def generate_bandit_histories(n_envs, dim, horizon, var, **kwargs):
    perturb = np.random.choice([0.0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5], size=n_envs)
    envs = [bandit_env.sample(dim, horizon, var) 
            for _ in range(n_envs)]
    trajs = generate_bandit_histories_from_envs(envs, **kwargs)
    return trajs

def generate_darkroom_histories(goals, dim, horizon, **kwargs):
    envs = [darkroom_env.DarkroomEnv(dim, goal, horizon) for goal in goals]
    trajs = generate_mdp_histories_from_envs(envs, **kwargs)
    return trajs


def generate_darkroom_permuted_histories(indices, dim, horizon, **kwargs):
    envs = [darkroom_env.DarkroomEnvPermuted(
        dim, index, horizon) for index in indices]
    trajs = generate_mdp_histories_from_envs(envs, **kwargs)
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
        
        elif env == "darkroom":
            goals = [np.array([dim - 1, dim - 1]) for _ in range(n_envs)]
            config.update({'dim': dim, 'horizon': horizon, 'rollin_type': 'uniform'})
            trajs1 = generate_darkroom_histories(goals[:n_envs_ratio1], **config)
            trajs = generate_darkroom_histories(goals[n_envs_ratio1:], **config)
            config.update({'rollin_type': 'expert'})
            trajs2 = generate_darkroom_histories(goals[n_envs_ratio1:], **config)

            config.update({'rollin_type': 'uniform'})

        all_trajs.extend(trajs2)
        all_trajs.extend(trajs1)
        eval_trajs.extend(trajs)

        random.shuffle(all_trajs)

        train_trajs = all_trajs[:n_train_envs]
        test_trajs = all_trajs[n_train_envs:]

        if env == 'bandit':
            train_filepath = build_data_filename(env, n_envs, config, mode=0)
            test_filepath = build_data_filename(env, n_envs, config, mode=1)
            eval_filepath = build_data_filename(env, n_eval_envs, config, mode=2)
        
        elif env == 'darkroom':
            train_filepath = build_darkroom_data_filename(env, n_envs, config, mode=0)
            test_filepath = build_darkroom_data_filename(env, n_envs, config, mode=1)
            eval_filepath = build_darkroom_data_filename(env, n_eval_envs, config, mode=2)

        with open(train_filepath, 'wb') as file:
            pickle.dump(train_trajs, file)
        print(f"Training data saved to {train_filepath}.")

        with open(test_filepath, 'wb') as file:
            pickle.dump(test_trajs, file)
        print(f"Testing data saved to {test_filepath}.")

        with open(eval_filepath, 'wb') as file:
            pickle.dump(eval_trajs, file)
        print(f"Evaluating data saved to {eval_filepath}.")



collect_data()
