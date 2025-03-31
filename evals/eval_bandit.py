import numpy as np
import scipy
import torch
import seaborn as sns
from IPython import embed
from evals.eval_base import deploy_online, deploy_online_vec
from envs.bandit_env import BanditEnvVec, BanditEnv, HardBanditEnv, HardBanditEnvVec
from utils import convert_to_tensor

import matplotlib.pyplot as plt



from ctrls.ctrl_bandit import (
    BanditTransformerController,
    BanditMOTransformerController,
    OptPolicy,
    UCBPolicy,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def online_sample(model_list, means, horizon, var, type = 'easy'):
    all_means = {}
    all_context = {}
    if type == 'easy':
        env = BanditEnv(means, horizon, var=var)
    elif type == 'hard':
        env = HardBanditEnv(means, horizon, var=var)
    true_context = np.array(means)
    true_context = true_context.reshape(1, -1)
    
    # Optimal policy
    controller = OptPolicy(
        env = env,
        batch_size=1)
    cum_means = deploy_online(env, controller, horizon).T
    all_means['opt'] = cum_means
    if type == 'hard':
        env.reset_optimal_picks()

    for (model, model_class) in model_list:
        if model_class == 'dpt':
            controller = BanditTransformerController(
                model,
                sample=True,
                batch_size=1)
            cum_means = deploy_online(env, controller, horizon).T
            all_means['dpt'] = cum_means
            if type == 'hard':
                env.reset_optimal_picks()
        
        elif model_class.startswith('ppt'):
            controller = BanditMOTransformerController(
                model,
                sample=True,
                batch_size=1)
            cum_means, meta = deploy_online(env, controller, horizon, include_meta=True)
            cum_means = cum_means.T
            context = meta['context']
            context = np.array(context)
            context_loss = np.zeros(horizon)
            
            for i in range(horizon):
                context_loss[i] = np.sum((context[:, i].squeeze() - true_context)**2)

            all_means[model_class] = cum_means
            all_context[model_class] = context_loss
            if type == 'hard':
                env.reset_optimal_picks()

    # UCB policy
    controller = UCBPolicy(
        env,
        const=1.0,
        batch_size=1)
    cum_means = deploy_online(env, controller, horizon).T
    all_means['UCB1.0'] = cum_means
    if type == 'hard':
        env.reset_optimal_picks()

    # Convert to numpy arrays
    all_means = {k: np.array(v) for k, v in all_means.items()}
    
    all_means_diff = {k: all_means['opt'] - v for k, v in all_means.items()}

    # Calculate cumulative regret
    cumulative_regret = {k: np.cumsum(v) for k, v in all_means_diff.items()}
    regret = {k: np.sum(v) for k, v in all_means_diff.items()}

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))

    # Plot suboptimality
    for key in all_means.keys():
        if key != 'opt' and key != 'UCB1.0' and key != 'dpt':
            ax1.scatter(np.arange(horizon), all_means[key], label=key, alpha=0.5)

    ax1.set_yscale('log')
    ax1.set_xlabel('Time steps')
    ax1.set_ylabel('Suboptimality')
    ax1.set_title('Online Evaluation')
    ax1.legend()

    # Plot cumulative regret
    for key in cumulative_regret.keys():
        if key != 'opt':
            ax2.plot(cumulative_regret[key], label=key)

    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('Regret Over Time')
    ax2.legend()

    # plot context loss
    for key in all_context.keys():
        ax3.plot(all_context[key], label=key)
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Context Loss')
    ax3.set_title('Context Loss')
    ax3.legend()


def online(eval_trajs, model_list, n_eval, horizon, var, type = 'easy'):
    # Dictionary to store means of different policies
    all_means = {}
    all_context = {}

    envs = []
    true_context = []
    for i_eval in range(n_eval):
        traj = eval_trajs[i_eval]
        means = traj['true_context']
        true_context.append(means)

        # Create bandit environment
        if type == 'easy':
            env = BanditEnv(means, horizon, var=var)    
        elif type == 'hard':
            env = HardBanditEnv(means, horizon, var=var)
        
        envs.append(env)

    if type == 'easy':
        vec_env = BanditEnvVec(envs)
    elif type == 'hard':
        vec_env = HardBanditEnvVec(envs)
    true_context = np.array(true_context)

    # Optimal policy
    controller = OptPolicy(
        envs,
        batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    all_means['opt'] = cum_means
    if type == 'hard':
        vec_env.reset_optimal_picks()

    for (model, model_class) in model_list:
        if model_class == 'dpt':
            controller = BanditTransformerController(
                model,
                sample=True,
                batch_size=len(envs))
            cum_means = deploy_online_vec(vec_env, controller, horizon).T
            all_means['dpt'] = cum_means
            if type == 'hard':
                vec_env.reset_optimal_picks()        
        elif model_class.startswith('ppt'):
            controller = BanditMOTransformerController(
                model,
                sample=True,
                batch_size=len(envs))
            cum_means, meta = deploy_online_vec(vec_env, controller, horizon, include_meta=True)
            cum_means = cum_means.T
            context = meta['context']
            context_loss = np.zeros(horizon)
            for i in range(horizon):
                batch_mean_loss = np.mean((context[:, i] - true_context)**2, axis=0)
                context_loss[i] = np.sum(batch_mean_loss)

            all_means[model_class] = cum_means
            all_context[model_class] = context_loss
            if type == 'hard':
                vec_env.reset_optimal_picks()


    # UCB policy
    controller = UCBPolicy(
        envs[0],
        const=1.0,
        batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    all_means['UCB1.0'] = cum_means
    if type == 'hard':
        vec_env.reset_optimal_picks

    # Convert to numpy arrays
    all_means = {k: np.array(v) for k, v in all_means.items()}

    # Update all_means to have 1 if the corresponding element in all_means['opt'] is higher, otherwise 0
    for key in all_means.keys():
        if key != 'opt':
            all_means[key] = np.where(all_means['opt'] > all_means[key], 0, 1)
    
    all_means['opt'] = np.ones(all_means['opt'].shape)
    all_means_diff = {k: all_means['opt'] - v for k, v in all_means.items()}
    # Calculate means and standard errors
    means = {k: np.mean(v, axis=0) for k, v in all_means_diff.items()}
    sems = {k: scipy.stats.sem(v, axis=0) for k, v in all_means_diff.items()}

    # Calculate cumulative regret
    cumulative_regret = {k: np.cumsum(v, axis=1) for k, v in all_means_diff.items()}
    regret = {k: np.sum(v, axis=1) for k, v in all_means_diff.items()}

    regret_means = {k: np.mean(v, axis=0) for k, v in cumulative_regret.items()}
    regret_sems = {k: scipy.stats.sem(v, axis=0) for k, v in cumulative_regret.items()}

    # Plotting
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(28, 6), constrained_layout=True)

    # Plot suboptimality
    for key in means.keys():
        if key != 'opt':
            ax1.plot(means[key], label=key)
            ax1.fill_between(np.arange(horizon), means[key] - sems[key], means[key] + sems[key], alpha=0.2)

    ax1.set_yscale('log')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Suboptimality')
    ax1.set_title('Online Evaluation')
    ax1.legend()

    # Plot cumulative regret
    for key in regret_means.keys():
        if key != 'opt':
            ax2.plot(regret_means[key], label=key)
            ax2.fill_between(np.arange(horizon), regret_means[key] - regret_sems[key], regret_means[key] + regret_sems[key], alpha=0.2)

    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('Regret Over Time')
    ax2.legend()

    # plot regret distribution
    for key in regret.keys():
        if key != 'opt' and key != 'UCB1.0':
            # ax3.hist(regret[key], bins=20, alpha=0.5, label=key)
            sns.kdeplot(regret[key], ax=ax3, label=key, shade=True, alpha=0.5)
    ax3.set_xlabel('Regret')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Regret Distribution')
    ax3.legend()

    # plot context loss
    for key in all_context.keys():
        ax4.plot(all_context[key], label=key)
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Context Loss')
    ax4.set_title('Context Loss')
    ax4.legend()

    regret = {k: v for k, v in regret.items() if k != 'UCB1.0'}
    regret_avg = {k: np.mean(v) for k, v in regret.items()}

    return regret_avg




def offline(eval_trajs, model, n_eval, horizon, var, bandit_type):
    # Lists to store rewards for different policies
    all_rs_lnr = []
    all_rs_greedy = []
    all_rs_opt = []
    all_rs_emp = []
    all_rs_pess = []
    all_rs_thmp = []

    num_envs = len(eval_trajs)

    tmp_env = BanditEnv(eval_trajs[0]['means'], horizon, var=var)
    context_states = np.zeros((num_envs, horizon, tmp_env.dx))
    context_actions = np.zeros((num_envs, horizon, tmp_env.du))
    context_next_states = np.zeros((num_envs, horizon, tmp_env.dx))
    context_rewards = np.zeros((num_envs, horizon, 1))

    envs = []

    print(f"Evaling offline horizon: {horizon}", end='\r')

    for i_eval in range(n_eval):
        # print(f"Eval traj: {i_eval}")
        traj = eval_trajs[i_eval]
        means = traj['means']

        # Create bandit environment
        env = BanditEnv(means, horizon, var=var)
        envs.append(env)
       #  assert len(traj['context_states']) >= horizon, "context_states is too short"
        assert len(traj['context_actions']) >= horizon, "context_actions is too short"
        assert len(traj['context_next_states']) >= horizon, "context_next_states is too short"
        assert len(traj['context_rewards']) >= horizon, "context_rewards is too short"


        # Update context variables
        context_states[i_eval, :, :] = traj['context_states'][:horizon]
        context_actions[i_eval, :, :] = traj['context_actions'][:horizon]
        context_next_states[i_eval, :, :] = traj['context_next_states'][:horizon]
        context_rewards[i_eval, :, :] = traj['context_rewards'][:horizon, None]

    vec_env = BanditEnvVec(envs)
    batch = {
        'context_states': context_states,
        'context_actions': context_actions,
        'context_next_states': context_next_states,
        'context_rewards': context_rewards,
    }

    # Optimal policy
    opt_policy = OptPolicy(envs, batch_size=num_envs)
    lnr_policy = BanditTransformerController(model, sample=False, batch_size=num_envs)

    # Set batch for each policy
    opt_policy.set_batch_numpy_vec(batch)
    lnr_policy.set_batch_numpy_vec(batch)

    # Deploy policies and collect rewards
    _, _, _, rs_opt = vec_env.deploy_eval(opt_policy)
    _, _, _, rs_lnr = vec_env.deploy_eval(lnr_policy)

    # Store rewards
    all_rs_opt = np.array(rs_opt)
    all_rs_lnr = np.array(rs_lnr)

    baselines = {
        'opt': all_rs_opt,
        'lnr': all_rs_lnr,
        'emp': all_rs_emp,
        'thmp': all_rs_thmp,
    }
    baselines_means = {k: np.mean(v) for k, v in baselines.items()}

    # Plot mean rewards
    colors = plt.cm.viridis(np.linspace(0, 1, len(baselines_means)))
    plt.bar(baselines_means.keys(), baselines_means.values(), color=colors)
    plt.title(f'Mean Reward on {n_eval} Trajectories')

    return baselines


def offline_graph(eval_trajs, model, n_eval, horizon, var, bandit_type):
    horizons = np.linspace(1, horizon, 50, dtype=int)

    all_means = []
    all_sems = []
    for h in horizons:
        config = {
            'horizon': h,
            'var': var,
            'n_eval': n_eval,
            'bandit_type': bandit_type,
        }
        config['horizon'] = h
        baselines = offline(eval_trajs, model, **config)
        plt.clf()

        means = {k: np.mean(v, axis=0) for k, v in baselines.items()}
        sems = {k: scipy.stats.sem(v, axis=0) for k, v in baselines.items()}
        all_means.append(means)

    # Plot suboptimality over different horizons
    for key in means.keys():
        if not key == 'opt':
            regrets = [all_means[i]['opt'] - all_means[i][key] for i in range(len(horizons))]
            plt.plot(horizons, regrets, label=key)
            plt.fill_between(horizons, regrets - sems[key], regrets + sems[key], alpha=0.2)

    plt.legend()
    plt.yscale('log')
    plt.xlabel('Dataset size')
    plt.ylabel('Suboptimality')
    config['horizon'] = horizon
