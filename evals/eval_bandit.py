import numpy as np
import scipy
import torch
import seaborn as sns
from IPython import embed
from evals.eval_base import deploy_online, deploy_online_vec
from envs.bandit_env import BanditEnvVec, BanditEnv
from utils import convert_to_tensor

import matplotlib.pyplot as plt



from ctrls.ctrl_bandit import (
    BanditTransformerController,
    BanditMOTransformerController,
    OptPolicy,
    UCBPolicy,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def online_sample(model0, model,  means, horizon, var):
    all_means = {}
    env = BanditEnv(means, horizon, var=var)

    controller = OptPolicy(
        env,
        batch_size=1)
    cum_means = deploy_online(env, controller, horizon).T
    all_means['opt'] = cum_means

    env.reset()
    controller = BanditTransformerController(
        model0,
        sample=True,
        batch_size=1)
    cum_means = deploy_online(env, controller, horizon).T
    all_means['trf'] = cum_means

    env.reset()
    controller = BanditMOTransformerController(
        model,
        sample=True,
        batch_size=1
    )
    cum_means = deploy_online(env, controller, horizon).T
    all_means['multi-output_trf'] = cum_means

    env.reset()
    controller = UCBPolicy(
        env,
        const=1.0,
        batch_size=1)
    cum_means = deploy_online(env, controller, horizon).T
    all_means['UCB'] = cum_means
    all_means = {k: np.array(v) for k, v in all_means.items()}

    # Plot rewards of opt and lnr in the same plot
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))  # Create two subplots side by side

    # Plot rewards
    axs[0].plot(np.arange(horizon), all_means['opt'], '-', label='opt', color='black', alpha=1.0)
    axs[0].plot(np.arange(horizon), all_means['trf'], 'o', label='pretrained transformer', color='blue', alpha=0.4)
    axs[0].plot(np.arange(horizon), all_means['multi-output_trf'], 'o', label='multi-output transformer', color='green', alpha=0.4)
    # axs[0].plot(np.arange(horizon), all_means['UCB'], '+', label='UCB', color='red', alpha=1.0)
    axs[0].set_xlabel('Time Steps')
    axs[0].set_ylabel('Rewards')
    axs[0].set_title('Rewards Comparison')
    axs[0].legend()

    # Calculate and plot regrets
    regrets = {k: all_means['opt'] - v for k, v in all_means.items() if k != 'opt'}
    # Calculate and plot cumulative regrets
    cumulative_regrets = {k: np.cumsum(v) for k, v in regrets.items()}
    for k, v in cumulative_regrets.items():
        axs[1].plot(np.arange(horizon), v, '-', label=k)
    axs[1].set_xlabel('Time Steps')
    axs[1].set_ylabel('Cumulative Regret')
    axs[1].set_title('Cumulative Regret Comparison')
    axs[1].legend()



def online(eval_trajs, model, model0, n_eval, horizon, var, evals_filename, save_filename):
    # Dictionary to store means of different policies
    all_means = {}

    envs = []
    for i_eval in range(n_eval):
        traj = eval_trajs[i_eval]
        means = traj['means']

        # Create bandit environment
        env = BanditEnv(means, horizon, var=var)
        if np.all((env.means >= 0.15) & (env.means <= 0.85)):
            envs.append(env)

    vec_env = BanditEnvVec(envs)

    # Optimal policy
    controller = OptPolicy(
        envs,
        batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    all_means['opt'] = cum_means

    # Pretrained transformer policy
    controller = BanditTransformerController(
        model0,
        sample=True,
        batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    all_means['pretrained_trf'] = cum_means

    # Multi-output transformer policy
    controller = BanditMOTransformerController(
        model,
        sample=True,
        batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    all_means['multi-output_trf'] = cum_means

    # UCB policy
    controller = UCBPolicy(
        envs[0],
        const=1.0,
        batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    all_means['UCB1.0'] = cum_means

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
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))

    # Plot suboptimality
    for key in means.keys():
        if key == 'opt':
            ax1.plot(means[key], label=key, linestyle='--',
                     color='black', linewidth=2)
            ax1.fill_between(np.arange(horizon), means[key] - sems[key], means[key] + sems[key], alpha=0.2, color='black')
        else:
            ax1.plot(means[key], label=key)
            ax1.fill_between(np.arange(horizon), means[key] - sems[key], means[key] + sems[key], alpha=0.2)

    ax1.set_yscale('log')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Suboptimality')
    ax1.set_title('Online Evaluation')
    ax1.legend()

    # Plot cumulative regret
    for key in regret_means.keys():
        if key != 'opt':
            ax2.plot(regret_means[key], label=key)
            ax2.fill_between(np.arange(horizon), regret_means[key] - regret_sems[key], regret_means[key] + regret_sems[key], alpha=0.2)

    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('Regret Over Time')
    ax2.legend()

    # plot regret distribution
    for key in regret.keys():
        if key != 'opt' and key != 'UCB1.0':
            ax3.hist(regret[key], bins=20, alpha=0.5, label=key)
            # sns.kdeplot(regret[key], ax=ax3, label=key, shade=True, alpha=0.5)
    ax3.set_xlabel('Regret')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Regret Distribution')
    ax3.legend()


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
