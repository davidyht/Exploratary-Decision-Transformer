import gymnasium as gym
import torch
import numpy as np
from envs.base_env import BaseEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample(dim, H, var, scl = None, type='uniform', cg_time_type='random'):
    means = np.zeros((2, dim))
    if scl is None:
        if type == 'uniform':
            means[0] = np.random.uniform(0, 1, dim)
            means[1] = np.random.uniform(0, 1, dim)
        elif type == 'bernoulli':
            means[0] = np.random.beta(1, 1, dim)
            means[1] = np.random.beta(1, 1, dim)
        else:
            raise NotImplementedError
    
    if cg_time_type == 'fixed':
        cg_time = H // 2
    elif cg_time_type == 'random':
        cg_time = np.random.randint(1, H)
    else:
        raise NotImplementedError

    env = CgbanditEnv(means, H = H, cg_time = cg_time, var=var, type=type)
    return env

class CgbanditEnv(BaseEnv):
    """
    Change-Point Bandit Environment.

    This environment simulates a bandit problem with a change-point. The means of the bandit arms can change at a specific time step.

    Args:
        pre_means (ndarray): The means of the bandit arms before the change-point.
        post_means (ndarray): The means of the bandit arms after the change-point.
        cg_time (int): The time step at which the change-point occurs.
        H (int, optional): The time horizon of the environment. Defaults to 1000.
        var (float, optional): The variance of the rewards. Defaults to 0.0.
        type (str, optional): The type of reward distribution. Can be 'uniform' or 'bernoulli'. Defaults to 'uniform'.

    Attributes:
        pre_opta_index (int): The index of the optimal arm before the change-point.
        post_opta_index (int): The index of the optimal arm after the change-point.
        pre_means (ndarray): The means of the bandit arms before the change-point.
        post_means (ndarray): The means of the bandit arms after the change-point.
        means (ndarray): The current means of the bandit arms.
        pre_opt_a (ndarray): The one-hot encoding of the optimal arm before the change-point.
        post_opt_a (ndarray): The one-hot encoding of the optimal arm after the change-point.
        dim (int): The number of bandit arms.
        observation_space (gym.spaces.Box): The observation space of the environment.
        action_space (gym.spaces.Box): The action space of the environment.
        state (ndarray): The current state of the environment.
        cg_time (int): The time step at which the change-point occurs.
        var (float): The variance of the rewards.
        type (str): The type of reward distribution.
        dx (int): The dimension of the state space.
        du (int): The dimension of the action space.
        topk (bool): Whether to use top-k action selection.
        H (int): The time horizon of the environment.

    Methods:
        get_arm_value(u): Returns the value of the selected arm given the action.
        reset(): Resets the environment to the initial state.
        transit(u): Transitions the environment to the next state given the action.
        step(action): Takes a step in the environment given the action.
        deploy_eval(ctrl): Deploys the controller for evaluation.
    """

    def __init__(self, means, cg_time, H, var=0.0, type='uniform'):
        assert cg_time in range(H + 2), \
            'change distribution time should be within time horizon'
        assert means.shape[0] == 2, \
            'means should store pre_means and post_means'
        self.means = means
        self.pre_means = means[0,:]
        self.post_means = means[1,:]
        self.pre_opta_index = np.argmax(self.pre_means)
        self.post_opta_index = np.argmax(self.post_means)
        self.dim = means.shape[1]
        self.pre_opt_a = np.zeros(self.dim)
        self.pre_opt_a[self.pre_opta_index] = 1.0
        self.post_opt_a = np.zeros(self.dim)
        self.post_opt_a[self.post_opta_index] = 1.0
        self.observation_space = gym.spaces.Box(low=1, high=1, shape=(1,))  
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.dim,))
        self.state = np.array([1])
        self.cg_time = cg_time
        self.var = var
        self.type = type
        self.dx = 1
        self.du = self.dim
        self.topk = False
        self.H = H
        self.sign = 1

        self.current_step = 0

    def get_arm_value(self, u):
        """
        Returns the value of the selected arm given the action.

        Args:
            u (ndarray): The action.

        Returns:
            float: The value of the selected arm.
        """
        if self.current_step >= self.cg_time + 1:
            return np.sum(self.post_means * u)
        else:
            return np.sum(self.pre_means * u)

    def reset(self):
        """
        Resets the environment to the initial state.

        Returns:
            ndarray: The initial state of the environment.
        """
        self.current_step = 0
        return self.state

    def transit(self, u):
        """
        Transitions the environment to the next state given the action.

        Args:
            u (ndarray): The action.

        Returns:
            tuple: A tuple containing the next state and the reward.
        """
        a = np.argmax(u)
        if self.current_step >= self.cg_time:
            n = 1
        else:
            n = 0

        if self.type == 'uniform':
            r = self.means[n,a] + np.random.normal(0, self.var)

        elif self.type == 'bernoulli':
                r = np.random.binomial(1, self.means[n,a])
    
        else:
            raise NotImplementedError

        return self.state.copy(), r

    def step(self, action):
        """
        Takes a step in the environment given the action.

        Args:
            action (ndarray): The action.

        Returns:
            tuple: A tuple containing the next state, the reward, a flag indicating if the episode is done, and additional information.
        """
        if self.current_step >= self.H + 1:
            raise ValueError("Episode has already ended")
        print(self.current_step, end = '\r')
        _, r = self.transit(action)
        self.current_step += 1
        done = (self.current_step >= self.sign)

        return self.state.copy(), r, done, {}
    
    def deploy(self, ctrl):
        x = self.state.copy()
        xs = []
        xps = []
        us = []
        rs = []
        cs = []
        done = False

        while not done:
            act, c = ctrl.act(x)

            xs.append(x)
            cs.append(c)
            us.append(act)

            x, r, done, _ = self.step(act)
            xps.append(x)
            rs.append(r)

        xs = np.array(xs)
        us = np.array(us)
        xps = np.array(xps)
        rs = np.array(rs)
        cs = np.array(cs)

        return xs, us, xps, rs, cs


    def deploy_eval(self, ctrl):
        """
        Deploys the controller for evaluation.

        Args:
            ctrl: The controller.

        Returns:
            Any: The result of the deployment.
        """
        # No variance during evaluation
        tmp = self.var
        self.var = 0.0
        res = self.deploy(ctrl)
        self.var = tmp
        return res
    
    def get_opt_arm(self):
        if self.current_step >= self.cg_time:
            return self.post_opt_a
        else:
            return self.pre_opt_a


class CgbanditEnvVec(BaseEnv):
    """
    Vectorized bandit environment.
    """

    def __init__(self, envs):
        self._envs = envs
        self._num_envs = len(envs)
        self.dx = envs[0].dx
        self.du = envs[0].du
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return [env.reset() for env in self._envs]

    def step(self, actions):
        next_obs, rews, dones = [], [], []
        for action, env in zip(actions, self._envs):
            next_ob, rew, done, _ = env.step(action)
            env.current_step = self.current_step + 1
            next_obs.append(next_ob)
            rews.append(rew)
            dones.append(done)
        self.current_step += 1
        return next_obs, rews, dones, {}

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def envs(self):
        return self._envs

    def deploy_eval(self, ctrl):
        # No variance during evaluation
        tmp = [env.var for env in self._envs]
        for env in self._envs:
            env.var = 0.0
        res = self.deploy(ctrl)
        for env, var in zip(self._envs, tmp):
            env.var = var
        return res

    def deploy(self, ctrl):
        x = [env.state.copy() for env in self._envs]
        xs = []
        xps = []
        us = []
        rs = []
        cs = []
        done = False

        while not done:
            u, c = ctrl.act_numpy_vec(x)

            xs.append(x)
            us.append(u)
            cs.append(c)

            x, r, done, _ = self.step(u)
            done = all(done)

            rs.append(r)
            xps.append(x)

        xs = np.concatenate(xs)
        us = np.concatenate(us)
        xps = np.concatenate(xps)
        rs = np.concatenate(rs)
        cs = np.concatenate(cs)
        return xs, us, xps, rs, cs

    def get_arm_value(self, us):
        #values = [np.sum(env.means * u) for env, u in zip(self._envs, us)]
        #return np.array(values)
        values = []
        for env, u in zip(self._envs, us):
            values.append(env.get_arm_value(u))
        values = np.array(values)
        return np.array(values)


