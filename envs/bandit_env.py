import gymnasium as gym
import numpy as np
import torch

from envs.base_env import BaseEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(dim, H, var, type='uniform', style = 'train'):
    if type == 'uniform':
        means = np.random.uniform(0, 1, dim)
        if style == 'eval':
            means = np.random.uniform(0.4, 0.6, dim)
    elif type == 'bernoulli':
        means = np.random.beta(1, 1, dim) # this is related to bayesian prior of bernoulli distribution
    else:
        raise NotImplementedError
    env = BanditEnv(means, H, var=var, type=type)
    return env


class BanditEnv(BaseEnv):
    def __init__(self, means, H, var=0.0, type='uniform'):
        opt_a_index = np.argmax(means)
        self.means = means
        self.opt_a_index = opt_a_index
        self.opt_a = np.zeros(means.shape)
        self.opt_a[opt_a_index] = 1.0
        self.dim = len(means)
        self.observation_space = gym.spaces.Box(low=1, high=1, shape=(1,)) # dummy observation space
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.dim,))
        self.state = np.array([1])
        self.var = var
        self.dx = 1 # state dim
        self.du = self.dim # action dim
        self.type = type
        self.H = H
        self.sign = 1 # dummy variable

    def get_arm_value(self, u):
        return np.sum(self.means * u)

    def reset(self):
        self.current_step = 0
        return self.state

    def transit(self, x, u):
        a = np.argmax(u)
        if self.type == 'uniform':
            r = self.means[a] + np.random.normal(0, self.var)
        elif self.type == 'bernoulli':
            r = np.random.binomial(1, self.means[a])
        else:
            raise NotImplementedError
        return self.state.copy(), r

    def step(self, action):
        # if self.current_step >= self.H + 1:
        #     raise ValueError("Episode has already ended")

        _, r = self.transit(self.state, action)
        self.current_step += 1
        done = (self.current_step >= self.sign)

        return self.state.copy(), r, done, {}

    def deploy_eval(self, ctrl):
        # No variance during evaluation
        tmp = self.var
        self.var = 0.0
        res = self.deploy(ctrl)
        self.var = tmp
        return res
    
    def deploy(self, ctrl):
        ob = self.reset()
        obs = []
        acts = []
        next_obs = []
        rews = []
        cs = []
        done = False

        while not done:
            act, c = ctrl.act(ob)

            obs.append(ob)
            acts.append(act)
            cs.append(c)

            ob, rew, done, _ = self.step(act)

            rews.append(rew)
            next_obs.append(ob)

        obs = np.array(obs)
        acts = np.array(acts)
        next_obs = np.array(next_obs)
        rews = np.array(rews)
        cs = np.array(cs)

        return obs, acts, next_obs, rews, cs


class BanditEnvVec(BaseEnv):
    """
    Vectorized bandit environment.
    """
    def __init__(self, envs):
        self._envs = envs
        self._num_envs = len(envs)
        self.dx = envs[0].dx
        self.du = envs[0].du

    def reset(self):
        return [env.reset() for env in self._envs]

    def step(self, actions):
        next_obs, rews, dones = [], [], []
        for action, env in zip(actions, self._envs):
            next_ob, rew, done, _ = env.step(action)
            next_obs.append(next_ob)
            rews.append(rew)
            dones.append(done)
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
        x = self.reset()
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
        values = [np.sum(env.means * u) for env, u in zip(self._envs, us)]
        return np.array(values)

class HardBanditEnv(BanditEnv):
    def __init__(self, means, H, var=0.0):
        super().__init__(means, H, var)
        self.freeze_index = np.argmax(means)
        self.optimal_picks = 0

    def transit(self, x, u):
        a = np.argmax(u)
        a_sec = np.argmax(u[1:])
        if a == self.freeze_index:
            # print(self.optimal_picks)
            self.optimal_picks += 1
            if self.optimal_picks >= 30:
                r = self.means[a] + np.random.normal(0, self.var)
            else:
                r = self.means[a_sec]-0.05 + np.random.normal(0, self.var)
        else:
            r = self.means[a] + np.random.normal(0, self.var)
        return self.state.copy(), r
    
    def reset_optimal_picks(self):
        self.optimal_picks = 0

class HardBanditEnvVec(BanditEnvVec):
    def __init__(self, envs):
        super().__init__(envs)
        self._freeze_indices = [env.freeze_index for env in envs]
        self._optimal_picks = [0 for _ in envs]

    def step(self, actions):
        next_obs, rews, dones = [], [], []
        for action, env in zip(actions, self._envs):
            next_ob, rew, done, _ = env.step(action)
            next_obs.append(next_ob)
            rews.append(rew)
            dones.append(done)
        return next_obs, rews, dones, {}

    def deploy(self, ctrl):
        x = self.reset()
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

    def reset_optimal_picks(self):
        for env in self._envs:
            env.reset_optimal_picks()

    def get_arm_value(self, us):
        values = [np.sum(env.means * u) for env, u in zip(self._envs, us)]
        return np.array(values)

    def get_optimal_picks(self):
        return [env.optimal_picks for env in self._envs]

    def get_freeze_indices(self):
        return self._freeze_indices