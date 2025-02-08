import numpy as np
import torch
from ctrls.ctrl_bandit import Controller

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class OptCgPolicy(Controller):
    def __init__(self, env, batch_size):
        super().__init__()
        self.env = env
        self.batch_size = batch_size

    def reset(self):
        return

    def act(self, x):
        opt_a = self.env.get_opt_arm()
        context = np.zeros(self.env.dim)
        
        return opt_a, context


    def act_numpy_vec(self, x):
        opt_as = [ env.get_opt_arm() for env in self.env ]
        context = np.zeros((self.batch_size, self.env[0].dim))
        for idx in range(len(self.env)):
            self.env[idx].current_step += 1

        return np.stack(opt_as, axis=0), context
    
class SlidingWindow(Controller):
    def __init__(self, env, window_len = 30, const = 1.0, batch_size=1):
        super().__init__()
        self.env = env
        self.window_len = window_len
        self.const = const
        self.batch_size = batch_size
        self.history = []
        
    def reset(self):
        self.history = []
        
    def act(self, x):
        actions = self.batch['context_actions'].cpu().detach().numpy()[0][-self.window_len:]
        rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()[-self.window_len:]
        context = np.zeros(self.env.dim)

        b = np.zeros(self.env.dim)
        counts = np.zeros(self.env.dim)
        for i in range(len(actions)):
            c = np.argmax(actions[i])
            b[c] += rewards[i]
            counts[c] += 1

        b_mean = b / np.maximum(1, counts)

        # compute the square root of the counts but clip so it's at least one
        bons = self.const / np.maximum(1, np.sqrt(counts))
        bounds = b_mean + bons

        i = np.argmax(bounds)
        a = np.zeros(self.env.dim)
        a[i] = 1.0
        self.a = a
        return self.a, context
    
    def act_numpy_vec(self, x):
        actions = self.batch['context_actions'][:,-self.window_len:,:]
        rewards = self.batch['context_rewards'][:,-self.window_len:,:]
        context = np.zeros((self.batch_size, self.env.dim))

        b = np.zeros((self.batch_size, self.env.dim))
        counts = np.zeros((self.batch_size, self.env.dim))
        action_indices = np.argmax(actions, axis=-1)
        for idx in range(self.batch_size):
            actions_idx = action_indices[idx]
            rewards_idx = rewards[idx]
            for c in range(self.env.dim):
                arm_rewards = rewards_idx[actions_idx == c]
                b[idx, c] = np.sum(arm_rewards)
                counts[idx, c] = len(arm_rewards)

        b_mean = b / np.maximum(1, counts)

        # compute the square root of the counts but clip so it's at least one
        bons = self.const / np.maximum(1, np.sqrt(counts))
        bounds = b_mean + bons

        i = np.argmax(bounds, axis=-1)
        j = np.argmin(counts, axis=-1)
        mask = np.zeros(self.batch_size, dtype=bool)
        for idx in range(self.batch_size):
            if counts[idx, j[idx]] == 0:
                mask[idx] = True
        i[mask] = j[mask]

        a = np.zeros((self.batch_size, self.env.dim))
        a[np.arange(self.batch_size), i] = 1.0
        self.a = a
        return self.a, context
    
class Exp3(Controller):
    def __init__(self, env, eta = 0.5, batch_size = 1) -> None:
        super().__init__()
        self.env = env
        self.eta = eta
        self.batch_size = batch_size
        self.dim = self.env.dim
        self.logweight = np.zeros((self.batch_size, self.dim))

    def get_probabilities(self):

        self.weights = np.exp(self.eta * self.logweight) 
        total_weight = np.sum(self.weights, axis = 1)
        p = np.zeros((self.batch_size, self.dim))
        for i in range(self.batch_size):
            p[i, :] = self.weights[i,:] / total_weight[i]

        return p
        
    def reset(self):
        return
    
    def act(self, x = [1]):
        a = np.zeros((self.batch_size, self.dim))

        for i in range(self.batch_size):
            idx = np.random.choice(self.dim, p=self.get_probabilities()[i,:])
            a[i,idx] = 1

        self.update(x)
        
        return a 
    
    def update(self, x):

        actions = self.batch['context_actions'].cpu().detach().numpy()
        rewards = self.batch['context_rewards'].cpu().detach().numpy()

        self.logweight += 1

        if actions.shape[1] > 0:
            a = actions[:, -1, :]
            reward = rewards[:, -1]
            for idx in range(self.batch_size):
                a_idx = np.argmax(a[idx,:])
                self.logweight[idx, a_idx] -=  (1 - reward[idx]) / self.get_probabilities()[idx, a_idx]
              





        
