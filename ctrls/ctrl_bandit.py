import numpy as np
import scipy
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Controller:
    def set_batch(self, batch):
        self.batch = batch

    def set_batch_numpy_vec(self, batch):
        self.set_batch(batch)

    def set_env(self, env):
        self.env = env


class OptPolicy(Controller):
    def __init__(self, env, batch_size=1):
        super().__init__()
        self.env = env
        self.batch_size = batch_size

    def reset(self):
        return

    def act(self, x):
        c = np.zeros(self.env.dim)
        return self.env.opt_a, c


    def act_numpy_vec(self, x):
        opt_as = [ env.opt_a for env in self.env ]
        c = np.zeros((self.batch_size, self.env[0].dim))

        return np.stack(opt_as, axis=0), c
        # return np.tile(self.env.opt_a, (self.batch_size, 1))


class GreedyOptPolicy(Controller):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def reset(self):
        return

    def act(self, x):
        rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()
        i = np.argmax(rewards)
        a = self.batch['context_actions'].cpu().detach().numpy()[0][i]
        self.a = a
        return self.a


class UCBPolicy(Controller):
    def __init__(self, env, const=1.0, batch_size=1):
        super().__init__()
        self.env = env
        self.const = const
        self.batch_size = batch_size

    def reset(self):
        return

    def act(self, x):
        actions = self.batch['context_actions'].cpu().detach().numpy()[0]
        rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()
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
        actions = self.batch['context_actions']
        rewards = self.batch['context_rewards']
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


class BanditMOTransformerController(Controller):
    def __init__(self, model, sample=True,  batch_size=1):
        self.model = model
        self.du = model[0].config['action_dim']
        self.dx = model[0].config['state_dim']
        self.H = model[0].horizon
        self.sample = sample
        self.batch_size = batch_size
        self.zeros = torch.zeros(batch_size, 2 * self.dx + 3 * self.du + 1).float().to(device)

    def set_env(self, env):
        return

    def set_batch_numpy_vec(self, batch):
        # Convert each element of the batch to a torch tensor
        new_batch = {}
        for key in batch.keys():
            new_batch[key] = torch.tensor(batch[key]).float().to(device)
        self.set_batch(new_batch)

    def act(self, x):
        self.batch['zeros'] = self.zeros

        states = torch.tensor(x)[None, :].float().to(device)
        self.batch['query_states'] = states
        c = self.model[1](self.batch)
        c = c.cpu().detach().numpy()
        # c = np.exp(c) / np.sum(np.exp(c))

        a = self.model[0](self.batch)
        a = a.cpu().detach().numpy()

        if self.sample:
            probs = scipy.special.softmax(a)
            i = np.random.choice(np.arange(self.du), p=probs.squeeze()) 
        else:
            i = np.argmax(a)

        a = np.zeros(self.du)
        a[i] = 1.0
        return a, c

    def act_numpy_vec(self, x):
        self.batch['zeros'] = self.zeros

        states = torch.tensor(np.array(x))
        if self.batch_size == 1:
            states = states[None,:]
        states = states.float().to(device)
        self.batch['query_states'] = states

        # context_actions = self.batch['context_actions'].cpu().detach().numpy()
        # context_rewards = self.batch['context_rewards'].cpu().detach().numpy().squeeze()

        # c = np.zeros((self.batch_size, 2 * self.du))
        # c[:, :self.du] = np.sum(context_actions, axis = 1)

        # num = c[:, :self.du]
        # rew_sum = np.expand_dims(context_rewards, axis=-1) * context_actions
        # rew_sum = np.sum(rew_sum, axis=1)
        # c[:, self.du:] = rew_sum / (num + 1e-6)

        c = self.model[1](self.batch)
        c = c.cpu().detach().numpy()
        # for i in range(self.batch_size):
        #     c[i, :] = np.exp(c[i, :]) / np.sum(np.exp(c[i, :]))

        a = self.model[0](self.batch)
        a = a.cpu().detach().numpy()

        if self.batch_size == 1:
            a = a[0]

        if self.sample:
            probs = scipy.special.softmax(a, axis=-1)
            action_indices = np.array([np.random.choice(np.arange(self.du), p=p) for p in probs])
        else:
            action_indices = np.argmax(a, axis=-1)

        actions = np.zeros((self.batch_size, self.du))
        actions[np.arange(self.batch_size), action_indices] = 1.0
        return actions, c
    
class BanditTransformerController(Controller):
    def __init__(self, model, sample=True,  batch_size=1):
        self.model = model
        self.du = model.config['action_dim']
        self.dx = model.config['state_dim']
        self.H = model.horizon
        self.sample = sample
        self.batch_size = batch_size
        self.zeros = torch.zeros(batch_size, self.dx**2 + self.du + 1).float().to(device)

    def set_env(self, env):
        return

    def set_batch_numpy_vec(self, batch):
        # Convert each element of the batch to a torch tensor
        new_batch = {}
        for key in batch.keys():
            new_batch[key] = torch.tensor(batch[key]).float().to(device)
        self.set_batch(new_batch)

    def act(self, x):
        self.batch['zeros'] = self.zeros

        states = torch.tensor(x)[None, :].float().to(device)
        self.batch['query_states'] = states

        a = self.model(self.batch)
        a = a.cpu().detach().numpy()[0]
        c = np.zeros(self.du)

        if self.sample:
            probs = scipy.special.softmax(a)
            i = np.random.choice(np.arange(self.du), p=probs)
        else:
            i = np.argmax(a)

        a = np.zeros(self.du)
        a[i] = 1.0
        return a, c

    def act_numpy_vec(self, x):
        self.batch['zeros'] = self.zeros

        states = torch.tensor(np.array(x))
        if self.batch_size == 1:
            states = states[None,:]
        states = states.float().to(device)
        self.batch['query_states'] = states

        a = self.model(self.batch)
        a = a.cpu().detach().numpy()
        c = np.zeros((self.batch_size, self.du))
        if self.batch_size == 1:
            a = a[0]

        if self.sample:
            probs = scipy.special.softmax(a, axis=-1)
            action_indices = np.array([np.random.choice(np.arange(self.du), p=p) for p in probs])
        else:
            action_indices = np.argmax(a, axis=-1)

        actions = np.zeros((self.batch_size, self.du))
        actions[np.arange(self.batch_size), action_indices] = 1.0
        return actions, c
    


class TransformerContextController(Controller):
    def __init__(self, model, batch_size=1):
        self.model = model
        self.du = model.config['action_dim']
        self.dx = model.config['state_dim']
        self.H = model.horizon
        self.batch_size = batch_size
        self.zeros = torch.zeros(batch_size, self.dx**2 + self.du + 1).float().to(device)

    def set_env(self, env):
        return

    def set_batch_numpy_vec(self, batch):
        # Convert each element of the batch to a torch tensor
        new_batch = {}
        for key in batch.keys():
            new_batch[key] = torch.tensor(batch[key]).float().to(device)
        self.set_batch(new_batch)

    def act(self, x):
        self.batch['zeros'] = self.zeros

        states = torch.tensor(x)[None, :].float().to(device)
        self.batch['query_states'] = states

        context = self.model(self.batch)
        context = context.cpu().detach().numpy()[0]
        action_index = np.argmax(context)

        a = np.zeros(self.du)
        a[action_index] = 1.0
        return a, context

    def act_numpy_vec(self, x):
        self.batch['zeros'] = self.zeros

        states = torch.tensor(np.array(x))
        if self.batch_size == 1:
            states = states[None, :]
        states = states.float().to(device)
        self.batch['query_states'] = states

        context = self.model(self.batch)
        context = context.cpu().detach().numpy()
        action_indices = np.argmax(context, axis=-1)

        actions = np.zeros((self.batch_size, self.du))
        actions[np.arange(self.batch_size), action_indices] = 1.0
        return actions, context