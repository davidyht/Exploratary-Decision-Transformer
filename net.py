import torch
import torch.nn as nn
import transformers
transformers.set_seed(0)
from transformers import GPT2Config, GPT2Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Context_extractor(nn.Module):
    def __init__(self, config):
        super(Context_extractor, self).__init__()

        self.config = config
        self.test = config['test']
        self.horizon = self.config['horizon']
        self.n_embd = self.config['n_embd']
        self.n_layer = self.config['n_layer']
        self.n_head = self.config['n_head']
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']
        self.dropout = self.config['dropout']

        config = GPT2Config(
            n_positions=4 * (1 + self.horizon),
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(config)

        self.embed_transition = nn.Linear(
            2 * self.state_dim + self.action_dim + 1, self.n_embd)
        self.pred_context = nn.Linear(self.n_embd, self.state_dim * self.action_dim)

    def forward(self, x):
        query_states = x['query_states'][:, None, :]
        zeros = x['zeros'][:, None, :]
        batch_size = query_states.shape[0]

        state_seq = torch.cat([query_states, x['context_states'][:, :, :]], dim=1)
        action_seq = torch.cat(
            [zeros[:, :, :self.action_dim], x['context_actions'][:, :, :]], dim=1)

        next_state_seq = torch.cat(
            [zeros[:, :, :self.state_dim], x['context_next_states'][:, :, :]], dim=1)
        reward_seq = torch.cat([zeros[:, :, :1], x['context_rewards'][:, :, :]], dim=1)

        seq = torch.cat(
            [state_seq, action_seq, next_state_seq, reward_seq], dim=2)
        stacked_inputs = self.embed_transition(seq)
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        preds = self.pred_context(transformer_outputs['last_hidden_state'])

        if self.test:
            return preds[:, -1, :]
        return preds[:, 1:, :] 

class pretrain_transformer(nn.Module):
    def __init__(self, config):
        super(pretrain_transformer, self).__init__()

        self.config = config
        self.test = config['test']
        self.horizon = self.config['horizon']
        self.n_embd = self.config['n_embd']
        self.n_layer = self.config['n_layer']
        self.n_head = self.config['n_head']
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']
        self.dropout = self.config['dropout']

        config = GPT2Config(
            n_positions=500,
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(config)

        self.embed_transition = nn.Linear(
            2 * self.state_dim + self.action_dim + 1, self.n_embd)
        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)
        self.context_extractor = nn.Linear(self.n_embd, self.action_dim)

    def forward(self, x):
        query_states = x['query_states'][:, None, :]
        zeros = x['zeros'][:, None, :]
        batch_size = query_states.shape[0]

        state_seq = torch.cat([query_states, x['context_states'][:, :, :]], dim=1)
        action_seq = torch.cat(
            [zeros[:, :, :self.action_dim], x['context_actions'][:, :, :]], dim=1)

        next_state_seq = torch.cat(
            [zeros[:, :, :self.state_dim], x['context_next_states'][:, :, :]], dim=1)
        reward_seq = torch.cat([zeros[:, :, :1], x['context_rewards'][:, :, :]], dim=1)

        seq = torch.cat(
            [state_seq, action_seq, next_state_seq, reward_seq], dim=2)
        stacked_inputs = self.embed_transition(seq)
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        preds = self.pred_actions(transformer_outputs['last_hidden_state'])
        context = self.context_extractor(transformer_outputs['last_hidden_state'])

        if self.test:
            return preds[:, -1, :]
        return preds[:, 1:, :] 

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        self.config = config
        self.test = config['test']
        self.horizon = self.config['horizon']
        self.n_embd = self.config['n_embd']
        self.n_layer = self.config['n_layer']
        self.n_head = self.config['n_head']
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']
        self.dropout = self.config['dropout']

        config = GPT2Config(
            n_positions=4 * (1 + self.horizon),
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(config)

        self.embed_transition = nn.Linear(
            2 * self.state_dim + (self.state_dim + 1) * self.action_dim + 1, self.n_embd)
        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)
        self.context_extractor = nn.Linear(self.n_embd, self.state_dim * self.action_dim)

    def forward(self, x):
        query_states = x['query_states'][:, None, :]
        zeros = x['zeros'][:, None, :]

        state_seq = torch.cat([query_states, x['context_states'][:, :, :]], dim=1)
        action_seq = torch.cat(
            [zeros[:, :, :self.action_dim], x['context_actions'][:, :, :]], dim=1)

        next_state_seq = torch.cat(
            [zeros[:, :, :self.state_dim], x['context_next_states'][:, :, :]], dim=1)
        reward_seq = torch.cat([zeros[:, :, :1], x['context_rewards'][:, :, :]], dim=1)
        context_seq = torch.cat([zeros[:, :, :self.state_dim * self.action_dim], x['context'][:, :, :]], dim=1)
    
        seq = torch.cat(
            [state_seq, action_seq, next_state_seq, reward_seq, context_seq], dim=2)
        stacked_inputs = self.embed_transition(seq)
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        preds = self.pred_actions(transformer_outputs['last_hidden_state'])
        if self.test:
            return preds[:, -1, :]
        return preds[:, 1:, :]
