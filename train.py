import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)  # or 'forkserver'

from itertools import chain
import argparse
import os
import time

import matplotlib.pyplot as plt
import torch

import numpy as np
import common_args
import random
from dataset import Dataset
from net import Transformer, Context_extractor, pretrain_transformer
from utils import (
    build_data_filename,
    build_model_filename,
    convert_to_tensor
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def printw(string, log_filename):
    """
    A drop-in replacement for print that also writes to a log file.
    """
    # Use the standard print function to print to the console
    print(string)
    # Write the same output to the log file
    with open(log_filename, 'a') as f:
        f.write(string + '\n')

def load_checkpoints(model, filename):
    # Check if there are existing model checkpoints
    checkpoint_files = [f for f in os.listdir('models') if f.startswith(f"{filename}_") and 'epoch' in f]
    if checkpoint_files:
        # Sort the checkpoint files based on the epoch number
        checkpoint_files.sort(key=lambda x: int(x.split('_epoch')[1].split('.pt')[0]))
        
        # Get the last checkpoint file
        last_checkpoint_file = checkpoint_files[-1]
        
        # Extract the epoch number from the checkpoint file name
        last_epoch = int(last_checkpoint_file.split('_epoch')[1].split('.pt')[0])
        
        # Load the model checkpoint
        model.load_state_dict(torch.load(os.path.join('models', last_checkpoint_file)))
        
        # Update the starting epoch
        start_epoch = last_epoch + 1
    else:
        # No existing model checkpoints, start from epoch 0
        start_epoch = 0
    return start_epoch, model

def load_args():
    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    common_args.add_train_args(parser)
    parser.add_argument('--seed', type=int, default=0)
    args = vars(parser.parse_args())
    print("Args: ", args)
    env = args['env']
    n_envs = args['envs']
    n_hists = args['hists']
    n_samples = args['samples']
    horizon = args['H']
    dim = args['dim']
    state_dim = 1
    action_dim = dim
    n_embd = args['embd']
    n_head = args['head']
    n_layer = args['layer']
    context_len = args['context_len']
    lr = args['lr']
    shuffle = args['shuffle']
    dropout = args['dropout']
    var = args['var']
    cov = args['cov']
    num_epochs = args['num_epochs']
    w = args['exploration_rate']
    seed = args['seed']
    tmp_seed = seed
    if seed == -1:
        tmp_seed = 0
    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(tmp_seed)
        torch.cuda.manual_seed_all(tmp_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(tmp_seed)
    random.seed(tmp_seed)
    
    dataset_config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
    }
    dataset_config.update({'var': var, 'cov': cov, 'type': 'uniform'})

    model_config = {
        'shuffle': shuffle,
        'lr': lr,
        'dropout': dropout,
        'n_embd': n_embd,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_envs': n_envs,
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
        'seed': seed,
        'context_len': context_len,
    }
    model_config.update({'var': var, 'cov': cov})
    config = {
        'horizon': horizon,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'shuffle': shuffle,
        'dropout': dropout,
        'test': False,
        'store_gpu': True,
        'context_len': context_len,
    }
    params = {
        'batch_size': 100,
        'shuffle': True,
        'drop_last': True,
    }

    return (
        config, model_config, dataset_config, params, env, num_epochs, w, n_envs, lr, horizon, dim, action_dim
    )


def train():
    if __name__ == '__main__':
        if not os.path.exists('figs/loss'):
            os.makedirs('figs/loss', exist_ok=True)
        if not os.path.exists('models'):
            os.makedirs('models', exist_ok=True)
        
        config, model_config, dataset_config, params, env, num_epochs, w, n_envs, lr, horizon, dim, action_dim = load_args()

        model_ctx = Context_extractor(config).to(device)
        model_act = Transformer(config).to(device)
        
        filename = build_model_filename(env, model_config)
        log_filename = f'figs/loss/{filename}_logs.txt'

        with open(log_filename, 'w') as f:
            pass

        path_train = build_data_filename(env, n_envs, dataset_config, mode=0)
        path_test = build_data_filename(env, n_envs, dataset_config, mode=1)
        train_dataset = Dataset(path = path_train, config = config)
        test_dataset = Dataset(path = path_test, config = config)
        train_loader0 = torch.utils.data.DataLoader(train_dataset, **params)
        test_loader = torch.utils.data.DataLoader(test_dataset, **params)
        optimizer_act = torch.optim.AdamW(model_act.parameters(), lr=lr, weight_decay=1e-4)
        optimizer_ctx = torch.optim.AdamW(model_ctx.parameters(), lr=lr, weight_decay=1e-4)


        loss_fn1 = torch.nn.MSELoss(reduction='sum')
        loss_fn2 = torch.nn.CrossEntropyLoss(reduction='sum')
        test_loss = []
        train_loss = []
        test_act_loss = []
        test_context_loss = []
        train_act_loss = []
        train_context_loss = []
        printw("Num train batches: " + str(len(train_loader0)), log_filename)
        printw("Num test batches: " + str(len(test_loader)), log_filename)
        filename_test_act = f'{filename}_model_act'
        filename_test_ctx = f'{filename}_model_ctx'

        start_epoch, model_act = load_checkpoints(model_act, filename_test_act)
        start_epoch, model_ctx = load_checkpoints(model_ctx, filename_test_ctx)
        filename = f'{filename}'

        if start_epoch == 0:
            printw("Starting from scratch.", log_filename)
        else:
            printw(f"Starting from epoch {start_epoch}", log_filename)
        
        train_loader = train_loader0

        for epoch in range(start_epoch, num_epochs):
            # EVALUATION
            printw(f"Epoch: {epoch + 1}", log_filename)
            start_time = time.time()
            with torch.no_grad():
                epoch_test_loss = 0.0
                epoch_test_act_loss = 0.0
                epoch_test_context_loss = 0.0
                for i, batch in enumerate(test_loader):
                    print(f"Batch {i} of {len(test_loader)}", end='\r')

                    batch = {k: v.to(device) for k, v in batch.items()}
                    means = batch['means'].clone().detach()
                    true_context = batch['context'].clone().detach()
                    context_copy = batch['context'].clone().detach()
                    true_actions = torch.zeros((params['batch_size'], horizon, action_dim)).to(device)
                    pre_opt_a = batch['optimal_actions'][:, 0:dim].clone().detach()
                    post_opt_a = batch['optimal_actions'][:, dim:].clone().detach()
                    cg_time = batch['cg_times'].clone().detach().squeeze().tolist()
                    cg_time = [int(t) for t in cg_time]

                    for i in range(params['batch_size']):
                        for idx in range(cg_time[i]):
                            if env == 'bandit':
                                true_actions[i, idx, :] = pre_opt_a[i, :]
                                true_context[i, idx, :] = means[i, :]
                            elif env == 'cgbandit':
                                true_actions[i, idx, :] = pre_opt_a[i, :]
                                true_context[i, idx, :] = means[i, 0, :]
                            
                        for idx in range(cg_time[i], horizon):
                            if env == 'bandit':
                                true_actions[i, idx, :] = post_opt_a[i, :]
                                true_context[i, idx, :] = means[i, :]
                            elif env == 'cgbandit': 
                                true_actions[i, idx, :] = post_opt_a[i, :]
                                true_context[i, idx, :] = means[i, 1, :]

                    context_pred = model_ctx(batch)
                    batch['context'] = context_pred.detach()
                    pred_actions = model_act(batch)
                    batch['context'] = context_copy
                    
                    true_actions = true_actions.reshape(-1, action_dim)
                    true_context = true_context.reshape(-1, action_dim)
                    pred_actions = pred_actions.reshape(-1, action_dim)
                    context_pred = context_pred.reshape(-1, action_dim)
                    loss_act = loss_fn2(pred_actions, true_actions)
                    loss_ctx = loss_fn1(context_pred, true_context)
                    loss = loss_act + loss_ctx
                    epoch_test_loss += loss.item() / horizon
                    epoch_test_act_loss += loss_act.item() / horizon
                    epoch_test_context_loss += loss_ctx.item() / horizon
            test_loss.append(epoch_test_loss / len(test_dataset))
            test_act_loss.append(epoch_test_act_loss / len(test_dataset))
            test_context_loss.append(epoch_test_context_loss / len(test_dataset))
            end_time = time.time()
            printw(f"\tTest loss: {test_loss[-1]}", log_filename)
            printw(f"\t Test Action loss: {test_act_loss[-1]}", log_filename)
            printw(f"\t Test Context loss: {test_context_loss[-1]}", log_filename)
            printw(f"\tEval time: {end_time - start_time}", log_filename)
            # TRAINING
            epoch_train_loss = 0.0
            epoch_train_act_loss = 0.0
            epoch_train_context_loss = 0.0
            start_time = time.time()

            
            for i, batch in enumerate(train_loader):
                print(f"Batch {i} of {len(train_loader)}", end='\r')
                batch = {k: v.to(device) for k, v in batch.items()}

                true_actions = torch.zeros((params['batch_size'], horizon, action_dim)).to(device)
                true_context = batch['context'].clone().detach()
                context_copy = batch['context'].clone().detach()
                means = batch['means']
                pre_opt_a = batch['optimal_actions'][:, :action_dim]  # of size (batch_size, action_dim)
                post_opt_a = batch['optimal_actions'][:, action_dim:]  # of size (batch_size, action_dim)
                cg_time = batch['cg_times']  # of size (batch_size, 1)
                cg_time = cg_time.squeeze().tolist()
                cg_time = [int(t) for t in cg_time]  # turn into iterable list

                for i in range(params['batch_size']):
                    for idx in range(cg_time[i]):
                        if env == 'bandit':
                            true_actions[i, idx, :] = pre_opt_a[i, :]
                            true_context[i, idx, :] = means[i, :]
                        elif env == 'cgbandit':
                            true_actions[i, idx, :] = pre_opt_a[i, :]
                            true_context[i, idx, :] = means[i, 0, :]
                        
                    for idx in range(cg_time[i], horizon):
                        if env == 'bandit':
                            true_actions[i, idx, :] = post_opt_a[i, :]
                            true_context[i, idx, :] = means[i, :]
                        elif env == 'cgbandit': 
                            true_actions[i, idx, :] = post_opt_a[i, :]
                            true_context[i, idx, :] = means[i, 1, :]
                    
                context_pred = model_ctx(batch)
                batch['context'] = context_pred.clone().detach()
                pred_actions = model_act(batch)

                optimizer_act.zero_grad()
                optimizer_ctx.zero_grad()
                loss_act_pred = loss_fn2(pred_actions.reshape(-1, action_dim), true_actions.reshape(-1, action_dim))

                loss_ctx = loss_fn1(context_pred.reshape(-1, action_dim), true_context.reshape(-1, action_dim))
                
                curiousity = (context_pred.detach() - true_context) ** 2
                loss_act_exp = torch.einsum('ijk ,ijk ->ij', curiousity, torch.nn.functional.softmax(pred_actions, dim=-1))

                loss_act_exp = loss_act_exp.sum()

                loss_act = loss_act_pred -  w * loss_act_exp
         
                loss = loss_ctx + loss_act
                loss.backward() 
                epoch_train_loss += loss.item() / horizon
                epoch_train_act_loss += loss_act.item() / horizon
                epoch_train_context_loss += loss_ctx.item() / horizon

                optimizer_act.step()
                optimizer_ctx.step()


            train_loss.append(epoch_train_loss / len(train_loader.dataset))
            train_act_loss.append(epoch_train_act_loss / len(train_loader.dataset))
            train_context_loss.append(epoch_train_context_loss / len(train_loader.dataset))
            end_time = time.time()
            printw(f"\tTrain loss: {train_loss[-1]}", log_filename)
            printw(f"\t Train Action loss: {train_act_loss[-1]}", log_filename)
            printw(f"\t Train Context loss: {train_context_loss[-1]}", log_filename)
            printw(f"\tTrain time: {end_time - start_time}", log_filename)
            # LOGGING
            if (epoch + 1) % 20 == 0:
                torch.save(model_ctx.state_dict(), f'models/{filename}_model_ctx_epoch{epoch+1}.pt')
                torch.save(model_act.state_dict(), f'models/{filename}_model_act_epoch{epoch+1}.pt')

            # PLOTTING
            if (epoch + 1) % 10 == 0:
                printw(f"Test Loss:        {test_loss[-1]}", log_filename)
                printw(f"Train Loss:       {train_loss[-1]}", log_filename)
                printw("\n", log_filename)
                plt.yscale('log')
                plt.plot(train_loss[1:], label="Train Loss")
                plt.plot(test_loss[1:], label="Test Loss")
                plt.legend()
                plt.savefig(f"figs/loss/{filename}_train_loss.png")
                plt.clf()
        print("Done.")

def pretrain():

    if __name__ == '__main__':
        if not os.path.exists('figs/loss'):
            os.makedirs('figs/loss', exist_ok=True)
        if not os.path.exists('models'):
            os.makedirs('models', exist_ok=True)
        parser = argparse.ArgumentParser()
        common_args.add_dataset_args(parser)
        common_args.add_model_args(parser)
        common_args.add_train_args(parser)
        parser.add_argument('--seed', type=int, default=0)
        args = vars(parser.parse_args())
        print("Args: ", args)
        env = args['env']
        n_envs = args['envs']
        n_hists = args['hists']
        n_samples = args['samples']
        horizon = args['H']
        dim = args['dim']
        state_dim = 1
        action_dim = dim
        n_embd = args['embd']
        n_head = args['head']
        n_layer = args['layer']
        context_len = args['context_len']
        lr = args['lr']
        shuffle = args['shuffle']
        dropout = args['dropout']
        var = args['var']
        cov = args['cov']
        num_epochs = args['num_epochs']
        seed = args['seed']
        tmp_seed = seed
        if seed == -1:
            tmp_seed = 0
        torch.manual_seed(tmp_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(tmp_seed)
            torch.cuda.manual_seed_all(tmp_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(tmp_seed)
        random.seed(tmp_seed)
       
        dataset_config = {
            'n_hists': n_hists,
            'n_samples': n_samples,
            'horizon': horizon,
            'dim': dim,
        }
        dataset_config.update({'var': var, 'cov': cov, 'type': 'uniform'})

        model_config = {
            'shuffle': shuffle,
            'lr': lr,
            'dropout': dropout,
            'n_embd': n_embd,
            'n_layer': n_layer,
            'n_head': n_head,
            'n_envs': n_envs,
            'n_hists': n_hists,
            'n_samples': n_samples,
            'horizon': horizon,
            'dim': dim,
            'seed': seed,
            'context_len': context_len,
        }
        model_config.update({'var': var, 'cov': cov})
        config = {
            'horizon': horizon,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'n_layer': n_layer,
            'n_embd': n_embd,
            'n_head': n_head,
            'shuffle': shuffle,
            'dropout': dropout,
            'test': False,
            'store_gpu': True,
            'context_len': context_len,
        }

        model = pretrain_transformer(config).to(device)
        params = {
            'batch_size': 100,
            'shuffle': True,
            'drop_last': True,
        }
        batch_size = 100
        filename = build_model_filename(env, model_config)
        log_filename = f'figs/loss/{filename}_logs.txt'
        with open(log_filename, 'w') as f:
            pass
        def printw(string):
            """
            A drop-in replacement for print that also writes to a log file.
            """
            # Use the standard print function to print to the console
            print(string)
            # Write the same output to the log file
            with open(log_filename, 'a') as f:
                f.write(string + '\n')
        path_train = build_data_filename(env, n_envs, dataset_config, mode=0)
        path_test = build_data_filename(env, n_envs, dataset_config, mode=1)
        train_dataset = Dataset(path = path_train, config = config)
        test_dataset = Dataset(path = path_test, config = config)
        train_loader0 = torch.utils.data.DataLoader(train_dataset, **params)
        test_loader = torch.utils.data.DataLoader(test_dataset, **params)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        test_loss = []
        train_loss = []

        printw("Num train batches: " + str(len(train_loader0)))
        printw("Num test batches: " + str(len(test_loader)))
        filename = f'{filename}'
        
        # start_epoch, model = load_checkpoints(model, filename)
        start_epoch = 0

        if start_epoch == 0:
            printw("Starting from scratch.")
        else:
            printw(f"Starting from epoch {start_epoch}")
        
        threshold = 1.0
        train_loader = train_loader0

        for epoch in range(start_epoch, num_epochs):
            # EVALUATION
            printw(f"Epoch: {epoch + 1}")
            start_time = time.time()
            with torch.no_grad():
                epoch_test_loss = 0.0
                for i, batch in enumerate(test_loader):
                    print(f"Batch {i} of {len(test_loader)}", end='\r')

                    batch = {k: v.to(device) for k, v in batch.items()}
                    true_actions = torch.zeros((params['batch_size'], horizon, action_dim)).to(device)
                    pre_opt_a = batch['optimal_actions'][:, 0:dim]
                    post_opt_a = batch['optimal_actions'][:, dim:]
                    cg_time = batch['cg_times'].squeeze().tolist()
                    cg_time = [int(t) for t in cg_time]
                    context_return = batch['context_rewards']

                    for i in range(params['batch_size']):
                        for idx in range(cg_time[i]):
                            true_actions[i, idx, :] = pre_opt_a[i, :]
                        for idx in range(cg_time[i], horizon):
                            true_actions[i, idx, :] = post_opt_a[i, :]

                    pred_actions = model(batch)
                    true_actions = true_actions.reshape(-1, action_dim)
                    pred_actions = pred_actions.reshape(-1, action_dim)
                
                    loss = loss_fn(pred_actions, true_actions)
                    epoch_test_loss += loss.item() / horizon
            test_loss.append(epoch_test_loss / len(test_dataset))
            end_time = time.time()
            printw(f"\tTest loss: {test_loss[-1]}")
            printw(f"\tEval time: {end_time - start_time}")
            # TRAINING
            epoch_train_loss = 0.0
            start_time = time.time()
            for i, batch in enumerate(train_loader):
                print(f"Batch {i} of {len(train_loader)}", end='\r')
                batch = {k: v.to(device) for k, v in batch.items()}

                true_actions = torch.zeros((params['batch_size'], horizon, action_dim)).to(device)
                pre_opt_a = batch['optimal_actions'][:, :action_dim]  # of size (batch_size, action_dim)
                post_opt_a = batch['optimal_actions'][:, action_dim:]  # of size (batch_size, action_dim)
                cg_time = batch['cg_times']  # of size (batch_size, 1)
                cg_time = cg_time.squeeze().tolist()
                cg_time = [int(t) for t in cg_time]  # turn into iterable list
                context_return = batch['context_rewards']

                for i in range(params['batch_size']):
                    for idx in range(cg_time[i]):
                        true_actions[i, idx, :] = pre_opt_a[i, :]
                    for idx in range(cg_time[i], horizon):
                        true_actions[i, idx, :] = post_opt_a[i, :]

                pred_actions = model(batch)
                pred_actions = pred_actions.reshape(-1, action_dim)
                optimizer.zero_grad()
                loss = loss_fn(pred_actions, true_actions.reshape(-1, action_dim))
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item() / horizon
            train_loss.append(epoch_train_loss / len(train_dataset))
            end_time = time.time()
            printw(f"\tTrain loss: {train_loss[-1]}")
            printw(f"\tTrain time: {end_time - start_time}")
            # LOGGING
            if (epoch + 1) % 50 == 0:
                torch.save(model.state_dict(), f'models/{filename}_epoch{epoch+1}.pt')
            # PLOTTING
            if (epoch + 1) % 10 == 0:
                printw(f"Test Loss:        {test_loss[-1]}")
                printw(f"Train Loss:       {train_loss[-1]}")
                printw("\n")
                plt.yscale('log')
                plt.plot(train_loss[1:], label="Train Loss")
                plt.plot(test_loss[1:], label="Test Loss")
                plt.legend()
                plt.savefig(f"figs/loss/{filename}_train_loss.png")
                plt.clf()
        torch.save(model.state_dict(), f'models/{filename}.pt')
        print("Done.")
pretrain()