import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)  # or 'forkserver'

from itertools import chain
import argparse
import os
import time

import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import LambdaLR

import numpy as np
import common_args
import random
from dataset import Dataset
from net import Transformer, Context_extractor, pretrain_transformer
from utils import (
    build_data_filename,
    build_model_filename,
    build_darkroom_data_filename,
    build_darkroom_model_filename,
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
        

def wsd_lr_scheduler(optimizer, warmup_steps, total_steps, base_lr):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Warmup
            return current_step / warmup_steps
        elif current_step < total_steps:
            # Step Phase
            return 1.0
        else:
            # Decay
            decay_steps = total_steps - warmup_steps
            return (0.1 ** ((current_step - warmup_steps) / decay_steps))
    
    return LambdaLR(optimizer, lr_lambda)

def load_checkpoints(model, filename):
    # Check if there are existing model checkpoints
    checkpoint_files = [f for f in os.listdir('models') if f.startswith(f"{filename}_") ]
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
    model_class = args['class']
    context_type = args['context_type']
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
    batch_size = args['batch_size']
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
        'class': model_class,
        'exploration_rate': w,
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
    # model_config is for building filename, config is now created for initializing the model
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
    # config for training
    params = {
        'batch_size': batch_size,
        'shuffle': True,
        'drop_last': True,
    }

    return (
        config, model_config, dataset_config, params, env, context_type, num_epochs, w, n_envs, lr, horizon, dim, action_dim
    )

def ppt_trainer(context_pred, action_pred, true_context, true_action, w):
    assert context_pred.shape == true_context.shape, f"Shape mismatch: context_pred {context_pred.shape} vs true_context {true_context.shape}"
    assert action_pred.shape == true_action.shape, f"Shape mismatch: action_pred {action_pred.shape} vs true_action {true_action.shape}"
    assert len(action_pred.shape) == 3, f"Expected 3D tensor, got action_pred of shape {action_pred.shape}"
    assert len(context_pred.shape) == 3, f"Expected 3D tensor, got context_pred of shape {context_pred.shape}"
    
    loss_ctx = torch.nn.functional.mse_loss(context_pred.reshape(-1, context_pred.shape[-1]), true_context.reshape(-1, context_pred.shape[-1]), reduction='sum')    
    loss_act_ce = torch.nn.functional.cross_entropy(action_pred.reshape(-1, action_pred.shape[-1]), true_action.reshape(-1, action_pred.shape[-1]), reduction='sum')    
    curiousity = (context_pred.detach() - true_context)**2
    bonus_act_exp = torch.einsum('ijk ,ijk ->ij', curiousity, torch.nn.functional.softmax(action_pred, dim=-1))
    loss_act = loss_act_ce - w * bonus_act_exp.sum()
    return (loss_act, loss_act_ce, loss_ctx)



def train_ppt():
    if not os.path.exists('figs/loss'):
        os.makedirs('figs/loss', exist_ok=True)
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)
    
    config, model_config, dataset_config, params, env, context_type, num_epochs, w, n_envs, lr, horizon, dim, action_dim = load_args()

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
            epoch_test_act_loss = 0.0
            epoch_test_context_loss = 0.0
            for i, batch in enumerate(test_loader):
                print(f"Batch {i} of {len(test_loader)}", end='\r')

                # Load batch
                batch = {k: v.to(device) for k, v in batch.items()}
                # true_context = batch['true_context'].clone().detach()
                # true_context = true_context.unsqueeze(1).expand(-1, horizon, -1)
                true_context = batch['context'].clone().detach()
                context_copy = batch['context'].clone().detach()
                true_actions = batch['optimal_actions'].clone().detach()
                true_actions = true_actions.unsqueeze(1).expand(-1, horizon, -1)

                # Rollout context predictions and add to batch
                context_pred = model_ctx(batch)
                batch['context'] = context_pred.detach()
                # Predict actions
                pred_actions = model_act(batch)
                batch['context'] = context_copy

                (loss_act, loss_act_ce, loss_ctx) = ppt_trainer(context_pred, pred_actions, true_context, true_actions, w)

                epoch_test_act_loss += loss_act_ce.item() / horizon
                epoch_test_context_loss += loss_ctx.item() / horizon

        test_act_loss.append(epoch_test_act_loss / len(test_dataset))
        test_context_loss.append(epoch_test_context_loss / len(test_dataset))
        end_time = time.time()
        printw(f"\t Test Action loss: {test_act_loss[-1]}", log_filename)
        printw(f"\t Test Context loss: {test_context_loss[-1]}", log_filename)
        printw(f"\tEval time: {end_time - start_time}", log_filename)
        # TRAINING
        epoch_train_act_loss = 0.0
        epoch_train_context_loss = 0.0
        start_time = time.time()

        # Early stopping parameters
        patience = 5  # Number of epochs to wait before stopping
        best_loss = float('inf')
        patience_counter = 0

        
        for i, batch in enumerate(train_loader):
            print(f"Batch {i} of {len(train_loader)}", end='\r')
                            # Load batch
            batch = {k: v.to(device) for k, v in batch.items()}

            context_copy = batch['context'].clone().detach()

            if context_type == 'ground_truth':
                true_context = batch['true_context'].clone().detach()
                true_context = true_context.unsqueeze(1).expand(-1, horizon, -1) 
            else:
                true_context = batch['context'].clone().detach()[:, -1]
                true_context = true_context.unsqueeze(1).expand(-1, horizon, -1)

            true_actions = batch['optimal_actions'].clone().detach()
            true_actions = true_actions.unsqueeze(1).expand(-1, horizon, -1)

            # Rollout context predictions and add to batch
            context_pred = model_ctx(batch)
            batch['context']= context_pred.detach()
            # Predict actions
            pred_actions = model_act(batch)
            batch['context'] = context_copy
            
            optimizer_act.zero_grad()
            optimizer_ctx.zero_grad()
            (loss_act, loss_act_ce, loss_ctx) = ppt_trainer(context_pred, pred_actions, true_context, true_actions, w)
            loss_act.backward()
            loss_ctx.backward()
            optimizer_act.step()
            optimizer_ctx.step()

            epoch_train_act_loss += loss_act_ce.item() / horizon
            epoch_train_context_loss += loss_ctx.item() / horizon

            optimizer_act.step()
            optimizer_ctx.step()

        train_act_loss.append(epoch_train_act_loss / len(train_loader.dataset))
        train_context_loss.append(epoch_train_context_loss / len(train_loader.dataset))
        end_time = time.time()
        printw(f"\t Train Action loss: {train_act_loss[-1]}", log_filename)
        printw(f"\t Train Context loss: {train_context_loss[-1]}", log_filename)
        printw(f"\tTrain time: {end_time - start_time}", log_filename)
        # LOGGING
        if (epoch + 1) % 10 == 0:
            torch.save(model_ctx.state_dict(), f'models/{filename}_model_ctx_epoch{epoch+1}.pt')
            torch.save(model_act.state_dict(), f'models/{filename}_model_act_epoch{epoch+1}.pt')

        # PLOTTING
        if (epoch + 1) % 10 == 0:
            printw(f"Test Action Loss:        {test_act_loss[-1]}", log_filename)
            printw(f"Train Action Loss:       {train_act_loss[-1]}", log_filename)
            printw(f"Test Context Loss:        {test_context_loss[-1]}", log_filename)
            printw(f"Train Context Loss:       {train_context_loss[-1]}", log_filename)
            printw("\n", log_filename)
            plt.yscale('log')
            plt.plot(train_act_loss[1:], label="Train Action Loss")
            plt.plot(test_act_loss[1:], label="Test Action Loss")
            plt.plot(train_context_loss[1:], label="Train Context Loss")
            plt.plot(test_context_loss[1:], label="Test Context Loss")
            plt.legend()
            plt.savefig(f"figs/loss/{filename}_train_loss.png")
            plt.clf()
                
        # # Early stopping logic
        # if epoch >= 10:  # Only apply early stopping after the 10th iteration
        #     if test_act_loss[-1] < best_loss:
        #         best_loss = test_act_loss[-1]
        #         patience_counter = 0  # Reset patience counter
        #     else:
        #         patience_counter += 1
        #         printw(f"\tTest loss increased. Patience counter: {patience_counter}")
        #         if patience_counter >= patience:
        #             printw("Early stopping triggered. Training halted.")
        #             break
    torch.save(model_ctx.state_dict(), f'models/{filename}_model_ctx.pt')
    torch.save(model_act.state_dict(), f'models/{filename}_model_act.pt')
    print("Done.")

def train_dpt():

    if not os.path.exists('figs/loss'):
        os.makedirs('figs/loss', exist_ok=True)
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)
    
    config, dpt_config, dataset_config, params, env, context_type, num_epochs, w, n_envs, lr, horizon, dim, action_dim = load_args()

    model = pretrain_transformer(config).to(device)
    
    if env == 'bandit':
        path_train = build_data_filename(env, n_envs, dataset_config, mode=0)
        path_test = build_data_filename(env, n_envs, dataset_config, mode=1)
        filename = build_model_filename(env, dpt_config)
    
    elif env.startswith('darkroom'):
        state_dim = 2
        action_dim = 5

        dataset_config.update({'rollin_type': 'uniform'})
        path_train = build_darkroom_data_filename(
            env, n_envs, dataset_config, mode=0)
        path_test = build_darkroom_data_filename(
            env, n_envs, dataset_config, mode=1)

        filename = build_darkroom_model_filename(env, model_config)

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

    
    train_dataset = Dataset(path = path_train, config = config)
    test_dataset = Dataset(path = path_test, config = config)
    train_loader = torch.utils.data.DataLoader(train_dataset, **params)
    test_loader = torch.utils.data.DataLoader(test_dataset, **params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    test_loss = []
    train_loss = []

    printw("Num train batches: " + str(len(train_loader)))
    printw("Num test batches: " + str(len(test_loader)))
    filename = f'{filename}'
    
    start_epoch, model = load_checkpoints(model, filename)

    if start_epoch == 0:
        printw("Starting from scratch.")
    else:
        printw(f"Starting from epoch {start_epoch}")

    # Early stopping parameters
    patience = 5  # Number of epochs to wait before stopping
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(start_epoch, num_epochs):
        # EVALUATION
        printw(f"Epoch: {epoch + 1}")
        start_time = time.time()
        with torch.no_grad():
            epoch_test_loss = 0.0
            for i, batch in enumerate(test_loader):
                print(f"Batch {i} of {len(test_loader)}", end='\r')

                batch = {k: v.to(device) for k, v in batch.items()}
                true_actions = batch['optimal_actions'].unsqueeze(1).expand(-1, horizon, -1)

                pred_actions = model(batch)
            
                loss = torch.nn.functional.cross_entropy(pred_actions.reshape(-1, action_dim), true_actions.reshape(-1, action_dim), reduction='sum')   
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
            true_actions = batch['optimal_actions'].unsqueeze(1).expand(-1, horizon, -1)

            pred_actions = model(batch)
        
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(pred_actions.reshape(-1, action_dim), true_actions.reshape(-1, action_dim), reduction='sum')   
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() / horizon
        train_loss.append(epoch_train_loss / len(train_dataset))
        end_time = time.time()
        printw(f"\tTrain loss: {train_loss[-1]}")
        printw(f"\tTrain time: {end_time - start_time}")
        # LOGGING
        if (epoch + 1) % 5 == 0:
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
        
        # # Early stopping logic
        # if test_loss[-1] < best_loss:
        #     best_loss = test_loss[-1]
        #     patience_counter = 0  # Reset patience counter
        # else:
        #     patience_counter += 1
        #     printw(f"\tTest loss increased. Patience counter: {patience_counter}")
        #     if patience_counter >= patience:
        #         printw("Early stopping triggered. Training halted.")
        #         torch.save(model.state_dict(), f'models/{filename}.pt')
        #         break
    torch.save(model.state_dict(), f'models/{filename}.pt')
    print("Done.")


if __name__ == '__main__':
    model_config = load_args()[1]
    if model_config['class'] == 'dpt':
        train_dpt()
    elif model_config['class'] == 'ppt':    
        train_ppt()
