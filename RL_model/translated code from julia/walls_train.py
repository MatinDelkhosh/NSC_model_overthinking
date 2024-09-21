import argparse
import os
import random
import numpy as np
import torch
from torch import nn, optim
import time

#ToPlanOrNotToPlan
import matplotlib.pyplot as plt
import matplotlib
from src.plotting import *
from src.train import *
from src.model import *
from src.loss_hyperparameters import *
from src.environment import *
from src.a2c import *
from src.initializations import *
from src.walls import *
from src.maze import *
from src.walls_build import *
from src.io import *
from src.model_planner import *
from src.planning import *
from src.human_utils_maze import *
import src.exports
from src.priors import *
from src.walls_baselines import *

# Set some reasonable plotting defaults
matplotlib.rcParams["font.size"] = 16
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False


# Helper functions for model and environment setup would need to be defined in the same way
# For example: build_environment, build_model, create_model_name, save_model, etc.

def parse_commandline():
    parser = argparse.ArgumentParser()

    parser.add_argument("--Nhidden", type=int, default=100, help="Number of hidden units")
    parser.add_argument("--Larena", type=int, default=4, help="Arena size (per side)")
    parser.add_argument("--T", type=int, default=50, help="Number of timesteps per episode")
    parser.add_argument("--Lplan", type=int, default=8, help="Maximum planning horizon")
    parser.add_argument("--load", type=bool, default=False, help="Load previous model instead of initializing a new one")
    parser.add_argument("--load_epoch", type=int, default=0, help="Which epoch to load")
    parser.add_argument("--seed", type=int, default=1, help="Which random seed to use")
    parser.add_argument("--save_dir", type=str, default="./", help="Save directory")
    parser.add_argument("--beta_p", type=float, default=0.5, help="Relative importance of predictive loss")
    parser.add_argument("--prefix", type=str, default="", help="Add prefix to model name")
    parser.add_argument("--load_fname", type=str, default="", help="Model to load")
    parser.add_argument("--n_epochs", type=int, default=1001, help="Total number of training epochs")
    parser.add_argument("--batch_size", type=int, default=40, help="Batch size for each gradient step")
    parser.add_argument("--lrate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--constant_rollout_time", type=bool, default=True, help="Do rollouts take a fixed amount of time")

    return parser.parse_args()

def main():
    # Global parameters
    args = parse_commandline()
    print(args)

    Larena = args.Larena
    Lplan = args.Lplan
    Nhidden = args.Nhidden
    T = args.T
    load = args.load
    seed = args.seed
    save_dir = args.save_dir
    prefix = args.prefix
    load_fname = args.load_fname
    n_epochs = args.n_epochs
    beta_p = args.beta_p
    batch_size = args.batch_size
    lrate = args.lrate
    constant_rollout_time = args.constant_rollout_time

    os.makedirs(save_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Define LossHyperparameters, RL environment, and agent model (need custom implementations)
    loss_hp = {
        "beta_p": beta_p,
        "beta_v": 0.05,
        "beta_e": 0.05,
        "beta_r": 1.0,
    }

    # Build environment and agent
    model_properties, wall_environment, model_eval = build_environment(Larena, Nhidden, T, Lplan=Lplan, constant_rollout_time=constant_rollout_time)
    model = build_model(model_properties, 5)

    # Create model name
    mod_name = create_model_name(Nhidden, T, seed, Lplan, prefix=prefix)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lrate)

    # Keep track of progress
    rews, preds = [], []
    epoch = 0

    if load:
        # Load model if specified
        fname = os.path.join(save_dir, "models", load_fname or f"{mod_name}_{args.load_epoch}")
        model, optimizer, store = recover_model(fname)
        rews, preds = store[0], store[1]
        epoch = len(rews)
        if load_fname:
            optimizer = optim.Adam(model.parameters(), lr=lrate)

    print("Parameter length:", sum(p.numel() for p in model.parameters()))
    for p in model.parameters():
        print(p.size())

    # Check multithreading support (using torch's threading support)
    Nthread = torch.get_num_threads()
    multithread = Nthread > 1
    print(f"Multithreading with {Nthread} threads")
    thread_batch_size = int(np.ceil(batch_size / Nthread))

    # Define the closure for the model loss
    def closure():
        loss = model_loss(model, wall_environment, loss_hp, thread_batch_size) / Nthread
        return loss

    # Training loop
    t0 = time.time()
    while epoch < n_epochs:
        epoch += 1

        # Evaluate the model
        Rmean, pred, mean_a, first_rew = model_eval(model, batch_size, loss_hp)
        model.reset()  # Reset model

        # Save model periodically
        if (epoch - 1) % 50 == 0:
            os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
            filename = os.path.join(save_dir, "models", f"{mod_name}_{epoch - 1}")
            store = [rews, preds]
            save_model(model, store, optimizer, filename, wall_environment, loss_hp, Lplan=Lplan)

        elapsed_time = round(time.time() - t0, 1)
        print(f"Progress: epoch={epoch} t={elapsed_time} R={Rmean} pred={pred} plan={mean_a} first={first_rew}")
        rews.append(Rmean)
        preds.append(pred)
        plot_progress(rews, preds)

        # Train in batches
        for batch in range(200):
            optimizer.zero_grad()
            loss = closure()
            loss.backward()
            optimizer.step()

    model.reset()
    filename = os.path.join(save_dir, "results", mod_name)
    store = [rews, preds]
    save_model(model, store, optimizer, filename, wall_environment, loss_hp, Lplan=Lplan)

if __name__ == "__main__":
    main()