import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelProperties:
    def __init__(self, Nout, Nhidden, Nin, Lplan, greedy_actions, no_planning=False):
        self.Nout = Nout
        self.Nhidden = Nhidden
        self.Nin = Nin
        self.Lplan = Lplan
        self.greedy_actions = greedy_actions
        self.no_planning = no_planning  # If true, never stand still

class ModularModel:
    def __init__(self, model_properties, network, policy, prediction, forward):
        self.model_properties = model_properties
        self.network = network
        self.policy = policy
        self.prediction = prediction
        self.forward = forward

def forward_modular():
    # Placeholder function for forward operation, depending on the rest of your model's logic
    pass

class GRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, x, hidden_state=None):
        out, hidden = self.gru(x, hidden_state)
        return out, hidden

class PolicyNetwork(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc(x)

class PredictionNetwork(nn.Module):
    def __init__(self, hidden_size, Naction, Npred_out):
        super(PredictionNetwork, self).__init__()
        self.fc1 = nn.Linear(hidden_size + Naction, Npred_out)
        self.fc2 = nn.Linear(Npred_out, Npred_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def Modular_model(mp, Naction, Nstates=None, neighbor=False):
    network = GRUNetwork(mp.Nin, mp.Nhidden)
    policy = PolicyNetwork(mp.Nhidden, Naction + 1)  # Policy and value function
    Npred_out = mp.Nout - Naction - 1
    prediction = PredictionNetwork(mp.Nhidden, Naction, Npred_out)
    
    return ModularModel(mp, network, policy, prediction, forward_modular)

def build_model(mp, Naction):
    return Modular_model(mp, Naction)

def create_model_name(Nhidden, T, seed, Lplan, prefix=""):
    mod_name = (
        f"{prefix}N{Nhidden}_T{T}_Lplan{Lplan}_seed{seed}"
    )
    return mod_name
