import numpy as np
import torch
import torch.nn.functional as F

def U_prior(state, Naction):
    # Uniform prior
    batch_size = state.shape[1]
    return torch.ones(Naction, batch_size, dtype=torch.float32) / Naction

def prior_loss(agent_output, state, active, mod):
    # Return -KL[q || p]
    act = active.float().unsqueeze(0)
    Naction = len(mod.policy[0].bias) - 1
    logp = torch.log(U_prior(state, Naction))  # KL regularization with uniform prior
    logπ = agent_output[:Naction, :]
    
    if mod.model_properties.no_planning:
        logπ = logπ[:Naction-1, :] - torch.logsumexp(logπ[:Naction-1, :], dim=0)
        logp = logp[:Naction-1, :] - torch.logsumexp(logp[:Naction-1, :], dim=0)
    
    logp = logp * act
    logπ = logπ * act
    lprior = torch.sum(torch.exp(logπ) * (logp - logπ))  # -KL
    return lprior