import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from scipy.special import softmax
import logging
from src.ToPlanOrNotToPlan import *

def sample_actions(model, policy_logits):
    # Don't differentiate through this sampling process
    batch = policy_logits.shape[1]
    actions = np.zeros((1, batch), dtype=np.int32)
    
    pi_t = softmax(policy_logits.astype(np.float64), axis=0)  # Probability of actions (up/down/right/left/stay)
    
    if np.isnan(pi_t).any() or np.isinf(pi_t).any():  # Fix issues with inf/nan in softmax
        for b in range(batch):
            if np.isnan(pi_t[:, b]).any() or np.isinf(pi_t[:, b]).any():
                pi_t[:, b] = np.zeros_like(pi_t[:, b])
                pi_t[np.argmax(policy_logits[:, b]), b] = 1

    if model.model_properties.greedy_actions:  # Optionally select greedy action
        actions[0, :] = np.argmax(pi_t, axis=0).astype(np.int32)
    else:
        actions[0, :] = [np.random.choice(range(pi_t.shape[0]), p=pi_t[:, b]) for b in range(batch)]
    
    return actions

def zeropad_data(rew, agent_input, actions, active):
    # Zero-pad data for finished episodes
    finished = np.where(~active)[0]
    if len(finished) > 0:
        rew[:, finished] = 0
        agent_input[:, finished] = 0
        actions[:, finished] = 0
    return rew, agent_input, actions

def calc_prediction_loss(agent_output, Naction, Nstates, s_index, active):
    new_Lpred = 0.0
    spred = agent_output[(Naction + 1):(Naction + 1 + Nstates), :]
    spred = spred - tf.math.reduce_logsumexp(spred, axis=0)  # Softmax over states
    for b in np.where(active)[0]:
        new_Lpred -= spred[s_index[b], b]  # -log p(s_true)
    return new_Lpred

def calc_rew_prediction_loss(agent_output, Naction, Nstates, r_index, active):
    new_Lpred = 0.0
    i1 = Naction + Nstates + 3
    i2 = Naction + Nstates + 2 + Nstates
    rpred = agent_output[i1:i2, :]
    rpred = rpred - tf.math.reduce_logsumexp(rpred, axis=0)  # Softmax over rewards
    for b in np.where(active)[0]:
        new_Lpred -= rpred[r_index[b], b]  # -log p(r_true)
    return new_Lpred

def construct_ahot(actions, Naction):
    batch = actions.shape[1]
    ahot = np.zeros((Naction, batch), dtype=np.float32)
    for b in range(batch):
        ahot[actions[0, b], b] = 1.0
    return ahot

def forward_modular(model, env_dimensions, x, h_rnn):
    Naction, Nstates, batch = env_dimensions.Naction, env_dimensions.Nstates, x.shape[1]
    h_rnn, ytemp = model.network[GRUind](x, h_rnn)  # Forward pass through recurrent RNN
    logpi_V = model.policy(ytemp)  # Apply policy and value network
    logpi = logpi_V[:Naction, :]  # Policy logits
    V = logpi_V[Naction:(Naction+1), :]  # Value function
    
    no_planning = model.model_properties.no_planning
    if isinstance(no_planning, bool):
        if no_planning:
            logpi[4, :] = -np.inf  # Set the log prob of "plan" to -inf
    else:
        logpi = logpi - tf.math.reduce_logsumexp(logpi, axis=0)  # Softmax normalization
        logpi[4, :] += float(no_planning)

    logpi = logpi - tf.math.reduce_logsumexp(logpi, axis=0)
    actions = sample_actions(model, logpi)  # Sample action based on log probabilities
    ahot = construct_ahot(actions, Naction)  # One-hot representation of actions

    prediction_input = np.vstack([ytemp, ahot])  # Concatenation of hidden state and actions
    prediction_output = model.prediction(prediction_input)  # Output of prediction module

    return h_rnn, np.vstack([logpi, V, prediction_output]), actions  # Return hidden state, outputs, and sampled action

def calc_deltas(rews, Vs):
    batch, N = rews.shape
    deltas = np.zeros_like(rews, dtype=np.float32)
    R = np.zeros(batch)
    
    for t in range(N - 1, -1, -1):
        R = rews[:, t] + R  # Cumulative reward
        deltas[:, t] = R - Vs[:, t]  # TD error
    return deltas

def run_episode(model, environment, loss_hp, reward_location=np.zeros(2), agent_state=np.zeros(2),
                hidden=True, batch=2, calc_loss=True, initial_params=None):
    ed = environment.dimensions
    Nout = model.model_properties.Nout
    Nhidden = model.model_properties.Nhidden
    Nstates = ed.Nstates
    Naction = ed.Naction
    T = ed.T

    world_state, agent_input = environment.initialize(
        reward_location, agent_state, batch, model.model_properties, initial_params=initial_params
    )

    agent_outputs = np.empty((Nout, batch, 0), dtype=np.float32)
    if hidden:
        hs = np.empty((Nhidden, batch, 0), dtype=np.float32)
        world_states = []

    actions = np.empty((1, batch, 0), dtype=np.int32)
    rews = np.empty((1, batch, 0), dtype=np.float32)

    h_rnn = np.zeros((Nhidden, batch), dtype=np.float32)

    Lpred = 0.0
    Lprior = 0.0

    while np.any(world_state.environment_state.time < T):
        if hidden:
            hs = np.concatenate([hs, h_rnn[:, :, np.newaxis]], axis=2)
            world_states.append(world_state)

        h_rnn, agent_output, a = forward_modular(model, ed, agent_input, h_rnn)
        active = world_state.environment_state.time < T

        if calc_loss:
            Lprior += prior_loss(agent_output, world_state.agent_state, active, model)

        rew, agent_input, world_state, predictions = environment.step(
            agent_output, a, world_state, ed, model.model_properties, model, h_rnn
        )
        rew, agent_input, a = zeropad_data(rew, agent_input, a, active)

        agent_outputs = np.concatenate([agent_outputs, agent_output[:, :, np.newaxis]], axis=2)
        rews = np.concatenate([rews, rew[:, :, np.newaxis]], axis=2)
        actions = np.concatenate([actions, a[:, :, np.newaxis]], axis=2)

        if calc_loss:
            s_index, r_index = predictions
            Lpred += calc_prediction_loss(agent_output, Naction, Nstates, s_index, active)
            Lpred += calc_rew_prediction_loss(agent_output, Naction, Nstates, r_index, active)

    if calc_loss:
        return agent_outputs, hs, actions, rews, Lpred, Lprior, world_states
    return agent_outputs, hs, actions, rews, world_states

def model_loss(model, environment, loss_hp, batch_size):
    """
    Wrapper function for TensorFlow/Keras that simulates an episode 
    and returns the scalar loss.
    
    Args:
        model: The neural network model.
        environment: The environment instance in which the agent operates.
        loss_hp: Hyperparameters related to loss computation.
        batch_size: Number of parallel episodes to run in a batch.
        
    Returns:
        loss: The computed loss for the batch.
    """
    
    # Run a full episode, collect outputs and get the loss
    loss, _, _, _, _ = run_episode(model, environment, loss_hp, hidden=False, batch=batch_size)
    
    return loss
