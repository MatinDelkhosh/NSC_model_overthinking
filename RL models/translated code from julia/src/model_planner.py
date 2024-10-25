import numpy as np
from scipy.special import logsumexp
from random import sample
from collections import defaultdict
from scipy.stats import rv_discrete
from src.ToPlanOrNotToPlan import *
from src.walls import *
from src.planning import *
from src.environment import *

# Assuming placeholders for undefined objects like GRUind, WorldState, etc.

def model_tree_search(goal, world_state, model, h_rnn, plan_inds, times, ed, mp, planner, Print=False):
    Larena, Naction = ed.Larena, ed.Naction
    Nstates = Larena ** 2

    batch = h_rnn.shape[1]
    path = np.zeros((4, planner.Lplan, batch))
    all_Vs = np.zeros((batch,), dtype=np.float32)
    found_rew = np.zeros((batch,), dtype=np.float32)
    plan_states = np.zeros((planner.Lplan, batch), dtype=np.int32)
    wall_loc = world_state.environment_state.wall_loc

    # Only consider planning states
    h_rnn = h_rnn[:, plan_inds]
    goal = goal[plan_inds]
    times = times[plan_inds]
    wall_loc = wall_loc[:, :, plan_inds]
    ytemp = h_rnn

    agent_input = np.zeros((mp.Nin,), dtype=np.float32)
    new_world_state = world_state

    for n_steps in range(1, planner.Lplan + 1):
        batch = len(goal)
        global GRUind
        if n_steps > 1:
            h_rnn, ytemp = model.network[GRUind].cell(h_rnn, agent_input)  # Forward pass
        
        # Generate actions from hidden activity
        logπ_V = model.policy(ytemp)
        logπ = logπ_V[:4, :] - logsumexp(logπ_V[:4, :], axis=0)
        Vs = logπ_V[5, :] / 10.0

        πt = np.exp(logπ)
        a = np.zeros((1, batch), dtype=np.int32)
        a[0, :] = [rv_discrete(values=(np.arange(1, 5), πt[:, b])).rvs() for b in range(batch)]

        # Record actions
        for ib, b in enumerate(plan_inds):
            path[a[0, ib] - 1, n_steps - 1, b] = 1.0

        # Generate predictions
        ahot = np.zeros((Naction, batch), dtype=np.float32)
        for b in range(batch):
            ahot[a[0, b] - 1, b] = 1.0
        prediction_input = np.vstack((ytemp, ahot))
        prediction_output = model.prediction(prediction_input)

        # Draw new states
        spred = prediction_output[:Nstates, :]
        spred = spred - logsumexp(spred, axis=0)
        state_dist = np.exp(spred)
        new_states = np.array([np.argmax(state_dist[:, b]) + 1 for b in range(batch)], dtype=np.int32)

        if Print:
            print(f"{n_steps} {batch} {np.mean(np.max(πt, axis=0))} {np.mean(np.max(state_dist, axis=0))}")

        # Check finished and not_finished states
        not_finished = np.where(new_states != goal)[0]
        finished = np.where(new_states == goal)[0]

        all_Vs[plan_inds] = Vs
        plan_states[n_steps - 1, plan_inds] = new_states
        found_rew[plan_inds[finished]] += 1.0

        if len(not_finished) == 0:
            return path, all_Vs, found_rew, plan_states

        # Update variables for next iteration
        h_rnn = h_rnn[:, not_finished]
        goal = goal[not_finished]
        plan_inds = plan_inds[not_finished]
        times = times[not_finished] + 1.0
        wall_loc = wall_loc[:, :, not_finished]
        reward_location = onehot_from_loc(Larena, goal)
        xplan = np.zeros((planner.Nplan_in, len(goal)), dtype=np.float32)
        rew = np.zeros((1, len(not_finished)), dtype=np.float32)

        new_world_state = WorldState(
            agent_state=state_from_loc(Larena, new_states[not_finished].T),
            environment_state=WallState(wall_loc, reward_location, time=times),
            planning_state=PlanState(xplan, None)
        )

        agent_input = gen_input(new_world_state, ahot[:, not_finished], rew, ed, mp)

    return path, all_Vs, found_rew, plan_states


def model_planner(world_state, ahot, ed, agent_output, at_rew, planner, model, h_rnn, mp, Print=False, returnall=False, true_transition=False):
    Larena = ed.Larena
    Naction = ed.Naction
    Nstates = ed.Nstates
    batch = ahot.shape[1]
    times = world_state.environment_state.time

    plan_inds = np.where((ahot[4, :] == 1) & (~at_rew))[0]
    rpred = agent_output[Naction + Nstates + 2:Naction + Nstates + 2 + Nstates, :]
    goal = [np.argmax(rpred[:, b]) + 1 for b in range(batch)]

    path, all_Vs, found_rew, plan_states = model_tree_search(goal, world_state, model, h_rnn, plan_inds, times, ed, mp, planner, Print=Print)

    xplan = np.zeros((planner.Nplan_in, batch))
    for b in range(batch):
        xplan[:, b] = np.concatenate((path[:, :, b].ravel(), [found_rew[b]]))

    planning_state = PlanState(xplan, plan_states)

    if returnall:
        return planning_state, plan_inds, (path, all_Vs, found_rew, plan_states)
    
    return planning_state, plan_inds