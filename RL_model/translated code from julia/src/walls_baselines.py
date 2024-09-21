import numpy as np

def random_policy(x, md, ed, stay=True):
    """
    Generate a random policy.
    
    Parameters:
    - x: Input tensor (not used directly in this function).
    - md: Model properties containing the output size.
    - ed: Environment descriptor containing the number of actions.
    - stay: Boolean indicating whether to include a 'stay' action.
    
    Returns:
    - ys: Log probabilities for each action.
    """
    batch = x.shape[1]
    ys = np.zeros((md.Nout, batch), dtype=np.float32)
    
    if stay:
        ys[:ed.Naction, :] = np.log(1 / ed.Naction)
    else:
        ys[:ed.Naction - 1, :] = np.log(1 / (ed.Naction - 1))
        ys[ed.Naction, :] = -np.inf
    
    return ys

def dist_to_rew(ps, wall_loc, Larena):
    """
    Compute geodesic distance to the reward from each state, considering walls.
    
    Parameters:
    - ps: State representations.
    - wall_loc: Wall locations.
    - Larena: Size of the arena.
    
    Returns:
    - dists: Matrix of distances to the reward.
    """
    Nstates = Larena**2
    deltas = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])  # Transitions for each action
    rew_loc = state_from_onehot(Larena, ps)  # 2x1
    dists = np.full((Larena, Larena), np.nan)  # Initialize distances to NaN
    dists[rew_loc[0], rew_loc[1]] = 0  # Reward has zero distance
    live_states = np.zeros(Nstates, dtype=bool)
    live_states[state_ind_from_state(Larena, rew_loc)[0]] = True  # Start from reward location
    
    for step in range(1, Nstates):  # Steps from reward
        for state_ind in np.flatnonzero(live_states):  # All states I was at in (step-1) steps
            state = state_from_loc(Larena, state_ind)
            for a in range(4):  # For each action
                if not wall_loc[state_ind, a, 0]:  # If not hitting a wall
                    newstate = state + deltas[a]  # New state
                    newstate = (newstate + Larena - 1) % Larena + 1  # Wrap around
                    if np.isnan(dists[newstate[0], newstate[1]]):  # If not reached in fewer steps
                        dists[newstate[0], newstate[1]] = step  # Distance in `step` steps
                        new_ind = state_ind_from_state(Larena, newstate)[0]
                        live_states[new_ind] = True  # Continue searching from here
            live_states[state_ind] = False  # Done searching for this state
    
    return dists

def optimal_policy(state, wall_loc, dists, ed):
    """
    Return the optimal policy minimizing the path length to the goal.
    
    Parameters:
    - state: Current state (2x1).
    - wall_loc: Wall locations.
    - dists: Matrix of distances to the reward.
    - ed: Environment descriptor containing the number of actions.
    
    Returns:
    - πt: Optimal policy (uniform over actions minimizing distance).
    """
    Naction, Larena = ed.Naction, ed.Larena
    deltas = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])  # Transitions for each action
    nn_dists = np.full(4, np.inf)  # Distance to reward for each action
    state_ind = state_ind_from_state(Larena, state)[0]  # Current state index
    
    for a in range(4):  # For each action
        if not wall_loc[state_ind, a, 0]:  # If not hitting a wall
            newstate = state + deltas[a]  # New state
            newstate = (newstate + Larena - 1) % Larena + 1  # Wrap around
            nn_dists[a] = dists[newstate[0], newstate[1]]  # Distance from new state
    
    optimal_actions = np.flatnonzero(nn_dists == np.min(nn_dists))  # Optimal actions
    πt = np.zeros(Naction)
    πt[optimal_actions] = 1 / len(optimal_actions)  # Uniform policy
    
    return πt

# Placeholder functions for state conversion (assuming these are defined elsewhere)
def state_from_onehot(Larena, ps):
    # Implement conversion from one-hot state representation to actual state coordinates
    pass

def state_ind_from_state(Larena, state):
    # Implement conversion from state coordinates to state index
    pass

def state_from_loc(Larena, state_ind):
    # Implement conversion from state index to state coordinates
    pass