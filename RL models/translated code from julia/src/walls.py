import numpy as np

class WallState:
    def __init__(self, reward_location, wall_loc, time=None):
        self.reward_location = np.array(reward_location, dtype=np.float32)
        self.wall_loc = np.array(wall_loc, dtype=np.int32)
        self.time = np.array(time if time is not None else np.zeros(1), dtype=np.float32)

def state_ind_from_state(Larena, state):
    """
    Convert state coordinates to state indices.

    Parameters:
    - Larena: Size of the arena.
    - state: Array of shape (2, batch) representing (x, y) coordinates.

    Returns:
    - Array of shape (batch,) with state indices.
    """
    return Larena * (state[0, :] - 1) + state[1, :]

def onehot_from_loc(Larena, loc):
    """
    Convert location indices to one-hot encoded representation.

    Parameters:
    - Larena: Size of the arena.
    - loc: Array of shape (batch,) with location indices.

    Returns:
    - Array of shape (Nstates, batch) with one-hot encoding.
    """
    Nstates = Larena**2
    batch = len(loc)
    shot = np.zeros((Nstates, batch), dtype=np.float32)
    for b in range(batch):
        shot[loc[b], b] = 1.0
    return shot

def onehot_from_state(Larena, state):
    """
    Convert state coordinates to one-hot encoded representation.

    Parameters:
    - Larena: Size of the arena.
    - state: Array of shape (2, batch) representing (x, y) coordinates.

    Returns:
    - Array of shape (Nstates, batch) with one-hot encoding.
    """
    state_ind = state_ind_from_state(Larena, state)
    return onehot_from_loc(Larena, state_ind)

def state_from_loc(Larena, loc):
    """
    Convert location indices to state coordinates.

    Parameters:
    - Larena: Size of the arena.
    - loc: Array of shape (batch,) with location indices.

    Returns:
    - Array of shape (2, batch) with (x, y) coordinates.
    """
    return np.vstack([(loc - 1) // Larena + 1, (loc - 1) % Larena + 1])

def state_from_onehot(Larena, shot):
    """
    Convert one-hot encoded state to state coordinates.

    Parameters:
    - Larena: Size of the arena.
    - shot: Array of shape (Nstates, batch) with one-hot encoding.

    Returns:
    - Array of shape (2, batch) with (x, y) coordinates.
    """
    loc = np.array([np.argsort(-shot[:, b])[0] for b in range(shot.shape[1])])
    return state_from_loc(Larena, loc)

def get_wall_input(state, wall_loc):
    """
    Get wall input for the environment.

    Parameters:
    - state: Array of shape (2, batch) with (x, y) coordinates.
    - wall_loc: Array of shape (Nstates, 4, batch) with wall locations.

    Returns:
    - Array with wall input information.
    """
    return np.vstack([wall_loc[:, 0, :], wall_loc[:, 2, :]])  # Horizontal and vertical walls

def gen_input(world_state, ahot, rew, ed, model_properties):
    """
    Generate input for the agent based on the current world state.

    Parameters:
    - world_state: Dictionary with 'agent_state', 'environment_state', and 'planning_state'.
    - ahot: One-hot encoded actions.
    - rew: Reward values.
    - ed: Environment dimensions.
    - model_properties: Model properties including input dimensions.

    Returns:
    - Array with input for the agent.
    """
    batch = rew.shape[1]
    newstate = world_state['agent_state']
    wall_loc = world_state['environment_state']['wall_loc']
    Naction = ed['Naction']
    Nstates = ed['Nstates']
    shot = onehot_from_state(ed['Larena'], newstate)
    wall_input = get_wall_input(newstate, wall_loc)
    Nwall_in = wall_input.shape[0]
    Nin = model_properties['Nin']
    plan_input = world_state['planning_state']['plan_input']
    Nplan_in = plan_input.shape[0] if plan_input.size > 0 else 0

    x = np.zeros((Nin, batch), dtype=np.float32)
    x[:Naction, :] = ahot
    x[Naction, :] = rew[:]
    x[Naction + 1, :] = world_state['environment_state']['time'] / 50.0  # Normalize time
    x[Naction + 2:Naction + 2 + Nstates, :] = shot
    x[Naction + 2 + Nstates:Naction + 2 + Nstates + Nwall_in, :] = wall_input

    if Nplan_in > 0:
        x[Naction + 2 + Nstates + Nwall_in:Naction + 2 + Nstates + Nwall_in + Nplan_in, :] = plan_input

    return x

def get_rew_locs(reward_location):
    """
    Get the indices of the reward locations.

    Parameters:
    - reward_location: Array with reward locations.

    Returns:
    - Array with indices of the reward locations.
    """
    return np.array([np.argmax(reward_location[:, i]) for i in range(reward_location.shape[1])])