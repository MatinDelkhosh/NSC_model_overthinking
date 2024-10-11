import numpy as np
from scipy.stats import rv_discrete
from src.ToPlanOrNotToPlan import *
from src.walls_baselines import *
from src.environment import *
from src.maze import *

def reset_agent_state(Larena, reward_location, batch):
    Nstates = Larena ** 2
    # Random starting location (2 x batch), make sure not to start at reward
    agent_state = np.random.randint(1, Larena + 1, (2, batch))  # Random start location
    for b in range(batch):
        tele_reward_location = np.ones(Nstates) / (Nstates - 1)
        tele_reward_location[reward_location[:, b].astype(bool)] = 0  # Exclude reward location
        agent_state[:, b] = state_from_loc(Larena, np.random.choice(Nstates, p=tele_reward_location))
    return agent_state

def gen_maze_walls(Larena, batch):
    wall_loc = np.zeros((Larena ** 2, 4, batch), dtype=np.float32)  # Walls between neighboring agent states
    for b in range(batch):
        wall_loc[:, :, b] = maze(Larena)  # Generate maze for each batch
    return wall_loc

def initialize_arena(reward_location, agent_state, batch, model_properties, environment_dimensions, initial_plan_state, initial_params=None):
    Larena = environment_dimensions['Larena']
    Nstates = Larena ** 2

    # Initialize reward location if not provided
    if np.max(reward_location) <= 0:
        reward_location = np.zeros((Nstates, batch), dtype=np.float32)
        rew_loc = np.random.choice(Nstates, batch)
        for b in range(batch):
            reward_location[rew_loc[b], b] = 1.0

    # Initialize agent state if not provided
    if np.max(agent_state) <= 0:
        agent_state = reset_agent_state(Larena, reward_location, batch)

    # Initialize wall locations if not provided
    if initial_params is not None:
        wall_loc = initial_params
    else:
        wall_loc = gen_maze_walls(Larena, batch)

    # Initialize world state
    wall_state = WallState(
        wall_loc=np.int32(wall_loc),
        reward_location=np.float32(reward_location),
        time=np.ones(batch, dtype=np.float32)
    )
    world_state = WorldState(
        environment_state=wall_state,
        agent_state=np.int32(agent_state),
        planning_state=initial_plan_state(batch)
    )

    ahot = np.zeros((environment_dimensions.Naction, batch), dtype=np.float32)
    rew = np.zeros((1, batch), dtype=np.float32)
    x = gen_input(world_state, ahot, rew, environment_dimensions, model_properties)

    return world_state, np.float32(x)