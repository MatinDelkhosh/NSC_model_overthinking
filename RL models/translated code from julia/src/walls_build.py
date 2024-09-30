import numpy as np
from scipy.special import logsumexp
from numpy.random import choice
from numpy.random import rand
from scipy.stats import multinomial
from src.ToPlanOrNotToPlan import *
from planning import *
from walls_baselines import *
from walls import *

def useful_dimensions(Larena, planner):
    """
    Compute useful dimensions for the RL environment.

    Parameters:
    - Larena: Size of the arena.
    - planner: Planning object.

    Returns:
    - Nstates, Nstate_rep, Naction, Nout, Nin: Dimensions of the network.
    """
    Nstates = Larena**2  # Number of states in arena
    Nstate_rep = 2  # Dimensionality of the state representation (e.g., '2' for x, y-coordinates)
    Naction = 5  # Number of actions available
    Nout = Naction + 1 + Nstates  # Actions and value function and prediction of state
    Nout += 1  # Needed for backward compatibility
    Nwall_in = 2 * Nstates  # Provide full info
    Nin = Naction + 1 + 1 + Nstates + Nwall_in  # 5 actions, 1 reward, 1 time, L^2 states, some walls

    Nin += planner.Nplan_in  # Additional inputs from planning
    Nout += planner.Nplan_out  # Additional outputs for planning

    return Nstates, Nstate_rep, Naction, Nout, Nin

def update_agent_state(agent_state, amove, Larena):
    """
    Update the agent's state based on the action taken.

    Parameters:
    - agent_state: Current agent state.
    - amove: Movement resulting from the action.
    - Larena: Size of the arena.

    Returns:
    - new_agent_state: Updated agent state.
    """
    new_agent_state = agent_state + np.array([
        amove[0, :] - amove[1, :],
        amove[2, :] - amove[3, :]
    ])  # 2xbatch
    new_agent_state = (new_agent_state + Larena - 1) % Larena + 1  # 1:L (2xbatch)
    return new_agent_state.astype(np.int32)

def act_and_receive_reward(a, world_state, planner, environment_dimensions, agent_output, model, h_rnn, mp):
    """
    Execute an action and receive a reward.

    Parameters:
    - a: Action taken by the agent.
    - world_state: Current state of the world.
    - planner: Planning object.
    - environment_dimensions: Dimensions of the environment.
    - agent_output: Output of the agent.
    - model: Model object.
    - h_rnn: Hidden state of the RNN.
    - mp: Model properties.

    Returns:
    - rew: Reward received.
    - new_world_state: Updated state of the world.
    - predictions: Predictions made by the agent.
    - ahot: One-hot encoding of the action.
    - at_rew: Indicator if the agent was at the reward location.
    """
    agent_state = world_state['agent_state']
    environment_state = world_state['environment_state']
    reward_location = environment_state['reward_location']
    wall_loc = environment_state['wall_loc']
    Naction = environment_dimensions['Naction']
    Larena = environment_dimensions['Larena']

    agent_state_ind = state_ind_from_state(Larena, agent_state)  # Extract index
    batch = a.shape[1]  # Batch size
    Nstates = Larena**2

    ahot = np.zeros((Naction, batch))  # Attempted action
    amove = np.zeros((Naction, batch))  # Actual movement
    rew = np.zeros((1, batch), dtype=np.float32)  # Reward collected

    # Construct array of attempted and actual movements
    for b in range(batch):
        abatch = int(a[0, b])  # Action
        ahot[abatch, b] = 1.0  # Attempted action
        if (abatch < 4.5) and wall_loc[agent_state_ind[b], abatch, b]:
            rew[0, b] -= 0.0  # Penalty for hitting wall?
        else:
            amove[abatch, b] = 1  # Only move if we don't hit a wall

    new_agent_state = update_agent_state(agent_state, amove, Larena)  # (x, y) coordinates
    shot = onehot_from_state(Larena, new_agent_state)  # One-hot encoding (Nstates x batch)
    s_index = np.concatenate([np.argsort(-shot[:, b])[:1] for b in range(batch)])  # Corresponding index
    r_index = get_rew_locs(reward_location)  # Index of reward location
    predictions = (s_index.astype(np.int32), r_index.astype(np.int32))  # Things to be predicted by the agent

    found_rew = reward_location[shot.astype(bool)]  # Moved to the reward
    s_old_hot = onehot_from_state(Larena, agent_state)  # One-hot encoding of previous agent_state
    at_rew = reward_location[s_old_hot.astype(bool)]  # At reward before action

    moved = np.sum(amove[:4, :], axis=0)  # Did I perform a movement? (size batch)
    rew[0, found_rew & (moved > 0.5)] = 1  # Get reward if agent moved to reward location

    # Teleport the agents that found the reward on the previous iteration
    for b in range(batch):
        if at_rew[b]:  # At reward
            tele_reward_location = np.ones(Nstates) / (Nstates - 1)  # Where can I teleport to (not reward location)
            tele_reward_location[reward_location[:, b].astype(bool)] = 0
            new_state = choice(np.arange(Nstates), p=tele_reward_location)  # Sample new state uniformly at random
            new_agent_state[:, b] = state_from_loc(Larena, new_state)  # Convert to (x, y) coordinates
            shot[:, b] = 0
            shot[new_state, b] = 1  # Update one-hot location

    # Run planning algorithm
    planning_state, plan_inds = planner.planning_algorithm(
        world_state,
        ahot,
        environment_dimensions,
        agent_output,
        at_rew,
        planner,
        model,
        h_rnn,
        mp
    )

    planned = np.zeros(batch, dtype=bool)
    planned[plan_inds] = True  # Which agents within the batch engaged in planning

    # Update the time elapsed for each episode
    new_time = environment_state['time'].copy()
    new_time[~planned] += 1.0  # Increment time for acting
    if planner.constant_rollout_time:
        new_time[planned] += planner.planning_time  # Increment time for planning
    else:
        plan_states = planning_state['plan_cache']
        plan_lengths = np.sum(plan_states[:, planned] > 0.5, axis=0)  # Number of planning steps for each batch
        new_time[planned] += plan_lengths * planner.planning_time / 5
        if rand() < 1e-5:
            print("Variable planning time!", plan_lengths * planner.planning_time / 5)
    
    rew[0, planned] += planner.planning_cost  # Cost of planning (in units of rewards; default 0)

    # Update the state of the world
    new_world_state = {
        'agent_state': new_agent_state,
        'environment_state': {
            'wall_loc': wall_loc,
            'reward_location': reward_location,
            'time': new_time
        },
        'planning_state': planning_state
    }

    return rew.astype(np.float32), new_world_state, predictions, ahot, at_rew

def build_environment(Larena, Nhidden, T, Lplan, greedy_actions=False, no_planning=False, constant_rollout_time=True):
    """
    Construct an environment object which includes 'initialize' and 'step' methods.

    Parameters:
    - Larena: Size of the arena.
    - Nhidden: Number of hidden units.
    - T: Maximum time.
    - Lplan: Planning depth.
    - greedy_actions: Whether to use greedy actions.
    - no_planning: Whether to disable planning.
    - constant_rollout_time: Whether to use constant rollout time.

    Returns:
    - model_properties: Model properties.
    - environment: Environment object.
    - model_eval: Evaluation function for the model.
    """
    # Create planner object
    planner, initial_plan_state = build_planner(Lplan, Larena, constant_rollout_time)
    Nstates, Nstate_rep, Naction, Nout, Nin = useful_dimensions(Larena, planner)  # Compute some useful quantities
    model_properties = {
        'Nout': Nout,
        'Nhidden': Nhidden,
        'Nin': Nin,
        'Lplan': Lplan,
        'greedy_actions': greedy_actions,
        'no_planning': no_planning
    }  # Initialize a model property object
    environment_dimensions = {
        'Nstates': Nstates,
        'Nstate_rep': Nstate_rep,
        'Naction': Naction,
        'T': T,
        'Larena': Larena
    }  # Initialize an environment dimension object

    def step(agent_output, a, world_state, environment_dimensions, model_properties, model, h_rnn):
        """
        Step function that updates the environment.

        Parameters:
        - agent_output: Output of the agent.
        - a: Action taken.
        - world_state: Current state of the world.
        - environment_dimensions: Dimensions of the environment.
        - model_properties: Model properties.
        - model: Model object.
        - h_rnn: Hidden state of the RNN.

        Returns:
        - rew: Reward received.
        - agent_input: Input for the agent.
        - new_world_state: Updated state of the world.
        - predictions: Predictions made by the agent.
        """
        rew, new_world_state, predictions, ahot, at_rew = act_and_receive_reward(
            a, world_state, planner, environment_dimensions, agent_output, model, h_rnn, model_properties
        )  # Take a step through the environment
        # Generate agent input
        agent_input = gen_input(new_world_state, ahot, rew, environment_dimensions, model_properties)
        return rew, agent_input.astype(np.float32), new_world_state, predictions

    def initialize(reward_location, agent_state, batch, mp, initial_params=[]):
        """
        Initialize the environment.

        Parameters:
        - reward_location: Location of the reward.
        - agent_state: Initial state of the agent.
        - batch: Batch size.
        - mp: Model properties.
        - initial_params: Initial parameters.

        Returns:
        - Environment initialization.
        """
        return initialize_arena(reward_location, agent_state, batch, mp, environment_dimensions, initial_plan_state, initial_params=initial_params)

    # Construct environment with initialize() and step() functions and a list of dimensions
    environment = {
        'initialize': initialize,
        'step': step,
        'dimensions': environment_dimensions
    }

    def model_eval(m, batch, loss_hp):
        """
        Evaluate the model.

        Parameters:
        - m: Model object.
        - batch: Batch size.
        - loss_hp: Hyperparameters for loss function.

        Returns:
        - mean_means: Mean of the rewards.
        - mean_preds: Mean of the predictions.
        - mean_all_actions: Mean of all actions.
        - mean_firstrews: Mean of the time to first reward.
        """
        Nrep = 5
        means = np.zeros((Nrep, batch))
        all_actions = np.zeros((Nrep, batch))
        firstrews = np.zeros((Nrep, batch))
        preds = np.zeros((T - 1, Nrep, batch))
        Naction = environment_dimensions['Naction']
        Nstates = environment_dimensions['Nstates']

        for i in range(Nrep):
            _, agent_outputs, rews, actions, world_states, _ = run_episode(
                m, environment, loss_hp, hidden=True, batch=batch
            )
            agent_states = np.concatenate([x['agent_state'] for x in world_states], axis=2)
            means[i, :] = np.sum(rews >= 0.5, axis=1)  # Compute total reward for each batch
            all_actions[i, :] = np.mean(actions == 5, axis=1) / np.mean(actions > 0.5, axis=1)  # Fraction of standing still for each batch
            for b in range(batch):
                firstrews[i, b] = np.argsort(-(rews[b, :] > 0.5))[0]  # Time to first reward
            for t in range(T - 1):
                for b in range(batch):
                    pred = np.argsort(-agent_outputs[Naction + 1 + 1:Naction + 1 + Nstates, b, t])[0]  # Predicted agent_state at time t
                    agent_state = agent_states[:, b, t + 1]  # True agent_state(t+1)
                    preds[t, i, b] = onehot_from_state(Larena, agent_state)[int(pred)]  # Did we get it right?

        return np.mean(means), np.mean(preds), np.mean(all_actions), np.mean(firstrews)

    return model_properties, environment, model_eval