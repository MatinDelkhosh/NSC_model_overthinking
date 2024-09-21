class EnvironmentDimensions:
    def __init__(self, Nstates, Nstate_rep, Naction, T, Larena):
        self.Nstates = Nstates
        self.Nstate_rep = Nstate_rep
        self.Naction = Naction
        self.T = T
        self.Larena = Larena

class Environment:
    def __init__(self, initialize_func, step_func, dimensions):
        self.initialize = initialize_func  # A function for initialization
        self.step = step_func  # A function for stepping through the environment
        self.dimensions = dimensions  # An instance of EnvironmentDimensions

class WorldState:
    def __init__(self, agent_state=None, environment_state=None, planning_state=None):
        self.agent_state = agent_state
        self.environment_state = environment_state
        self.planning_state = planning_state
