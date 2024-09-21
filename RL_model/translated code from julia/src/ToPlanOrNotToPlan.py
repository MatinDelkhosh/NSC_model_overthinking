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