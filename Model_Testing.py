import torch
import numpy as np
import random
from copy import copy
from CT_models.LTCCNN import *

model = torch.load("CT_models/maze_solver_final.pth", weights_only=False)

constants = np.random.randn(15).tolist()
constants = torch.tensor([constants], dtype=torch.float32)
maze_map = [[1,0,1,0,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1],
            [1,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0],
            [1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1],
            [0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1],
            [1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1],
            [1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1],
            [1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1],
            [1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0],
            [1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1],
            [0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0],
            [1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0],
            [1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,1,1,1,1],
            [1,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1],
            [1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1],
            [1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,0,1,1,1,0,1,1,1,1,1],
            [1,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0],
            [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,1,1],
            [0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,1],
            [1,0,1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1],
            [1,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,1],
            [1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1],
            [0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,1,0,0],
            [1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1]]
maze_map = np.array(maze_map, dtype=np.float32)
dt = 0.02
dt = torch.tensor([[dt]], dtype=torch.float32)
#####################################
#######     Maze Updater    #########
#####################################

import random

class KruskalMaze:
    def __init__(self, size=(27,27)):
        self.size = size
        self.maze = [[0 for _ in range(size[1])] for _ in range(size[0])]
        self.sets = {}
        self.player = {'x': 0, 'y': 0}
        self.timer = 0
        self.max_move_speed = 0.03

    def find_set(self, cell):
        if self.sets[cell] != cell:
            self.sets[cell] = self.find_set(self.sets[cell])
        return self.sets[cell]

    def union_sets(self, cell1, cell2):
        self.sets[self.find_set(cell2)] = self.find_set(cell1)

    def generate_maze(self):
        for y in range(0, self.size[1], 2):
            for x in range(0, self.size[0], 2):
                cell = (x, y)
                self.sets[cell] = cell
                self.maze[y][x] = 1
        
        walls = []
        for y in range(0, self.size[1], 2):
            for x in range(0, self.size[0], 2):
                if x + 2 < self.size[0]:
                    walls.append(((x, y), (x + 2, y)))
                if y + 2 < self.size[1]:
                    walls.append(((x, y), (x, y + 2)))
        
        random.shuffle(walls)
        
        for (x1, y1), (x2, y2) in walls:
            set1 = self.find_set((x1, y1))
            set2 = self.find_set((x2, y2))
            
            if set1 != set2:
                self.union_sets(set1, set2)
                self.maze[y1][x1] = 1
                self.maze[y2][x2] = 1
                self.maze[(y1 + y2) // 2][(x1 + x2) // 2] = 1
        
        self.maze[0][0] = 1
        self.maze[self.size[1] - 1][self.size[0] - 1] = 1

    def start_game(self, maze=None):
        if maze is None: self.generate_maze()
        else: self.maze = maze.copy()
        self.player = {'x': 0, 'y': 0}
        self.timer = 0
        self.move_difference = 0

    def move_player(self, direction, dt=0.02):
        dx, dy = 0, 0
        if direction == "up":
            dy = -1
        elif direction == "down":
            dy = 1
        elif direction == "left":
            dx = -1
        elif direction == "right":
            dx = 1
        elif direction == "none":
            pass

        self.timer += dt
        self.move_difference += dt

        new_x = self.player['x'] + dx
        new_y = self.player['y'] + dy

        if 0 <= new_x < self.size[0] and 0 <= new_y < self.size[1] and self.maze[new_x][new_y] == 1. and self.move_difference >= self.max_move_speed and direction != "none":
            self.move_difference = 0
            self.player['x'] = new_x
            self.player['y'] = new_y
            return direction, True
        else:
            return direction, False
    
    def check_win(self):
        return self.player['x'] == self.size[0] - 1 and self.player['y'] == self.size[1] - 1
    
    def get_conv_channels(self):
        agent_channel = np.zeros(self.size, dtype=np.float32)
        maze_channel = np.array(self.maze, dtype=np.float32)
        r, c = self.player.values()
        if 0 <= r < self.size[0] and 0 <= c < self.size[1]:
            agent_channel[r, c] = 1.0
        img = np.stack([maze_channel, agent_channel], axis=0)
        return img

    def get_command_vector(self, neuron_state):
        """
        Maps the given neuron state to the closest predefined direction based on Euclidean distance.
        """
        d = neuron_state.detach().numpy()

        directions = {
            "up":       np.array([1, 0, 0], dtype=np.float32),
            "down":     np.array([-1,0, 0], dtype=np.float32),
            "right":    np.array([0,-1, 0], dtype=np.float32),
            "left":     np.array([0, 1, 0], dtype=np.float32),
            "reset":    np.array([0, 0, 1], dtype=np.float32),
            "none":     np.array([0, 0, 0], dtype=np.float32)
        }

        # Compute distances and find the closest match
        closest_direction = min(directions.keys(), key=lambda dir: np.linalg.norm(d - directions[dir]))

        return closest_direction
    
    def get_command_vector_probablity(self, neuron_state):
        """
        Maps the given neuron state to a command based on probability.
        Closer directions have a higher probability of being chosen.
        USE WHEN dt == max_move_speed!!
        """
        d = neuron_state.detach().numpy()

        directions = {
            "up":       np.array([1, 0, 0], dtype=np.float32),
            "down":     np.array([-1,0, 0], dtype=np.float32),
            "right":    np.array([0,-1, 0], dtype=np.float32),
            "left":     np.array([0, 1, 0], dtype=np.float32),
            "reset":    np.array([0, 0, 1], dtype=np.float32),
            "none":     np.array([0, 0, 0], dtype=np.float32)
        }

        # Compute Euclidean distances
        distances = {dir: np.linalg.norm(d - vec) for dir, vec in directions.items()}

        # Convert distances to probabilities (closer → higher probability)
        inv_distances = {dir: 1 / (dist + 1e-6) for dir, dist in distances.items()}  # Avoid division by zero
        exp_values = np.array(list(inv_distances.values()))
        probabilities = exp_values / np.sum(exp_values)  # Normalize into probabilities

        # Choose based on probability
        chosen_command = np.random.choice(list(directions.keys()), p=probabilities)

        return chosen_command

#####################################
######     Using the Model    #######
#####################################

output_seq = []
h_seq = []
maze_seq = []
time_stamps = []

Maze_Env = KruskalMaze()
Maze_Env.start_game(maze_map)

model.eval()
win = False
h1,h2 = None, None
move_count = 0
while not win:
    img = Maze_Env.get_conv_channels()
    maze_seq.append(img.copy())

    outputs,h1,h2 = model(torch.tensor([[img.copy()]], dtype=torch.float32), constants, dt, h1, h2)
    Maze_Env.move_player(Maze_Env.get_command_vector_probablity(outputs),dt)
    win = Maze_Env.check_win()

    output_seq.append(outputs.detach().numpy().reshape(1,3))
    h_seq.append((h1.detach().numpy(),h2.detach().numpy()))
    time_stamps.append(float(Maze_Env.timer))
    if move_count > 1000: break
    else: move_count += 1
    print(f'\r{move_count}',end='')

    
from CT_models.data_preprocess import visualize_processed_data

visualize_processed_data(np.array(maze_seq), np.array(output_seq).reshape((len(time_stamps),3)), np.array(time_stamps))

# -----------------------
# Plot the hidden state values
# -----------------------
import matplotlib.pyplot as plt
# Convert list of hidden states into numpy arrays.
# h_seq is a list of tuples: (h1, h2)
h1_seq = np.array([h[0] for h in h_seq])  # shape: (steps, hidden_dim)
h2_seq = np.array([h[1] for h in h_seq])  # shape: (steps, hidden_dim)
time_steps = np.array(time_stamps)

# Plotting for h1 hidden layer.
plt.figure(figsize=(12, 6))
for neuron_idx in range(h1_seq.shape[1]):
    plt.plot(time_steps, h1_seq[:, neuron_idx], label=f'h1 neuron {neuron_idx}')
plt.xlabel("Time")
plt.ylabel("Hidden state value")

for neuron_idx in range(h2_seq.shape[1]):
    plt.plot(time_steps, h2_seq[:, neuron_idx], label=f'h2 neuron {neuron_idx}')
plt.xlabel("Time")
plt.ylabel("Hidden state value")
plt.title("Hidden State Values for h over Time")
plt.show()
