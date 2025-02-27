import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime

####################################
# 1. Maze Configuration Preprocessing
####################################

def process_maze_config(maze_config):
    """
    Convert a maze configuration (a list-of-lists) into a torch tensor.
    Here we assume:
      - maze_config is a 2D list/array with values 0 and 1.
      - We add a channel dimension.
    """
    maze_np = np.array(maze_config, dtype=np.float32)  # shape: (H, W)
    # Normalize if needed. For example, you can keep 0/1 or scale to 0-1.
    maze_tensor = torch.from_numpy(maze_np).unsqueeze(0)  # shape: (1, H, W)
    return maze_tensor

####################################
# 2. Movement Log Preprocessing
####################################

def process_movement_log(movement_log, time_step=0.03):
    """
    Process a movement log to obtain:
      - a list/array of time indices (or time differences in steps)
      - a one-hot encoding (or label) for the directions.

    For this example, we define three groups:
      * up/down  -> index 0
      * left/right -> index 1
      * reset (or special commands) -> index 2

    We will map the "direction" string accordingly.
    Adjust this mapping as needed.
    """
    # Define a mapping for directions.
    direction_map = {
        "start": None,  # you may choose to ignore the start marker
        "Up": 0,
        "Down": 0,
        "Left": 1,
        "Right": 1,
        "win": None,  # may be terminal, ignore in training input
        "reset": 2  # if you have a reset command
    }
    
    # Convert timestamp strings to datetime objects.
    # Assume ISO format, e.g., "2025-02-25T09:39:23.771Z" (we remove the trailing Z)
    def parse_time(ts):
        return datetime.fromisoformat(ts.replace("Z", ""))
    
    # Filter out commands we don't want to use (e.g., "start" or "win")
    valid_entries = []
    for entry in movement_log:
        if entry["direction"] in ["start", "win"]:
            continue
        # Only include if valid==True (or you could incorporate invalid moves differently)
        if not entry["valid"]:
            continue
        valid_entries.append(entry)
    
    if not valid_entries:
        raise ValueError("No valid movement commands found.")
    
    # Calculate relative time in seconds (from the first valid entry)
    base_time = parse_time(valid_entries[0]["time"])
    time_steps = []
    direction_labels = []
    for entry in valid_entries:
        t = parse_time(entry["time"])
        # difference in seconds divided by time_step gives an approximate step index.
        step_index = (t - base_time).total_seconds() / time_step
        time_steps.append(step_index)
        # Map the direction string to our index.
        d = entry["direction"]
        if d not in direction_map or direction_map[d] is None:
            # For safety, you might choose to skip or assign a default value.
            continue
        direction_labels.append(direction_map[d])
    
    # For a simple training scenario, you might want to create a tensor of labels.
    # For example, one-hot encode them into a size 3 vector.
    labels = []
    for label in direction_labels:
        one_hot = [0, 0, 0]
        one_hot[label] = 1
        labels.append(one_hot)
    
    labels = torch.tensor(labels, dtype=torch.float32)  # shape: (seq_len, 3)
    time_steps = torch.tensor(time_steps, dtype=torch.float32)  # shape: (seq_len,)
    
    return time_steps, labels

####################################
# 3. The Neural Network Architecture
####################################

# (Reusing the MazeCNN, ConstantsNet, LTCCell, LTC, and MazeSolverNet definitions from before.)
# We now add an option to process maze config from a single channel image.

class MazeCNN(nn.Module):
    def __init__(self, output_features, input_channels=3, img_size=64):
        """
        Processes a maze configuration image.
        Args:
            output_features (int): size of the output feature vector.
            input_channels (int): number of channels in the input image.
            img_size (int): assumed square image size.
        """
        super(MazeCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # reduces size by half
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        conv_output_size = 64 * (img_size // 8) * (img_size // 8)
        self.fc = nn.Linear(conv_output_size, output_features)

    def forward(self, x):
        # x: (batch, channels, height, width)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ConstantsNet(nn.Module):
    def __init__(self, input_dim=15, output_dim=3):
        """
        Reduces a 15-dimensional constant input to a lower representation.
        """
        super(ConstantsNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.net(x)

class LTCCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        A simple LTC cell updating its hidden state.
        """
        super(LTCCell, self).__init__()
        self.hidden_size = hidden_size
        self.input2hidden = nn.Linear(input_size, hidden_size)
        self.hidden2hidden = nn.Linear(hidden_size, hidden_size)
        self.log_tau = nn.Parameter(torch.zeros(hidden_size))
        self.activation = torch.tanh

    def forward(self, x, h):
        tau = F.softplus(self.log_tau) + 1e-3  # ensure tau > 0
        dt = 0.01  # can be scaled with the actual dt if desired
        pre_activation = self.input2hidden(x) + self.hidden2hidden(h)
        h_new = (1 - dt / tau) * h + (dt / tau) * self.activation(pre_activation)
        return h_new

class LTC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Processes a sequence input with an LTC cell.
        """
        super(LTC, self).__init__()
        self.hidden_size = hidden_size
        self.cell = LTCCell(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        """
        Args:
            x: (batch, seq_len, input_size)
            h0: initial hidden state
        Returns:
            outputs: (batch, seq_len, output_size)
            h: final hidden state
        """
        batch_size, seq_len, _ = x.size()
        if h0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h = h0
        outputs = []
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
            out = self.output_layer(h)
            outputs.append(out.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs, h

class MazeSolverNet(nn.Module):
    def __init__(self,
                 maze_img_size=64,
                 constant_dim=15,
                 cnn_out_dim=16,
                 constant_out_dim=3,
                 ltc_hidden_size=32,
                 ltc_output_dim=3):
        """
        Combines the MazeCNN, ConstantsNet, and LTC network.
        """
        super(MazeSolverNet, self).__init__()
        self.maze_cnn = MazeCNN(output_features=cnn_out_dim, input_channels=1, img_size=maze_img_size)
        self.const_net = ConstantsNet(input_dim=constant_dim, output_dim=constant_out_dim)
        self.ltc_input_dim = cnn_out_dim + constant_out_dim
        self.ltc = LTC(input_size=self.ltc_input_dim, hidden_size=ltc_hidden_size, output_size=ltc_output_dim)

    def forward(self, maze_img, constants, seq_len=10):
        """
        Args:
            maze_img: (batch, channels, height, width)
            constants: (batch, constant_dim)
            seq_len: number of time steps for the LTC network.
        Returns:
            outputs: (batch, seq_len, 3)
        """
        cnn_features = self.maze_cnn(maze_img)       # (batch, cnn_out_dim)
        constant_features = self.const_net(constants)  # (batch, constant_out_dim)
        combined_features = torch.cat([cnn_features, constant_features], dim=1)  # (batch, ltc_input_dim)
        ltc_input = combined_features.unsqueeze(1).repeat(1, seq_len, 1)
        outputs, _ = self.ltc(ltc_input)
        return outputs

####################################
# 4. Example Usage Putting It All Together
####################################

if __name__ == '__main__':
    # Example maze configuration as provided.
    maze_config = [
        [1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,0,1],
        [0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,1],
        [1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0],
        [1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1],
        [1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1],
        [1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,0,1,0,1,0,1],
        [1,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0],
        [1,0,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1],
        [1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0],
        [1,1,1,1,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1],
        [0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,0,1,0,0,0,0],
        [1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1],
        [0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,1],
        [1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1],
        [1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,1],
        [1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,0,1,1,1,0,1],
        [0,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1],
        [1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,1,1,1,1,1,1,0,1],
        [0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
        [1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,1,1],
        [0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1],
        [1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,0,0,1],
        [1,1,1,1,1,0,1,1,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1],
        [1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1],
        [1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1]
    ]
    
    # Process the maze into a tensor.
    maze_tensor = process_maze_config(maze_config)  # shape: (1, H, W)
    # For our network we need a batch dimension. For example, we use batch_size=1.
    maze_tensor = maze_tensor.unsqueeze(0)  # shape: (1, 1, H, W)
    
    # Suppose we also have 15 constant numerical inputs.
    dummy_constants = torch.randn(1, 15)
    
    # Create an instance of MazeSolverNet.
    net = MazeSolverNet(maze_img_size=64, constant_dim=15, cnn_out_dim=16,
                        constant_out_dim=3, ltc_hidden_size=32, ltc_output_dim=3)
    
    # Now, assume we have a movement log (as provided).
    movement_log = [
        {"direction":"start","valid":True,"time":"2025-02-25T09:39:23.771Z"},
        {"direction":"Right","valid":True,"time":"2025-02-25T09:39:25.576Z"},
        {"direction":"Right","valid":True,"time":"2025-02-25T09:39:25.756Z"},
        {"direction":"Down","valid":True,"time":"2025-02-25T09:39:25.892Z"},
        {"direction":"Down","valid":True,"time":"2025-02-25T09:39:26.094Z"},
        {"direction":"Down","valid":False,"time":"2025-02-25T09:39:26.270Z"},
        {"direction":"Right","valid":True,"time":"2025-02-25T09:39:26.339Z"},
        # ... (continue for the rest of your log)
        {"direction":"win","valid":True,"time":"2025-02-25T09:39:35.441Z"}
    ]
    
    # Process the movement log.
    time_steps, labels = process_movement_log(movement_log, time_step=0.03)
    print("Time steps (in steps):", time_steps)
    print("Labels shape:", labels.shape)
    
    # Suppose for training we want the network to output a sequence that has the same
    # number of time steps as the valid movement commands. For example:
    seq_len = labels.shape[0]
    
    # Forward pass.
    outputs = net(maze_tensor, dummy_constants, seq_len=seq_len)
    print("Network output shape:", outputs.shape)  # Expected: (batch, seq_len, 3)
    
    # Here, you could then compute a loss between outputs and your processed labels,
    # for example using MSELoss or CrossEntropyLoss (with suitable modifications).
