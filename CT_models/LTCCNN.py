import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


####################################
#  The Neural Network Architecture
####################################

class MazeCNN(nn.Module):
    def __init__(self, output_features, input_channels=2, img_size=27):
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
        conv_output_size = 64 * 3 * 3
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
    def __init__(self, input_size, hidden_size, solver_steps=1, max_dt=0.02):
        """
        A simple LTC cell updating its hidden state with a configurable time step.
        If dt is larger than max_dt, multiple Euler ODE steps are taken.
        
        Args:
            input_size: Size of the input.
            hidden_size: Size of the hidden state.
            solver_steps: Number of Euler steps per substep.
            max_dt: Maximum allowed dt per substep.
        """
        super(LTCCell, self).__init__()
        self.hidden_size = hidden_size
        self.input2hidden = nn.Linear(input_size, hidden_size)
        self.hidden2hidden = nn.Linear(hidden_size, hidden_size)

        # Learnable time constant (tau)
        self.log_tau = nn.Parameter(torch.zeros(hidden_size))

        # Number of ODE solver steps per substep.
        self.solver_steps = solver_steps
        
        # Maximum dt allowed per substep.
        self.max_dt = max_dt

        # Activation function parameters (sigmoid-based gating similar to LTC)
        self.mu = nn.Parameter(torch.rand(hidden_size))
        self.sigma = nn.Parameter(torch.rand(hidden_size) * 5)

        # Leak term and capacitance.
        self.gleak = nn.Parameter(torch.ones(hidden_size))
        self.cm_t = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, h, dt):
        """
        Args:
            x: Input vector.
            h: Previous hidden state.
            dt: Total time step to integrate (can be larger than max_dt).
        Returns:
            h_new: Updated hidden state after integrating over dt.
        """
        tau = F.softplus(self.log_tau) + 1e-3  # Ensure tau > 0
        input_effect = self.input2hidden(x)
        
        # Convert dt to a scalar value if it is a tensor.
        dt_val = dt.item() if isinstance(dt, torch.Tensor) else dt
        
        # Compute number of full steps (each of size max_dt) and the remainder.
        n_full_steps = int(dt_val // self.max_dt)
        dt_remainder = dt_val - n_full_steps * self.max_dt

        # First, take full steps using max_dt.
        for _ in range(n_full_steps * self.solver_steps):
            pre_activation = input_effect + self.hidden2hidden(h)
            w_activation = torch.sigmoid((pre_activation - self.mu) * self.sigma)
            h_update = w_activation * pre_activation
            h = (1 - self.max_dt / tau) * h + (self.max_dt / (self.cm_t + self.gleak)) * h_update

        # Then, if there is any remaining time, take a final set of steps with dt = dt_remainder.
        if dt_remainder > 1e-8:
            for _ in range(self.solver_steps):
                pre_activation = input_effect + self.hidden2hidden(h)
                w_activation = torch.sigmoid((pre_activation - self.mu) * self.sigma)
                h_update = w_activation * pre_activation
                h = (1 - dt_remainder / tau) * h + (dt_remainder / (self.cm_t + self.gleak)) * h_update

        return h
    
class LTC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LTC, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cell = LTCCell(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, time_stamps, h0=None):
        batch_size, seq_len, _ = x.size()
        if h0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h = h0
        outputs = []
        # For dt calculation, assume time_stamps is a tensor of shape (seq_len,)
        for t in range(seq_len):
            if t == 0:
                dt = time_stamps[0]  # or assume some initial dt
            else:
                dt = time_stamps[t] - time_stamps[t-1]
            # Ensure dt is a tensor of correct type:
            dt = dt if isinstance(dt, torch.Tensor) else torch.tensor(dt, device=x.device, dtype=x.dtype)
            '''h1 = self.cell(x[:, t, :], h[:,:self.hidden_size], dt=dt)
            h2 = self.output_layer(h1,h[:,self.hidden_size:], dt=dt)
            outputs.append(h2.unsqueeze(1))
            h = torch.cat((h1,h2), dim=1)'''
            h = self.cell(x[:, t, :], h, dt=dt)
            out = self.output_layer(h)
            outputs.append(out.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs, h

####################################
#     Putting It All Together
####################################
# Assume MazeCNN, ConstantsNet, LTCCell, and LTC are defined as before.
# For dynamic input, the MazeCNN is now used on a sequence of images with 2 channels.

class MazeSolverNetDynamic(nn.Module):
    def __init__(self, maze_img_size=27, constant_dim=15, cnn_out_dim=16,
                 constant_out_dim=3, ltc_hidden_size=32, ltc_output_dim=3):
        """
        Processes a dynamic maze sequence and constant inputs.
        maze_seq: (batch, seq_len, 2, H, W) where channel 0 is maze layout,
                  channel 1 is the one-hot agent location.
        constants: (batch, constant_dim)
        """
        super(MazeSolverNetDynamic, self).__init__()
        # Use 2 input channels since the agent location is dynamic.
        self.maze_cnn = MazeCNN(output_features=cnn_out_dim, input_channels=2, img_size=maze_img_size)
        self.const_net = ConstantsNet(input_dim=constant_dim, output_dim=constant_out_dim)
        self.ltc_input_dim = cnn_out_dim + constant_out_dim
        self.ltc = LTC(input_size=self.ltc_input_dim, hidden_size=ltc_hidden_size, output_size=ltc_hidden_size)
        self.ltc2 = LTC(input_size=ltc_hidden_size, hidden_size=ltc_hidden_size, output_size=ltc_output_dim)
        self.tanh = nn.Tanh()

    def forward(self, maze_seq, constants, time_stamps, h1=None, h2=None):
        """
        Args:
            maze_seq: Tensor of shape (batch, seq_len, 2, H, W)
            constants: Tensor of shape (batch, constant_dim)
            time_stamps: Tensor of shape (batch, seq_len) or (seq_len,) if same across batch.
        Returns:
            outputs: Tensor of shape (batch, seq_len, 3)
        """
        batch, seq_len, C, H, W = maze_seq.size()
        # Process each time step individually using the CNN.
        maze_seq = maze_seq.view(batch * seq_len, C, H, W)
        cnn_features = self.maze_cnn(maze_seq)  # (batch * seq_len, cnn_out_dim)
        cnn_features = cnn_features.view(batch, seq_len, -1)
        constant_features = self.const_net(constants)  # (batch, constant_out_dim)
        # Repeat constant features over the time dimension.
        constant_features = constant_features.unsqueeze(1).repeat(1, seq_len, 1)
        combined_features = torch.cat([cnn_features, constant_features], dim=2)  # (batch, seq_len, ltc_input_dim)

        # Assume time_stamps is of shape (batch, seq_len) or (seq_len,). If it has batch dimension,
        # and if they are identical across the batch, we can simply take the first row.
        if time_stamps.dim() == 2:
            time_stamps = time_stamps[0]  # shape: (seq_len,)
        outputs, h_1 = self.ltc(combined_features, time_stamps, h1)
        outputs, h_2 = self.ltc2(outputs, time_stamps, h2)
        outputs = self.tanh(outputs)
        return outputs, h_1, h_2

# -------------------------
#     Dataset Definition
# -------------------------

class MazeDataset(Dataset):
    def __init__(self, data_list):
        """
        data_list: A list where each element is a tuple:
                   (input_sequence, labels, constants, time_stamps)
                   - input_sequence: np.array (T, 2, H, W)
                   - labels: np.array (T, 3)
                   - constants: np.array (15,)
                   - time_stamps: np.array (T,) (optional, for reference)
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        input_seq, labels, constants, time_stamps = self.data_list[idx]
        # Convert all to torch tensors.
        input_seq = torch.tensor(input_seq, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        constants = torch.tensor(constants, dtype=torch.float32)
        time_stamps = torch.tensor(time_stamps, dtype=torch.float32)
        return input_seq, labels, constants, time_stamps

from torch.nn.utils.rnn import pad_sequence

def custom_collate(batch):
    """
    Custom collate function to handle variable-length sequences.
    Each item in batch is a tuple: (input_seq, labels, constants, time_stamps)
    where:
      - input_seq: Tensor of shape (T, 2, 27, 27)
      - labels: Tensor of shape (T, 3)
      - constants: Tensor of shape (15,)
      - time_stamps: Tensor of shape (T,)
    This function pads the input_seq, labels, and time_stamps along the time dimension.
    """
    input_seqs = [torch.tensor(item[0], dtype=torch.float32) for item in batch]  # List of (T_i, 2, 27, 27)
    labels = [torch.tensor(item[1], dtype=torch.float32) for item in batch]      # List of (T_i, 3)
    time_stamps = [torch.tensor(item[3], dtype=torch.float32) for item in batch] # List of (T_i,)
    constants = [torch.tensor(item[2], dtype=torch.float32) for item in batch]   # List of (15,)
    
    # Pad sequences along the time dimension (first dimension).
    input_seqs_padded = pad_sequence(input_seqs, batch_first=True)   # (batch, T_max, 2, 27, 27)
    labels_padded = pad_sequence(labels, batch_first=True)           # (batch, T_max, 3)
    time_stamps_padded = pad_sequence(time_stamps, batch_first=True) # (batch, T_max)
    constants = torch.stack(constants, dim=0)                        # (batch, 15)
    
    return input_seqs_padded, labels_padded, constants, time_stamps_padded