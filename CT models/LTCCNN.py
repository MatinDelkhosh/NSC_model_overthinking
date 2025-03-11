import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime

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
    def __init__(self, input_size, hidden_size):
        super(LTCCell, self).__init__()
        self.hidden_size = hidden_size
        self.input2hidden = nn.Linear(input_size, hidden_size)
        self.hidden2hidden = nn.Linear(hidden_size, hidden_size)
        self.log_tau = nn.Parameter(torch.zeros(hidden_size))
        self.activation = torch.tanh

    def forward(self, x, h, dt):
        tau = F.softplus(self.log_tau) + 1e-3  # ensure tau > 0
        pre_activation = self.input2hidden(x) + self.hidden2hidden(h)
        h_new = (1 - dt / tau) * h + (dt / tau) * self.activation(pre_activation)
        return h_new

class LTC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LTC, self).__init__()
        self.hidden_size = hidden_size
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
            h = self.cell(x[:, t, :], h, dt=dt)
            out = self.output_layer(h)
            outputs.append(out.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs, h

####################################
#     Putting It All Together
####################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
        self.ltc = LTC(input_size=self.ltc_input_dim, hidden_size=ltc_hidden_size, output_size=ltc_output_dim)

    def forward(self, maze_seq, constants, time_stamps):
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
        outputs, _ = self.ltc(combined_features, time_stamps)
        return outputs

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

# -------------------------
#      Training Loop
# -------------------------

import pickle
train_data_temp = pickle.load(open('Data/processed_data.pk','rb'))
train_data = []
for d in train_data_temp:
    train_data += d

train_dataset = MazeDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)

# Instantiate the network.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MazeSolverNetDynamic(maze_img_size=27, constant_dim=15, cnn_out_dim=16,
                             constant_out_dim=3, ltc_hidden_size=32, ltc_output_dim=3).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
num_epochs = 20

for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (maze_seq, labels, constants, time_stamps) in enumerate(train_loader):
        # maze_seq: (batch, T, 2, 27, 27)
        # labels: (batch, T, 3)
        # constants: (batch, 15)
        # time_stamps: (batch, T)
        maze_seq = maze_seq.to(device)
        labels = labels.to(device)
        constants = constants.to(device)

        optimizer.zero_grad()
        outputs = model(maze_seq, constants, time_stamps)
        print(f'\rforwarded feed {batch_idx}',end='')
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print('\rbackprop',end='')
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"\rEpoch {epoch+1}/{num_epochs} Loss: {avg_loss:.4f}")

print("Training complete.")
torch.save(model, "CT models/maze_solver_full256.pth")
print("Saved the model.")
