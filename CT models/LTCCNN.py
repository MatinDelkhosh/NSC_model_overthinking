import torch
import torch.nn as nn
import torch.nn.functional as F

##############################
# 1. CNN for Maze Configuration
##############################

class MazeCNN(nn.Module):
    def __init__(self, output_features, input_channels=3, img_size=64):
        """
        Processes a maze configuration image.
        Args:
            output_features (int): size of the output feature vector.
            input_channels (int): number of channels in the input image.
            img_size (int): assumed square image size (e.g., 64x64).
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
        # After 3 poolings the spatial dimension is img_size//8.
        conv_output_size = 64 * (img_size // 8) * (img_size // 8)
        self.fc = nn.Linear(conv_output_size, output_features)

    def forward(self, x):
        # x: (batch, channels, height, width)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x

##############################
# 2. Network for 15 Constant Inputs
##############################

class ConstantsNet(nn.Module):
    def __init__(self, input_dim=15, output_dim=3):
        """
        Reduces a 15-dimensional constant input to a lower (e.g., 3-D) representation.
        """
        super(ConstantsNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, output_dim),
            nn.ReLU()  # you can remove or change the activation as needed
        )
        
    def forward(self, x):
        # x: (batch, input_dim)
        return self.net(x)

##############################
# 3. LTC (Liquid Time-Constant) Network Components
##############################

class LTCCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        A simple LTC cell that updates its hidden state according to:
        
            h_new = (1 - dt/tau) * h + (dt/tau) * tanh( input2hidden(x) + hidden2hidden(h) )
        
        where tau is a learnable time constant.
        """
        super(LTCCell, self).__init__()
        self.hidden_size = hidden_size
        self.input2hidden = nn.Linear(input_size, hidden_size)
        self.hidden2hidden = nn.Linear(hidden_size, hidden_size)
        # Log of tau so we can ensure positivity via softplus.
        self.log_tau = nn.Parameter(torch.zeros(hidden_size))
        self.activation = torch.tanh

    def forward(self, x, h):
        # Compute tau with a softplus to guarantee it’s positive.
        tau = F.softplus(self.log_tau) + 1e-3  # avoid division by zero
        dt = 1.0  # time step; adjust if you have variable dt from timestamps
        pre_activation = self.input2hidden(x) + self.hidden2hidden(h)
        # LTC update: an Euler integration step of the underlying ODE
        h_new = (1 - dt / tau) * h + (dt / tau) * self.activation(pre_activation)
        return h_new

class LTC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Processes a sequence input using the LTCCell.
        """
        super(LTC, self).__init__()
        self.hidden_size = hidden_size
        self.cell = LTCCell(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_size)
            h0: Optional initial hidden state of shape (batch, hidden_size)
        Returns:
            outputs: Tensor of shape (batch, seq_len, output_size)
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

##############################
# 4. Main Maze Solver Network Combining Everything
##############################

class MazeSolverNet(nn.Module):
    def __init__(self,
                 maze_img_size=64,
                 constant_dim=15,
                 cnn_out_dim=16,
                 constant_out_dim=3,
                 ltc_hidden_size=32,
                 ltc_output_dim=3):
        """
        Combines the maze CNN, constants network, and LTC network.
        
        The maze image and constants are processed into feature vectors which are concatenated.
        This combined feature is then repeated (to form a sequence) and passed into the LTC,
        which outputs 3 neurons per time step corresponding to your movement commands.
        """
        super(MazeSolverNet, self).__init__()
        self.maze_cnn = MazeCNN(output_features=cnn_out_dim, input_channels=3, img_size=maze_img_size)
        self.const_net = ConstantsNet(input_dim=constant_dim, output_dim=constant_out_dim)
        # The input to the LTC is the concatenation of the CNN and constant features.
        self.ltc_input_dim = cnn_out_dim + constant_out_dim
        self.ltc = LTC(input_size=self.ltc_input_dim, hidden_size=ltc_hidden_size, output_size=ltc_output_dim)

    def forward(self, maze_img, constants, seq_len=10):
        """
        Args:
            maze_img: Tensor of shape (batch, channels, height, width)
            constants: Tensor of shape (batch, constant_dim)
            seq_len: How many time steps to simulate in the LTC network.
        Returns:
            outputs: Tensor of shape (batch, seq_len, 3)
        """
        # Process the maze image and constant inputs.
        cnn_features = self.maze_cnn(maze_img)            # (batch, cnn_out_dim)
        constant_features = self.const_net(constants)       # (batch, constant_out_dim)
        # Concatenate the two feature vectors.
        combined_features = torch.cat([cnn_features, constant_features], dim=1)  # (batch, ltc_input_dim)
        
        # If the input to LTC is meant to be constant over time,
        # we can repeat the combined feature vector for each time step.
        ltc_input = combined_features.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, ltc_input_dim)
        
        # Pass through the LTC network.
        outputs, _ = self.ltc(ltc_input)
        return outputs

##############################
# Example Usage
##############################

if __name__ == '__main__':
    # Create dummy inputs.
    batch_size = 4
    # Example maze image: assume 3 channels (e.g., maze, agent, goal) and 64x64 pixels.
    dummy_maze = torch.randn(batch_size, 3, 64, 64)
    # 15 constant numerical inputs.
    dummy_constants = torch.randn(batch_size, 15)
    
    # Instantiate the network.
    net = MazeSolverNet(maze_img_size=64, constant_dim=15, cnn_out_dim=16,
                        constant_out_dim=3, ltc_hidden_size=32, ltc_output_dim=3)
    
    # Forward pass with a sequence length of 10 time steps.
    outputs = net(dummy_maze, dummy_constants, seq_len=10)
    print("Output shape:", outputs.shape)  # Expected: (batch_size, 10, 3)
