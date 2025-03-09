import numpy as np
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def get_command_vector(direction):
    """
    Maps direction string to a 3-element label vector.
    Up/Down affect neuron 1, Right/Left affect neuron 2, reset affects neuron 3.
    """
    d = direction.lower()
    if d == "up":
        return np.array([1, 0, 0], dtype=np.float32)
    elif d == "down":
        return np.array([-1, 0, 0], dtype=np.float32)
    elif d == "right":
        return np.array([0, -1, 0], dtype=np.float32)
    elif d == "left":
        return np.array([0, 1, 0], dtype=np.float32)
    elif d == "reset":
        return np.array([0, 0, 1], dtype=np.float32)
    else:
        return np.array([0, 0, 0], dtype=np.float32)

def update_agent_position(current, direction):
    """
    Update agent position given the current (row, col) and direction.
    Uses standard grid movement (ignores label sign differences).
    """
    d = direction.lower()
    row, col = current
    if d == "up":
        return (row - 1, col)
    elif d == "down":
        return (row + 1, col)
    elif d == "right":
        return (row, col + 1)
    elif d == "left":
        return (row, col - 1)
    elif d == "reset":
        return (0, 0)
    else:
        return current

# --- Preprocessing Function ---
def preprocess_movement_data(maze_config, movement_log, constants,
                             command_duration=0.05,
                             maze_size=(27, 27)):
    """
    Preprocess a single data instance.

    For each movement event (ignoring "start" and "win"), three samples are generated:
      1. Pre-command: label = [0,0,0], agent position unchanged.
      2. At command: label = command vector (e.g., [1,0,0] for "Up"), agent position unchanged.
      3. Post-command: label = [0,0,0] and if the move is valid, update the agent position.

    The maze configuration (channel 0) is constant; channel 1 is a one-hot agent location.
    The same two-channel image is used for all three samples except that after the move (sample 3)
    the agent's location is updated (if the move was valid).

    Args:
        maze_config: 2D list/array (maze_size) representing the maze layout (e.g., 1 for wall, 0 for free).
        movement_log: List of dictionaries with keys "direction" (str), "valid" (bool), and "time" (ISO datetime string).
        constants: 15-element list/array (constant inputs for this sample).
        command_duration: Duration in seconds for a command event (default 0.05).
        maze_size: Tuple with maze dimensions (default (27, 27)).

    Returns:
        input_sequence: np.array of shape (T, 2, maze_size[0], maze_size[1]),
                        where channel 0 is the maze layout and channel 1 is the one-hot agent location.
        labels: np.array of shape (T, 3) with command labels.
        constants: np.array of shape (15,) (unchanged).
    """
    # Convert maze layout to numpy array (assume it's maze_size)
    maze_array = np.array(maze_config, dtype=np.float32)

    # Parse movement events (include all events for timeline but only generate command for moves)
    events = []
    for entry in movement_log:
        # Skip "start" and "win" events in command generation (they will yield zero labels)
        d_lower = entry["direction"].lower()
        if d_lower in ["start", "win"]:
            continue
        ts_str = entry["time"].replace("Z", "")  # remove trailing Z
        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception as e:
            raise ValueError(f"Error parsing timestamp {ts_str}: {e}")
        events.append({
            "time": ts,
            "direction": entry["direction"],
            "valid": entry["valid"]
        })

    if len(events) == 0:
        raise ValueError("No movement events to process.")

    # For this triangle shape, each event produces 3 samples.
    # We ignore the actual timestamp spacing here and simply output 3 samples per event.
    num_events = len(events)
    T = num_events * 3

    # Initialize labels: for each event, sample 1 and sample 3 will be zeros, sample 2 is the command.
    labels = np.zeros((T, 3), dtype=np.float32)
    # Initialize agent position timeline.
    # Agent starts at (0, 0)
    agent_positions = np.zeros((T, 2), dtype=np.int32)
    current_agent = (0, 0)

    # We'll also build the input_sequence as we go.
    input_sequence = []
    # Pre-cast the constant maze layout (channel 0)
    maze_channel = maze_array.copy()  # shape: (27,27)

    # For each event, create 3 samples.
    for i, ev in enumerate(events):
        # Determine command vector for the event.
        cmd_vector = get_command_vector(ev["direction"])  # e.g., [1,0,0] for Up, etc.
        base_index = i * 3

        # Sample 1 (pre-command): label is zeros; agent position is current.
        labels[base_index] = np.array([0, 0, 0], dtype=np.float32)
        # Sample 2 (at command): label is the command vector; agent position still unchanged.
        labels[base_index + 1] = cmd_vector
        # Sample 3 (post-command): label is zeros.
        labels[base_index + 2] = np.array([0, 0, 0], dtype=np.float32)

        # For samples 1 and 2, agent position remains the same.
        for j in range(2):
            agent_positions[base_index + j] = current_agent

        # For sample 3, update agent position if the movement is valid.
        d_lower = ev["direction"].lower()
        if ev["valid"] and d_lower in ["up", "down", "left", "right", "reset"]:
            new_agent = update_agent_position(current_agent, ev["direction"])
            r, c = new_agent
            # Optional: boundary check for maze_size.
            if 0 <= r < maze_size[0] and 0 <= c < maze_size[1]:
                current_agent = new_agent
        agent_positions[base_index + 2] = current_agent

        # Build input images for these three samples.
        for j in range(3):
            # Create the agent channel: one-hot of the current agent position.
            # Note: for samples 1 and 2, use the old position; for sample 3, use updated position.
            if j < 2:
                pos = agent_positions[base_index]  # unchanged
            else:
                pos = agent_positions[base_index + 2]  # updated position
            agent_channel = np.zeros(maze_size, dtype=np.float32)
            r, c = pos
            if 0 <= r < maze_size[0] and 0 <= c < maze_size[1]:
                agent_channel[r, c] = 1.0
            # Stack channels: channel 0 = maze layout, channel 1 = agent location.
            img = np.stack([maze_channel, agent_channel], axis=0)
            input_sequence.append(img)

    input_sequence = np.array(input_sequence, dtype=np.float32)  # shape: (T, 2, 27, 27)
    constants = np.array(constants, dtype=np.float32)  # shape: (15,)

    return input_sequence, labels, constants

from matplotlib.animation import FuncAnimation

def visualize_processed_data(input_sequence, labels, time_resolution=0.01):
    """
    Visualize the processed data with animation showing agent movement and neuron activations over time.
    """
    T = input_sequence.shape[0]

    # Extract maze layout and agent positions with NaN handling
    maze_layout = input_sequence[0, 0, :, :]
    agent_positions = []
    for t in range(T):
        pos = np.argwhere(input_sequence[t, 1, :, :] == 1)
        if pos.size > 0:
            agent_positions.append(pos[0].astype(float))
        else:
            agent_positions.append([np.nan, np.nan])  # Ensure consistent format
    agent_positions = np.array(agent_positions)

    # Create time axis
    time_axis = np.arange(T) * time_resolution

    # Set up figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Maze subplot setup
    ax1.imshow(maze_layout, cmap='gray_r')
    path_line, = ax1.plot([], [], 'ro-', markersize=4, label="Agent Path")
    current_pos, = ax1.plot([], [], 'bo', markersize=8, label="Current Position")  # Initialize with empty lists
    ax1.set_title("Maze Layout and Agent Path")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")
    ax1.legend()
    ax1.invert_yaxis()

    # Neuron activations subplot setup
    line_up, = ax2.plot([], [], 'blue', label="Up/Down")
    line_left, = ax2.plot([], [], 'green', label="Left/Right")
    line_reset, = ax2.plot([], [], 'red', label="Reset")
    ax2.set_title("Output Neurons Over Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Neuron Output")
    ax2.set_xlim(0, T*time_resolution)
    ax2.set_ylim(-1.1, 1.1)
    ax2.legend()

    def init():
        """Initialize animation elements"""
        path_line.set_data([], [])
        current_pos.set_data([], [])  # Initialize with empty lists
        line_up.set_data([], [])
        line_left.set_data([], [])
        line_reset.set_data([], [])
        return (path_line, current_pos, line_up, line_left, line_reset)

    def update(frame):
        """Update function for each animation frame"""
        # Get current position coordinates
        x = agent_positions[frame, 1]
        y = agent_positions[frame, 0]
        
        # Update current position (wrap scalars in lists)
        current_pos.set_data([x], [y])  # FIX: Pass coordinates as lists
        
        # Update path (use slicing to get sequences)
        path_line.set_data(agent_positions[:frame+1, 1], agent_positions[:frame+1, 0])
        
        # Update neuron plots
        line_up.set_data(time_axis[:frame+1], labels[:frame+1, 0])
        line_left.set_data(time_axis[:frame+1], labels[:frame+1, 1])
        line_reset.set_data(time_axis[:frame+1], labels[:frame+1, 2])

        return (path_line, current_pos, line_up, line_left, line_reset)

    # Create animation
    ani = FuncAnimation(fig, update, frames=T, init_func=init,
                        blit=True, interval=time_resolution*1000, repeat=False)

    plt.tight_layout()
    plt.show()
      
# --- Example usage ---
if __name__ == '__main__':
    #df = pd.read_excel('D:\Matin\stuff\NSC\data\test export\Single Sheet data.xlsx')

    # Example 27x27 maze configuration (for brevity, here we create a dummy maze).
    # In practice, provide your own 27x27 maze (list of lists of 0/1).
    maze_config = [[1,0,1,0,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1],
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

    # Example movement log.
    movement_log = [{"direction":"start","valid":True,"time":"2025-02-28T21:34:01.852Z"},
                    {"direction":"Down","valid":True,"time":"2025-02-28T21:34:04.581Z"},
                    {"direction":"Right","valid":False,"time":"2025-02-28T21:34:05.134Z"},
                    {"direction":"Down","valid":True,"time":"2025-02-28T21:34:24.304Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:24.722Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:25.002Z"},
                    {"direction":"Down","valid":True,"time":"2025-02-28T21:34:25.320Z"},
                    {"direction":"Down","valid":True,"time":"2025-02-28T21:34:25.668Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:26.069Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:26.301Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:26.551Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:27.100Z"},
                    {"direction":"Down","valid":True,"time":"2025-02-28T21:34:27.453Z"},
                    {"direction":"Down","valid":True,"time":"2025-02-28T21:34:28.016Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:28.369Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:28.565Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:28.749Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:28.951Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:29.183Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:29.700Z"},
                    {"direction":"Up","valid":True,"time":"2025-02-28T21:34:31.033Z"},
                    {"direction":"Up","valid":True,"time":"2025-02-28T21:34:31.482Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:31.900Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:32.150Z"},
                    {"direction":"Up","valid":True,"time":"2025-02-28T21:34:32.765Z"},
                    {"direction":"Up","valid":True,"time":"2025-02-28T21:34:33.115Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:33.681Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:34.013Z"},
                    {"direction":"Down","valid":True,"time":"2025-02-28T21:34:34.564Z"},
                    {"direction":"Down","valid":True,"time":"2025-02-28T21:34:34.829Z"},
                    {"direction":"Down","valid":True,"time":"2025-02-28T21:34:35.013Z"},
                    {"direction":"Down","valid":True,"time":"2025-02-28T21:34:35.212Z"},
                    {"direction":"Down","valid":True,"time":"2025-02-28T21:34:35.395Z"},
                    {"direction":"Down","valid":True,"time":"2025-02-28T21:34:35.593Z"},
                    {"direction":"Left","valid":True,"time":"2025-02-28T21:34:36.295Z"},
                    {"direction":"Left","valid":True,"time":"2025-02-28T21:34:36.526Z"},
                    {"direction":"Down","valid":True,"time":"2025-02-28T21:34:36.979Z"},
                    {"direction":"Down","valid":True,"time":"2025-02-28T21:34:37.592Z"},
                    {"direction":"Left","valid":True,"time":"2025-02-28T21:34:37.912Z"},
                    {"direction":"Left","valid":True,"time":"2025-02-28T21:34:38.110Z"},
                    {"direction":"Left","valid":True,"time":"2025-02-28T21:34:38.324Z"},
                    {"direction":"Left","valid":True,"time":"2025-02-28T21:34:38.642Z"},
                    {"direction":"Down","valid":True,"time":"2025-02-28T21:34:39.011Z"},
                    {"direction":"Down","valid":True,"time":"2025-02-28T21:34:39.208Z"},
                    {"direction":"Down","valid":True,"time":"2025-02-28T21:34:39.375Z"},
                    {"direction":"Down","valid":True,"time":"2025-02-28T21:34:39.675Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:44.012Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:44.199Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:44.381Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:44.579Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:44.782Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:45.359Z"},
                    {"direction":"Down","valid":True,"time":"2025-02-28T21:34:45.679Z"},
                    {"direction":"Down","valid":True,"time":"2025-02-28T21:34:45.926Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:46.412Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:46.610Z"},
                    {"direction":"Up","valid":True,"time":"2025-02-28T21:34:47.045Z"},
                    {"direction":"Up","valid":True,"time":"2025-02-28T21:34:47.244Z"},
                    {"direction":"Up","valid":True,"time":"2025-02-28T21:34:47.409Z"},
                    {"direction":"Up","valid":True,"time":"2025-02-28T21:34:47.610Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:47.977Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:48.160Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:48.326Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:48.510Z"},
                    {"direction":"Right","valid":True,"time":"2025-02-28T21:34:48.709Z"}]
    
    # Example 15 constant inputs (dummy data).
    constants = np.random.randn(15).tolist()
    data = open('CT models/saveddata.pk','rb')
    data_load = pickle.load(data)
    data.close()

    processed_data = []

    for d in data_load:
        processed_data.append([])
        for constants,maze_config,movement_log in d:
            # Preprocess the data.
            inputs_seq, labels_seq, const_arr = preprocess_movement_data(maze_config, movement_log, constants)
            processed_data[-1].append((inputs_seq,labels_seq,const_arr))

    with open('processed_data.pk','ab') as data_dump:
        pickle.dump(processed_data, data_dump)
    
    #print("Input sequence shape:", inputs_seq.shape)  # (T, 2, 27, 27)
    #print("Labels sequence shape:", labels_seq.shape)  # (T, 3)
    #print("Constants shape:", const_arr.shape)         # (15,)

    # Visualize the processed data
    visualize_processed_data(inputs_seq, labels_seq)