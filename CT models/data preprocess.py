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

def preprocess_movement_data(maze_config, movement_log, constants,
                             time_resolution=0.01, command_duration=0.05,
                             maze_size=(27, 27)):
    """
    Preprocess a single data instance.
    
    Args:
        maze_config: 2D list or array of size maze_size (e.g. 27x27) representing the maze layout.
                     (For example, 1 for wall and 0 for free path.)
        movement_log: List of dictionaries, each with keys:
                      "direction" (str), "valid" (bool), "time" (ISO datetime string).
                      E.g., {"direction": "Right", "valid": True, "time": "2025-02-25T09:39:25.576Z"}
        constants: 15-element list or array (constant inputs for this sample).
        time_resolution: Sampling interval in seconds (default 0.01s).
        command_duration: Duration in seconds for which a command label should be active (default 0.03s).
        maze_size: Tuple with maze dimensions (default (27,27)).
        
    Returns:
        input_sequence: np.array of shape (T, 2, maze_size[0], maze_size[1])
                        where channel 0 is the maze layout (repeated) and channel 1 is the one-hot agent location.
        labels: np.array of shape (T, 3) with command labels.
        constants: np.array of shape (15,) (unchanged).
    """
    # Convert maze layout to numpy array (assume it's already maze_size, e.g., 27x27)
    maze_array = np.array(maze_config, dtype=np.float32)
    
    # Determine duration from movement log based on timestamps.
    # Parse all timestamps (ignoring "start" and "win" for command generation but using them for timeline).
    events = []
    for entry in movement_log:
        # We include all entries in timeline but only assign a command vector for certain directions.
        # (For "start" and "win", command will be [0,0,0].)
        ts_str = entry["time"].replace("Z", "")  # remove trailing Z if present
        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception as e:
            raise ValueError(f"Error parsing timestamp {ts_str}: {e}")
        print(f'\r{entry}',end='')
        try: events.append({
            "time": ts,
            "direction": entry["direction"],
            "valid": entry["valid"]
        })
        except: print('\nsth wrong')
    
    if len(events) == 0:
        raise ValueError("Movement log is empty.")
    
    # Use the time of the first event as base time.
    base_time = events[0]["time"]
    
    # Determine sample indices for events.
    duration_samples = int(round(command_duration / time_resolution))  # e.g., 3 samples
    event_samples = []
    for ev in events:
        # Compute time difference in seconds.
        t_diff = (ev["time"] - base_time).total_seconds()
        sample_index = int(round(t_diff / time_resolution))
        ev["sample_index"] = sample_index
        event_samples.append(sample_index)
    
    # Determine total number of time samples: from 0 to max event index + command_duration duration.
    T = max(event_samples) + duration_samples
    # Initialize label sequence with zeros.
    labels = np.zeros((T, 3), dtype=np.float32)
    
    # Prepare a dictionary mapping sample index to a list of events that occur at that sample.
    events_dict = {}
    for ev in events:
        idx = ev["sample_index"]
        if idx not in events_dict:
            events_dict[idx] = []
        events_dict[idx].append(ev)
    
    # Initialize agent position timeline.
    # Agent starts at top-left corner: (0, 0).
    agent_positions = np.zeros((T, 2), dtype=np.int32)
    current_agent = (0, 0)
    
    # For each time step, if an event occurs at that sample, assign command labels (for command_duration)
    # and update the agent position if the event is valid.
    for t in range(T):
        if t in events_dict:
            # Process events in order.
            for ev in events_dict[t]:
                # Determine command vector (for valid movement commands).
                # For "start" and "win", we leave the label as zeros.
                d_lower = ev["direction"].lower()
                if d_lower in ["up", "down", "left", "right", "reset"]:
                    cmd = get_command_vector(ev["direction"])
                else:
                    cmd = np.array([0, 0, 0], dtype=np.float32)
                # Set label for the duration of the command (overlapping commands override earlier ones).
                for dt in range(duration_samples):
                    if t + dt < T:
                        labels[t + dt] = cmd
                # Update agent position only if the event is valid.
                if ev["valid"] and d_lower in ["up", "down", "left", "right", "reset"]:
                    new_agent = update_agent_position(current_agent, ev["direction"])
                    # Optionally, you can add boundary checking here to ensure the new position is within maze bounds.
                    # For example:
                    r, c = new_agent
                    if 0 <= r < maze_size[0] and 0 <= c < maze_size[1]:
                        current_agent = new_agent
                    # If out-of-bounds, you might choose to ignore the update.
        # For each time step, record the current agent position.
        agent_positions[t] = current_agent
    
    # Now, build the input sequence.
    # For each time step, create a 2-channel image of shape (2, maze_size[0], maze_size[1]):
    # - Channel 0: constant maze layout.
    # - Channel 1: one-hot encoding of the agent position.
    input_sequence = []
    # Pre-cast maze channel once.
    maze_channel = maze_array.copy()  # shape: (27,27)
    
    for t in range(T):
        # Create agent channel as zeros.
        agent_channel = np.zeros(maze_size, dtype=np.float32)
        r, c = agent_positions[t]
        # Make sure the agent's position is within bounds.
        if 0 <= r < maze_size[0] and 0 <= c < maze_size[1]:
            agent_channel[r, c] = 1.0
        # Stack to create a 2-channel image.
        # We use channels-first convention: shape (2, H, W)
        img = np.stack([maze_channel, agent_channel], axis=0)
        input_sequence.append(img)
    
    input_sequence = np.array(input_sequence, dtype=np.float32)  # shape: (T, 2, maze_size[0], maze_size[1])
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