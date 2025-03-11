import numpy as np
from datetime import datetime
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

# --- Updated Preprocessing Function ---
def preprocess_movement_data(maze_config, movement_log, constants,
                             command_duration=0.05,
                             maze_size=(27, 27)):
    """
    Preprocess a single data instance.
    
    For each movement event (ignoring "start" and "win"), three samples are generated:
      1. Pre-command sample: label = [0,0,0] at time = event_time - command_duration/2.
      2. At-command sample: label = command vector at time = event_time.
      3. Post-command sample: label = [0,0,0] at time = event_time + command_duration/2,
         and if the move is valid, the agent’s position is updated.
    
    The maze configuration (channel 0) is constant; channel 1 is a one-hot agent location.
    The same two-channel image is used for all three samples except that on the post-command
    sample the agent position is updated.
    
    Args:
        maze_config: 2D list/array (maze_size) representing the maze layout (e.g., 1 for wall, 0 for free).
        movement_log: List of dictionaries with keys "direction" (str), "valid" (bool), and "time" (ISO datetime string).
        constants: 15-element list/array (constant inputs for this sample).
        command_duration: Duration in seconds for a command event (default 0.05).
        maze_size: Tuple with maze dimensions (default (27, 27)).
        
    Returns:
        input_sequence: np.array of shape (T, 2, maze_size[0], maze_size[1])
                        where channel 0 is the maze layout and channel 1 is the one-hot agent location.
        labels: np.array of shape (T, 3) with command labels.
        constants: np.array of shape (15,) (unchanged).
        time_stamps: np.array of shape (T,) with the relative time (in seconds) for each sample.
    """
    # Convert maze layout to numpy array (assume it's maze_size)
    maze_array = np.array(maze_config, dtype=np.float32)

    # Parse movement events (skip "start" and "win")
    events = []
    for entry in movement_log:
        try: d_lower = entry["direction"].lower()
        except: continue
        if d_lower in ["start", "win"]:
            continue
        ts_str = entry["time"].replace("Z", "")
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

    # For each event we produce 3 samples.
    num_events = len(events)
    T = num_events * 3

    labels = np.zeros((T, 3), dtype=np.float32)
    agent_positions = np.zeros((T, 2), dtype=np.int32)
    time_stamps = np.zeros((T,), dtype=np.float32)
    current_agent = (0, 0)

    input_sequence = []
    maze_channel = maze_array.copy()  # constant maze layout

    # We will set times relative to the first event.
    first_event_time = events[0]["time"].timestamp()
    
    for i, ev in enumerate(events):
        # Compute relative event time in seconds.
        event_time = ev["time"].timestamp() - first_event_time
        
        # For the triangle, we define:
        # Sample 1 time: event_time - command_duration/2
        # Sample 2 time: event_time
        # Sample 3 time: event_time + command_duration/2
        sample_times = [event_time - command_duration/2,
                        event_time,
                        event_time + command_duration/2]
        
        base_index = i * 3

        # Sample 1: pre-command, label zeros, agent unchanged.
        labels[base_index] = np.array([0, 0, 0], dtype=np.float32)
        time_stamps[base_index] = sample_times[0]
        agent_positions[base_index] = current_agent

        # Sample 2: at-command, label = command vector, agent still unchanged.
        cmd_vector = get_command_vector(ev["direction"])
        labels[base_index + 1] = cmd_vector
        time_stamps[base_index + 1] = sample_times[1]
        agent_positions[base_index + 1] = current_agent

        # Sample 3: post-command, label zeros, then update agent if valid.
        labels[base_index + 2] = np.array([0, 0, 0], dtype=np.float32)
        time_stamps[base_index + 2] = sample_times[2]
        d_lower = ev["direction"].lower()
        if ev["valid"] and d_lower in ["up", "down", "left", "right", "reset"]:
            new_agent = update_agent_position(current_agent, ev["direction"])
            r, c = new_agent
            if 0 <= r < maze_size[0] and 0 <= c < maze_size[1]:
                current_agent = new_agent
        agent_positions[base_index + 2] = current_agent

        # Build input images for these three samples.
        for j in range(3):
            # For samples 1 and 2, use the old agent position; for sample 3, use updated position.
            pos = agent_positions[base_index] if j < 2 else agent_positions[base_index + 2]
            agent_channel = np.zeros(maze_size, dtype=np.float32)
            r, c = pos
            if 0 <= r < maze_size[0] and 0 <= c < maze_size[1]:
                agent_channel[r, c] = 1.0
            img = np.stack([maze_channel, agent_channel], axis=0)
            input_sequence.append(img)

    input_sequence = np.array(input_sequence, dtype=np.float32)  # (T, 2, H, W)
    constants = np.array(constants, dtype=np.float32)  # (15,)
    
    return input_sequence, labels, constants, time_stamps

# --- Updated Visualization Function ---
def visualize_processed_data(input_sequence, labels, time_stamps):
    """
    Visualizes the processed data using animation.
    
    Args:
        input_sequence: np.array of shape (T, 2, maze_size[0], maze_size[1])
                        - Channel 0: constant maze layout.
                        - Channel 1: agent location (one-hot encoded).
        labels: np.array of shape (T, 3), representing neuron activations.
        time_stamps: np.array of shape (T,), containing the actual timestamps of each frame.
    """
    T = len(time_stamps)
    maze_size = input_sequence.shape[2:]  # (H, W)

    # Create a figure with two subplots.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Subplot 1: Maze and agent position.
    maze_layout = (input_sequence[0, 0]-1) * -1  # Maze layout (constant)
    ax1.imshow(maze_layout, cmap='gray_r')
    agent_marker, = ax1.plot([], [], 'ro', markersize=6)
    ax1.set_title("Maze and Agent Position")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")
    ax1.invert_yaxis()  # Ensure (0,0) is at the top-left

    # Subplot 2: Neuron outputs vs. time.
    ax2.set_title("Output Neuron Activations Over Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Neuron Output")
    line0, = ax2.plot([], [], 'b-', label="Up/Down")
    line1, = ax2.plot([], [], 'g-', label="Left/Right")
    line2, = ax2.plot([], [], 'r-', label="Reset")
    ax2.legend()
    ax2.set_xlim(time_stamps[0], time_stamps[-1])
    ax2.set_ylim(np.min(labels) - 0.5, np.max(labels) + 0.5)

    # Initialization function: clear data.
    def init():
        agent_marker.set_data([], [])
        line0.set_data([], [])
        line1.set_data([], [])
        line2.set_data([], [])
        return agent_marker, line0, line1, line2

    # Update function for animation.
    def update(frame):
        # Update agent marker (find position where channel 1 is 1).
        agent_pos = np.argwhere(input_sequence[frame, 1] == 1)
        if agent_pos.size > 0:
            # Wrap the scalar positions in a list.
            agent_marker.set_data([agent_pos[0][1]], [agent_pos[0][0]])  # (col, row)
        else:
            agent_marker.set_data([], [])
            
        # Update neuron output plot (plot all data up to the current frame).
        t_current = time_stamps[:frame+1]
        line0.set_data(t_current, labels[:frame+1, 0])
        line1.set_data(t_current, labels[:frame+1, 1])
        line2.set_data(t_current, labels[:frame+1, 2])
        
        return agent_marker, line0, line1, line2

    # Create the animation using the actual time differences.
    ani = animation.FuncAnimation(
        fig, update, frames=range(T),
        init_func=init, blit=True, interval=200, repeat=False
    )

    plt.tight_layout()
    plt.show()


# --- Example usage ---
if __name__ == '__main__':

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
    
    data = open('Data/saveddata.pk','rb')
    data_load = pickle.load(data)
    data.close()

    processed_data = []

    for d in data_load:
        processed_data.append([])
        for constants,maze_config,movement_log in d:
            # Preprocess the data.
            inputs_seq, labels_seq, const_arr, time_stamps = preprocess_movement_data(maze_config, movement_log, constants)
            processed_data[-1].append((inputs_seq,labels_seq,const_arr,time_stamps))

    with open('Data/processed_data.pk','wb') as data_dump:
        pickle.dump(processed_data, data_dump)

    # Visualize the processed data
    visualize_processed_data(inputs_seq, labels_seq, time_stamps)