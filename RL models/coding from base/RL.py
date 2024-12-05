import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define GRU-based Actor-Critic Network
class GRUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, action_dim)  # Actor (policy)
        self.critic = nn.Linear(hidden_dim, 1)         # Critic (value)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        actor_out = self.actor(out)  # Policy logits
        critic_out = self.critic(out)  # State value
        return actor_out, critic_out, h

# Define Imaginator (predicts next state)
class Imaginator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Imaginator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, state_action):
        return self.fc(state_action)  # Predict next state

# A2C Agent with GRU
class A2C:
    def __init__(self, env, input_dim, hidden_dim, action_dim, lr=0.001, gamma=0.99):
        self.env = env
        self.gamma = gamma

        self.actor_critic = GRUNetwork(input_dim, hidden_dim, action_dim)
        self.imaginator = Imaginator(input_dim + action_dim, hidden_dim, input_dim)

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.imag_optimizer = optim.Adam(self.imaginator.parameters(), lr=lr)

    def select_action(self, state, hidden):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        logits, _, hidden = self.actor_critic(state_tensor, hidden)
        action_probs = torch.softmax(logits[0, -1], dim=-1)
        action = torch.multinomial(action_probs, 1).item()
        return action, hidden

    def train(self, num_episodes=1000):
        hidden = None

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, hidden = self.select_action(state, hidden)
                next_state, reward, done = self.env.step(action)
                episode_reward += reward

                # Train imaginator
                self.train_imaginator(state, action, next_state)

                state = next_state

            print(f"Episode {episode + 1}: Reward = {episode_reward}")

    def train_imaginator(self, state, action, next_state):
        state_action = np.concatenate([state, [action]])
        pred_next_state = self.imaginator(torch.tensor(state_action, dtype=torch.float32))
        loss = nn.MSELoss()(pred_next_state, torch.tensor(next_state, dtype=torch.float32))
        self.imag_optimizer.zero_grad()
        loss.backward()
        self.imag_optimizer.step()

# Define the custom Maze environment
class SimpleMazeEnv:
    def __init__(self, size=(5, 5)):
        self.size = size
        self.state = np.zeros(size)
        self.goal = (size[0] - 1, size[1] - 1)
        self.agent_pos = (0, 0)

    def reset(self):
        self.agent_pos = (0, 0)
        return self._get_state()

    def step(self, action):
        x, y = self.agent_pos

        # Move agent based on action
        if action == 0 and x > 0: x -= 1  # Up
        elif action == 1 and x < self.size[0] - 1: x += 1  # Down
        elif action == 2 and y > 0: y -= 1  # Left
        elif action == 3 and y < self.size[1] - 1: y += 1  # Right

        self.agent_pos = (x, y)
        done = self.agent_pos == self.goal
        reward = 1 if done else -0.01
        return self._get_state(), reward, done

    def _get_state(self):
        state = np.zeros(self.size)
        state[self.agent_pos] = 1
        return state.flatten()

# Instantiate Environment and A2C Agent
env = SimpleMazeEnv(size=(5, 5))
input_dim = env.size[0] * env.size[1]
hidden_dim = 128
action_dim = 4  # Up, Down, Left, Right

agent = A2C(env, input_dim, hidden_dim, action_dim)
agent.train(num_episodes=100)