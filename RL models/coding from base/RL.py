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
        self.imaginator = Imaginator(input_dim + 1, hidden_dim, input_dim)

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.imag_optimizer = optim.Adam(self.imaginator.parameters(), lr=lr)

    def select_action(self, state, hidden):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        logits, _, hidden = self.actor_critic(state_tensor, hidden)
        action_probs = torch.softmax(logits[0, -1], dim=-1)
        action = torch.multinomial(action_probs, 1).item()
        return action, hidden

    def train_actor_critic(self, state, action, reward, next_state, done, hidden):
        # Convert state and next_state to tensors
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [Batch=1, Seq=1, Input_dim]
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Initialize hidden state if None
        if hidden is None:
            hidden = torch.zeros(1, 1, self.actor_critic.gru.hidden_size)

        # Detach hidden state to avoid gradient through time
        hidden = hidden.detach()

        # Forward pass for the current state
        logits, critic_value, hidden = self.actor_critic(state_tensor, hidden)

        # Forward pass for the next state (detached hidden state)
        with torch.no_grad():  # Avoid computing gradients for next state
            _, next_critic_value, _ = self.actor_critic(next_state_tensor, hidden.detach())

        # Calculate the target value
        target_value = reward + (0 if done else self.gamma * next_critic_value.item())

        # Critic loss (Mean Squared Error between current value and target value)
        critic_loss = nn.MSELoss()(critic_value, torch.tensor([[target_value]], dtype=torch.float32))

        # Calculate the advantage
        advantage = target_value - critic_value.item()

        # Actor loss (Policy Gradient with advantage)
        action_probs = torch.softmax(logits[0, -1], dim=-1)  # Action probabilities for the last step
        actor_loss = -torch.log(action_probs[action]) * advantage

        # Combine losses
        total_loss = actor_loss + critic_loss

        # Optimize both actor and critic
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


    def train(self, num_episodes=1000):
        hidden = None

        total_steps = 0
        total_reward = 0
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False

            steps = 0
            while not done:
                action, hidden = self.select_action(state, hidden)
                next_state, reward, done = self.env.step(action)
                episode_reward += reward
                steps += 1
                if steps >= 100 and not done:
                    done = True
                    episode_reward -= 5

                # Train actor-critic
                self.train_actor_critic(state, action, reward, next_state, done, hidden)

                # Train imaginator
                self.train_imaginator(state, action, next_state)

                state = next_state

            print(f"\rEpisode {episode + 1}:\tReward = {round(episode_reward,2)},     \t\tSteps = {steps}\t",end='')
            total_steps += steps
            total_reward += episode_reward
            if episode % 20 == 0: print(f'\tmean steps: {total_steps/20} \tmean reward: {round(total_reward/20,2)}'); total_steps = 0; total_reward = 0

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
        self.cross = ((size[0]-1)**2 + (size[1]-1)**2)**.5

    def distance_calculator(self):
        self.distance = ((self.goal[0] - self.agent_pos[0])**2 + (self.goal[1]-self.agent_pos[1])**2)**.5

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
        self.distance_calculator()
        reward = 10 if done else -self.distance/(self.cross) / 5
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
agent.train(num_episodes=1000)