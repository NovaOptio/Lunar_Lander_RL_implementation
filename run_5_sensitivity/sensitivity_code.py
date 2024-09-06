import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import pickle
from collections import deque, namedtuple

# Define the DQN Model
class DQN(nn.Module):
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, outputs)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# Replay Buffer for Experience Replay
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity, batch_size, seed=None):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
        if seed:
            random.seed(seed)

    def add_experience(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample_batch(self, device):
        batch = random.sample(self.buffer, self.batch_size)
        states = torch.tensor(np.vstack([t.state for t in batch]), dtype=torch.float32).to(device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(np.vstack([t.next_state for t in batch]), dtype=torch.float32).to(device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32).unsqueeze(1).to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# DQN Agent
class LunarLanderDQNAgent:
    def __init__(self, state_space, action_space, alpha=0.0005, eps=1.0, eps_decay=0.995, eps_min=0.01, gamma=0.99, batch_size=64, buffer_size=100000, seed=None):
        self.state_space = state_space
        self.action_space = action_space
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_space, action_space).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.memory = ReplayBuffer(buffer_size, batch_size, seed)

    def select_action(self, state):
        if np.random.random() < self.eps:
            return np.random.randint(0, self.action_space)
        else:
            state_tensor = torch.tensor([state], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                return self.policy_net(state_tensor).argmax(dim=1).item()

    def update_q(self, state, action, next_state, reward, done):
        self.memory.add_experience(state, action, reward, next_state, done)
        if done:
            self.eps = max(self.eps_min, self.eps * self.eps_decay)

        if len(self.memory) > self.memory.batch_size:
            experiences = self.memory.sample_batch(self.device)
            loss = self.learn(experiences)
            return loss
        return 0.0

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        next_q_values = self.policy_net(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        current_q_values = self.policy_net(states).gather(1, actions)
        loss = nn.MSELoss()(current_q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save_network(self, path):
        torch.save(self.policy_net.state_dict(), path)

# Sensitivity Analysis Variables
learning_rates = [ 0.001]
eps_decay_rates = [0.99, 0.985, 0.995]

# Set up the environment and seeds
seed_list = [7, 99, 42, 2024]
env_name = 'LunarLander-v2'
num_episodes = 1000
save_dir = os.getcwd()

# Loop over each combination of alpha (learning rate) and epsilon decay
for alpha in learning_rates:
    for eps_decay in eps_decay_rates:
        # Skip the combination (alpha=0.0005 and eps_decay=0.995)
        if alpha == 0.0005 and eps_decay == 0.995:
            print(f"Skipping alpha={alpha}, eps_decay={eps_decay} (results already available).")
            continue
        
        print(f"Testing with alpha={alpha}, eps_decay={eps_decay}")

        # Run the experiment for each seed
        for seed in seed_list:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            env = gym.make(env_name)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)

            state_space = env.observation_space.shape[0]
            action_space = env.action_space.n

            # Initialize agent
            agent = LunarLanderDQNAgent(state_space, action_space, alpha=alpha, eps_decay=eps_decay)

            # Metrics to track
            rewards = []
            losses = []

            for episode in range(num_episodes):
                state, _ = env.reset()
                done = False
                total_reward = 0
                episode_loss = 0
                num_updates = 0

                while not done:
                    action = agent.select_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    # Update agent and accumulate loss
                    loss = agent.update_q(state, action, next_state, reward, done)
                    episode_loss += loss
                    num_updates += 1

                    state = next_state
                    total_reward += reward

                # Track metrics
                rewards.append(total_reward)
                if num_updates > 0:
                    episode_loss /= num_updates
                losses.append(episode_loss)

                print(f"Seed {seed} | Alpha {alpha} | Eps Decay {eps_decay} | Episode {episode + 1} | Reward: {total_reward} | Loss: {episode_loss}")

            # Save metrics
            metrics_file = f"metrics_lr_{alpha}_epsdecay_{eps_decay}_seed_{seed}.pkl"
            with open(os.path.join(save_dir, metrics_file), 'wb') as f:
                pickle.dump({
                    'rewards': rewards,
                    'losses': losses,
                    'alpha': alpha,
                    'eps_decay': eps_decay,
                    'seed': seed
                }, f)

            print(f"Metrics saved for alpha {alpha}, eps_decay {eps_decay}, seed {seed}")
            env.close()