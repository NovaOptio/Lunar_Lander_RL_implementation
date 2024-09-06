import gymnasium as gym
import numpy as np
from collections import deque, namedtuple
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
import pickle
import os
import time


run = 4


save_dir = os.path.join(os.getcwd(), 'training_results_4')
print(f"Save directory: {save_dir}")

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)








#Shallow Network
class DQN(nn.Module):
 def __init__(self, inputs, outputs):
     super().__init__()
     self.fc1 = nn.Linear(in_features=inputs, out_features=128)
     self.out = nn.Linear(in_features=128, out_features=outputs)

 def forward(self, t):
     t = F.relu(self.fc1(t))
     t = self.out(t)
     return t

# Define the ReplayBuffer for experience replay
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity, batch_size, seed=None):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
        if seed is not None:
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

# Define the DQN Agent
class LunarLanderDQNAgent:
    def __init__(self, state_space, action_space, alpha=0.0005, eps=1.0, eps_decay=0.995, eps_min=0.01, gamma=0.99, batch_size=64, buffer_size=100000, seed=None):
        self.state_space = state_space
        self.action_space = action_space
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"size of state space: {state_space}")
        print(f"size of action space: {action_space}")


        self.policy_net = DQN(state_space, action_space).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.memory = ReplayBuffer(buffer_size, batch_size, seed)

    def select_action(self, state):
        if np.random.random() < self.eps:
            return np.random.randint(0, self.action_space)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor([state], dtype=torch.float32).to(self.device)
                return self.policy_net(state_tensor).argmax(dim=1).item()

    def update_q(self, state, action, next_state, reward, done):
        # Check the shapes of the states
        if np.shape(state) != np.shape(next_state):
            raise ValueError(f"Inconsistent state shapes: state {np.shape(state)}, next_state {np.shape(next_state)}")

        # Add the experience to the replay buffer
        self.memory.add_experience(state, action, reward, next_state, done)

        # Decay epsilon after the episode ends
        if done:
            self.eps = max(self.eps_min, self.eps * self.eps_decay)

        # Only learn if the replay buffer has enough experiences
        if len(self.memory) > self.memory.batch_size:
            experiences = self.memory.sample_batch(self.device)
            loss = self.learn(experiences)
            return loss  # Return the loss for tracking

        return 0.0  # Return 0.0 if no learning is done


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

    def load_network(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()





# Set the seed
seed_list = [ 7, 99, 2024, 31415, 8888]
        
for i in seed_list:
  seed = i
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


  # Initialize the environment
  env = gym.make('LunarLander-v2')
  env.action_space.seed(seed)
  env.observation_space.seed(seed)

  state_space = env.observation_space.shape[0]
  action_space = env.action_space.n
  agent = LunarLanderDQNAgent(state_space, action_space)

  # Metrics to track
  rewards = []
  epsilons = []
  actions_taken = []
  losses = []

  num_episodes = 1000
  for episode in range(num_episodes):
      state, _ = env.reset()
      done = False
      total_reward = 0
      episode_actions = []
      episode_loss = 0
      num_updates = 0


      while not done:
          action = agent.select_action(state)
          next_state, reward, terminated, truncated, _ = env.step(action)
          done = terminated or truncated

          episode_actions.append(action)

          # Update Q-values and get the loss
          loss = agent.update_q(state, action, next_state, reward, done)

          # Accumulate loss and count updates
          episode_loss += loss
          num_updates += 1

          state = next_state
          total_reward += reward

      # Average the loss over the number of updates, if any
      if num_updates > 0:
          episode_loss /= num_updates

      rewards.append(total_reward)
      epsilons.append(agent.eps)
      actions_taken.append(episode_actions)
      losses.append(episode_loss)


      print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.eps}, Loss: {episode_loss}, updates: {num_updates}")

      if (episode + 1) % 100 == 0:
          model_file_name = f"lunar_lander_dqn_{run}_{episode + 1}.pth"
          model_file_path = os.path.join(save_dir, model_file_name)

          # Check if the model file already exists and increment version if it does
          version = 1
          while os.path.exists(model_file_path):
              model_file_name = f"lunar_lander_dqn_{run}_{episode + 1}_V{version+1}.pth"
              model_file_path = os.path.join(save_dir, model_file_name)
              version += 1

          agent.save_network(model_file_path)
          print(f"Model saved to {model_file_path}")

  # Ensure a unique filename for the pickle file
  pickle_file_name = f'training_metrics_{run}_seed_{seed}.pkl'
  pickle_file_path = os.path.join(save_dir, pickle_file_name)

  # Check if the pickle file already exists and increment version if it does
  version = 1
  while os.path.exists(pickle_file_path):
      pickle_file_name = f'training_metrics_{run}_seed_{seed}_V{version+1}.pkl'
      pickle_file_path = os.path.join(save_dir, pickle_file_name)
      version += 1

  # Save all the metrics to a file with a unique name
  with open(pickle_file_path, 'wb') as f:
      pickle.dump({
          'rewards': rewards,
          'epsilons': epsilons,
          'actions_taken': actions_taken,
          'losses': losses
      }, f)

  print(f"Training metrics have been saved successfully to {pickle_file_path}.")

  env.close()