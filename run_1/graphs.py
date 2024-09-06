import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

save_dir = 'run_1/'

# Filenames for the four training metrics files
metric_files = [
    'training_metrics_1_seed_7.pkl',
    'training_metrics_1_seed_42.pkl',
    'training_metrics_1_seed_99.pkl',
    'training_metrics_1_seed_2024.pkl',
    'training_metrics_1_seed_8888.pkl',
    'training_metrics_1_seed_31415.pkl'
]

# Function to calculate moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Initialize lists to store the moving averages for each file
all_moving_avg_rewards = []
all_moving_avg_losses = []

# Load and compute moving averages for rewards and losses
for metric_file in metric_files:
    file_path = os.path.join(save_dir, metric_file)

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            metrics = pickle.load(f)
            rewards = np.array(metrics['rewards'])
            losses = np.array(metrics['losses'])

            # Calculate 40-episode moving averages
            moving_avg_rewards = moving_average(rewards, 40)
            moving_avg_losses = moving_average(losses, 40)

            # Store the moving averages
            all_moving_avg_rewards.append(moving_avg_rewards)
            all_moving_avg_losses.append(moving_avg_losses)
    else:
        print(f"File {metric_file} not found in directory {save_dir}.")

# Define episode range (1000 episodes for each file)
episode_range = range(39, 1000)  # 39 because the moving average window is 40

# Set larger font sizes
plt.rcParams.update({'font.size': 14})  # You can adjust the number for different sizes

# Plot 40-episode moving averages for losses
plt.figure(figsize=(12, 8))
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

for i, moving_avg_loss in enumerate(all_moving_avg_losses):
    plt.plot(episode_range[:len(moving_avg_loss)], moving_avg_loss, label=f'Training Metric {i+1}', color=colors[i])

plt.xlabel('Episode', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('40-Episode Moving Average of Loss per Episode', fontsize=18)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(os.path.join(save_dir, 'moving_avg_40_losses_separate.png'))
plt.show()

# Plot 40-episode moving averages for rewards
plt.figure(figsize=(12, 8))
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

for i, moving_avg_reward in enumerate(all_moving_avg_rewards):
    plt.plot(episode_range[:len(moving_avg_reward)], moving_avg_reward, label=f'Training Metric {i+1}', color=colors[i])

plt.xlabel('Episode', fontsize=16)
plt.ylabel('Total Reward', fontsize=16)
plt.title('40-Episode Moving Average of Total Reward per Episode', fontsize=18)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(os.path.join(save_dir, 'moving_avg_40_rewards_separate.png'))
plt.show()
