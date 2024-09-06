import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

# Define the directory containing the metric files
save_dir = './run_5_sensitivity'

# Learning rates and epsilon decay rates
learning_rates = [0.0001, 0.0005, 0.001]
eps_decay_rates = [0.99, 0.995, 0.985]
seeds = [7, 42, 99, 2024]

# Moving average window size (can be changed)
moving_avg_window = 30  # Set this to the desired window size

# Variables for setting y-axis ranges (change these to set the limits)
reward_y_max = 300
reward_y_min = -300
loss_y_max = 110
loss_y_min = 0

# Function to calculate moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Initialize dictionaries to hold average rewards and losses per run
average_rewards_per_run = {}
average_losses_per_run = {}

# Loop through each combination of learning rate and epsilon decay
for lr in learning_rates:
    for eps in eps_decay_rates:
        all_rewards = []
        all_losses = []
        
        # Loop through each seed
        for seed in seeds:
            file_name = f'metrics_lr_{lr}_epsdecay_{eps}_seed_{seed}.pkl'
            file_path = os.path.join(save_dir, file_name)
            
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    metrics = pickle.load(f)
                    rewards = np.array(metrics['rewards'])
                    losses = np.array(metrics['losses'])
                    
                    all_rewards.append(rewards)
                    all_losses.append(losses)
            else:
                print(f"File {file_name} not found.")
        
        # Compute the average rewards and losses across the 4 seeds for this combination
        avg_rewards = np.mean(all_rewards, axis=0)
        avg_losses = np.mean(all_losses, axis=0)
        
        # Apply moving average
        avg_rewards = moving_average(avg_rewards, moving_avg_window)
        avg_losses = moving_average(avg_losses, moving_avg_window)
        
        # Store the results
        key = f'LR: {lr}, Eps: {eps}'
        average_rewards_per_run[key] = avg_rewards
        average_losses_per_run[key] = avg_losses

# Adjust the episode range to match the moving average length
episode_range = range(moving_avg_window - 1, len(avg_rewards) + moving_avg_window - 1)

# Colors for each run
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive']

# Set larger font sizes for all text elements
plt.rcParams.update({
    'font.size': 16,       # Default text size
    'axes.titlesize': 18,  # Titles of each subplot
    'axes.labelsize': 16,  # Labels for x and y axes
    'legend.fontsize': 14, # Font size for the legend
    'xtick.labelsize': 14, # Font size for x-axis tick labels
    'ytick.labelsize': 14  # Font size for y-axis tick labels
})

# Plot average rewards for all runs
plt.figure(figsize=(14, 8))

# Subplot for rewards
plt.subplot(1, 2, 1)
for i, (key, avg_rewards) in enumerate(average_rewards_per_run.items()):
    plt.plot(episode_range, avg_rewards, label=f'{key}', color=colors[i], linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title(f'Average Reward per Episode (Moving Avg = {moving_avg_window})')
plt.ylim(reward_y_min, reward_y_max)  # Set the reward y-axis limits
plt.legend()

# Subplot for losses
plt.subplot(1, 2, 2)
for i, (key, avg_losses) in enumerate(average_losses_per_run.items()):
    plt.plot(episode_range, avg_losses, label=f'{key}', color=colors[i], linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Average Loss')
plt.title(f'Average Loss per Episode (Moving Avg = {moving_avg_window})')
plt.ylim(loss_y_min, loss_y_max)  # Set the loss y-axis limits
plt.legend()

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'average_comparison_rewards_losses_MA{moving_avg_window}_all_runs.png'))
plt.show()
