import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

# Directory containing the metric files
save_dir = 'run_5_sensitivity/'  # Update with your correct path

# Learning rates and epsilon decay rates
learning_rates = [0.0001, 0.0005, 0.001]
eps_decay_rates = [0.99, 0.995, 0.985]

# Seed values
seeds = [7, 42, 99, 2024]  # Ensure you're using the right seeds

# Function to calculate moving average (optional)
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Initialize dictionaries to hold the data for rewards and losses
results = {}

# Loop through each learning rate and epsilon decay rate combination
for lr in learning_rates:
    for eps in eps_decay_rates:
        # Get all files matching this combination of learning rate and epsilon decay
        matching_files = [f for f in os.listdir(save_dir) if f'lr_{lr}' in f and f'epsdecay_{eps}' in f]
        
        # Initialize lists to hold rewards and losses for each seed
        all_rewards = []
        all_losses = []
        
        # Load the data from each matching file
        for file in matching_files:
            file_path = os.path.join(save_dir, file)
            with open(file_path, 'rb') as f:
                metrics = pickle.load(f)
                rewards = np.array(metrics['rewards'])
                losses = np.array(metrics['losses'])
                
                all_rewards.append(rewards)
                all_losses.append(losses)

        # Compute the average rewards and losses across all seeds for this combination
        avg_rewards = np.mean(all_rewards, axis=0)
        avg_losses = np.mean(all_losses, axis=0)

        # Store the results
        results[(lr, eps)] = {
            'avg_rewards': avg_rewards,
            'avg_losses': avg_losses,
            'all_rewards': all_rewards,
            'all_losses': all_losses
        }

# Plotting the results: 18 plots (9 for rewards, 9 for losses)
plt.rcParams.update({'font.size': 14})  # Set larger font size for plots

# Create 9 reward plots and 9 loss plots
for lr in learning_rates:
    for eps in eps_decay_rates:
        data = results[(lr, eps)]
        
        # Create figure for reward and loss
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Get episode range based on data length
        episode_range = range(len(data['avg_rewards']))
        
        # Plot raw rewards with a lighter color, paired with seed
        for rewards, seed in zip(data['all_rewards'], seeds):  # Ensure correct pairing of rewards and seeds
            axes[0].plot(episode_range, rewards, label=f'Seed {seed} - Reward', alpha=0.3)  # Lighter color for raw data
        axes[0].plot(episode_range, data['avg_rewards'], label=f'Average Reward', color='black', linewidth=2)  # Average in black
        axes[0].set_title(f'Rewards (LR: {lr}, Eps Decay: {eps})')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Average Reward')
        axes[0].legend()
        
        # Plot raw losses with a lighter color, paired with seed
        for losses, seed in zip(data['all_losses'], seeds):  # Ensure correct pairing of losses and seeds
            axes[1].plot(episode_range, losses, label=f'Seed {seed} - Loss', alpha=0.3)  # Lighter color for raw data
        axes[1].plot(episode_range, data['avg_losses'], label=f'Average Loss', color='black', linewidth=2)  # Average in black
        axes[1].set_title(f'Losses (LR: {lr}, Eps Decay: {eps})')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Average Loss')
        axes[1].legend()
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'avg_reward_loss_lr_{lr}_eps_{eps}.png'))
        plt.show()
