import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

# Define the directory containing the metric files
save_dir = '/Users/marco/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/LunarLander_RL/training_data'

# Filenames for the four training metrics files (6 seeds per model)
metric_files = [
    # Model 1 files
    'training_metrics_1_seed_7.pkl',
    'training_metrics_1_seed_42.pkl',
    'training_metrics_1_seed_99.pkl',
    'training_metrics_1_seed_2024.pkl',
    'training_metrics_1_seed_8888.pkl',
    'training_metrics_1_seed_31415.pkl',
    # Model 2 files
    'training_metrics_2_seed_7.pkl',
    'training_metrics_2_seed_42.pkl',
    'training_metrics_2_seed_99.pkl',
    'training_metrics_2_seed_2024.pkl',
    'training_metrics_2_seed_8888.pkl',
    'training_metrics_2_seed_31415.pkl',
    # Model 3 files
    'training_metrics_3_seed_7.pkl',
    'training_metrics_3_seed_42.pkl',
    'training_metrics_3_seed_99.pkl',
    'training_metrics_3_seed_2024.pkl',
    'training_metrics_3_seed_8888.pkl',
    'training_metrics_3_seed_31415.pkl',
    # Model 4 files
    'training_metrics_4_seed_7.pkl',
    'training_metrics_4_seed_42.pkl',
    'training_metrics_4_seed_99.pkl',
    'training_metrics_4_seed_2024.pkl',
    'training_metrics_4_seed_8888.pkl',
    'training_metrics_4_seed_31415.pkl'
]

# Titles for each model
model_titles = {
    'Model 1': 'Simple Model, Exp Decay Epsilon',
    'Model 2': 'Shallow Model, Exp Decay Epsilon',
    'Model 3': 'Simple Model, Sigmoid Decay Epsilon',
    'Model 4': 'Shallow Model, Sigmoid Decay Epsilon'
}

# Seed values corresponding to each run
seeds = {
    'Model 1': [7, 42, 99, 2024, 8888, 31415],
    'Model 2': [7, 42, 99, 2024, 8888, 31415],
    'Model 3': [7, 42, 99, 2024, 8888, 31415],
    'Model 4': [7, 42, 99, 2024, 8888, 31415]
}

# Group the files by model (6 runs per model)
models = {
    'Model 1': metric_files[0:6],
    'Model 2': metric_files[6:12],
    'Model 3': metric_files[12:18],
    'Model 4': metric_files[18:24],
}

# Set larger font sizes for all text elements
plt.rcParams.update({
    'font.size': 20,       # Default text size
    'axes.titlesize': 20,  # Titles of each subplot
    'axes.labelsize': 20,  # Labels for x and y axes
    'legend.fontsize': 16, # Font size for the legend
    'xtick.labelsize': 16, # Font size for x-axis tick labels
    'ytick.labelsize': 16  # Font size for y-axis tick labels
})

# Function to calculate moving average (if needed)
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Initialize lists to store the average rewards and losses per model
average_rewards_per_model = {}
average_losses_per_model = {}

# Load, compute averages, and store the results
for model, files in models.items():
    all_rewards = []
    all_losses = []
    
    for metric_file in files:
        file_path = os.path.join(save_dir, metric_file)
        
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                metrics = pickle.load(f)
                rewards = np.array(metrics['rewards'])
                losses = np.array(metrics['losses'])
                
                all_rewards.append(rewards)
                all_losses.append(losses)
        else:
            print(f"File {metric_file} not found.")
    
    # Compute average across runs (per episode)
    avg_rewards = np.mean(all_rewards, axis=0)
    avg_losses = np.mean(all_losses, axis=0)
    
    # Store the averages for each model
    average_rewards_per_model[model] = avg_rewards
    average_losses_per_model[model] = avg_losses

    # Define the episode range (assuming 1000 episodes for each run)
    episode_range = range(len(avg_rewards))  # Assumes the same length for all models
    
    # Create a single figure with two subplots for rewards and losses
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot raw losses with mean for the current model on the first subplot
    for i, losses in enumerate(all_losses):
        ax1.plot(episode_range, losses, label=f'Seed {seeds[model][i]} - Loss', alpha=0.5)
    ax1.plot(episode_range, avg_losses, label=f'{model} - Average Loss', color='black', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss - {model_titles[model]}')
    ax1.legend()
    ax1.tick_params(axis='both', which='major')

    # Plot raw rewards with mean for the current model on the second subplot
    for i, rewards in enumerate(all_rewards):
        ax2.plot(episode_range, rewards, label=f'Seed {seeds[model][i]} - Reward', alpha=0.5)
    ax2.plot(episode_range, avg_rewards, label=f'{model} - Average Reward', color='black', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.set_title(f'Reward - {model_titles[model]}')
    ax2.legend()
    ax2.tick_params(axis='both', which='major')

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'loss_reward_{model}.png'))
    plt.show()


