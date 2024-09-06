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

# Group the files by model (6 runs per model)
models = {
    'Model 1': metric_files[0:6],
    'Model 2': metric_files[6:12],
    'Model 3': metric_files[12:18],
    'Model 4': metric_files[18:24],
}

# Moving average window size (can be changed)
moving_avg_window = 20  # Set this to the desired window size

# Function to calculate moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Initialize lists to store the average rewards and losses per model
average_rewards_per_model = {}
average_losses_per_model = {}

# Load, compute averages, apply moving average, and store the results
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
    
    # Apply the moving average
    avg_rewards = moving_average(avg_rewards, moving_avg_window)
    avg_losses = moving_average(avg_losses, moving_avg_window)
    
    # Store the averages for each model
    average_rewards_per_model[model] = avg_rewards
    average_losses_per_model[model] = avg_losses

# Adjust the episode range to match the moving average length
episode_range = range(moving_avg_window - 1, len(avg_rewards) + moving_avg_window - 1)

# Colors for each model
colors = ['blue', 'green', 'red', 'purple']

# Set larger font sizes for all text elements
plt.rcParams.update({
    'font.size': 20,       # Default text size
    'axes.titlesize': 20,  # Titles of each subplot
    'axes.labelsize': 20,  # Labels for x and y axes
    'legend.fontsize': 16, # Font size for the legend
    'xtick.labelsize': 16, # Font size for x-axis tick labels
    'ytick.labelsize': 16  # Font size for y-axis tick labels
})

# Final comparison plot: only averages of rewards and losses for each model with moving average
plt.figure(figsize=(16, 8))

# Subplot for rewards
plt.subplot(1, 2, 1)
for i, (model, avg_rewards) in enumerate(average_rewards_per_model.items()):
    plt.plot(episode_range, avg_rewards, label=f'{model} - Avg Reward (MA={moving_avg_window})', color=colors[i], linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title(f'Average Reward per Episode (Moving Avg = {moving_avg_window})')
plt.legend()

# Subplot for losses
plt.subplot(1, 2, 2)
for i, (model, avg_losses) in enumerate(average_losses_per_model.items()):
    plt.plot(episode_range, avg_losses, label=f'{model} - Avg Loss (MA={moving_avg_window})', color=colors[i], linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title(f'Average Loss per Episode (Moving Avg = {moving_avg_window})')
plt.legend()

# Adjust layout and save the final comparison plot
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'average_comparison_rewards_losses_MA{moving_avg_window}_all_models.png'))
plt.show()
