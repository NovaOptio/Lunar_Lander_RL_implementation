# Deep Q-Network with Epsilon Decay for Lunar Lander

This repository contains an implementation of a Deep Q-Network (DQN) with an epsilon-greedy decay algorithm. The main focus is to compare two different network architectures (simple and shallow) and two epsilon decay strategies (exponential and sigmoid) in the Lunar Lander environment from OpenAI Gym.

## Overview

The goal of this project is to explore the performance of different network architectures and epsilon decay strategies in a reinforcement learning task. We experiment with **six different random seeds** and assess the impact of different **learning rates** and **epsilon decay rates** in a sensitivity analysis.

### Experiments
The experiments conducted include the following configurations:
- **Network Architectures**: Simple vs. Shallow.
- **Epsilon Decay**: Exponential decay vs. Sigmoid decay.

### Seeds
The following seeds were used in all experiments to ensure consistency:
seeds = [7, 42, 99, 2024, 8888, 31415]


## Folder Structure

This repository is organized into the following folders:

1. **Folder 1**: Runs the experiments with the *simple model using exponential epsilon decay*.
2. **Folder 2**: Contains results for the *simple model using sigmoid epsilon decay*.
3. **Folder 3**: Contains results for the *shallow model using exponential epsilon decay*.
4. **Folder 4**: Runs the experiments for the *shallow model using sigmoid epsilon decay*.
5. **Graph Maker**: This folder contains scripts to generate performance graphs for the models mentioned above.

### `run_5_sensitivity` Folder

This folder contains the sensitivity analysis conducted on the best-performing model identified from the initial experiments. We selected **Model 1 (simple model with exponential epsilon decay)** as the best performer and conducted further analysis on various learning rates and epsilon decay rates.

#### Sensitivity Analysis Hyperparameters:
- **Learning Rates**: `[0.0001, 0.0005, 0.001]`
- **Epsilon Decay Rates**: `[0.99, 0.995, 0.985]`
- **Seeds**: `[7, 42, 99, 2024]`

## Conclusion

From the sensitivity analysis, we concluded that:
- A **learning rate of 0.0005** provides the best balance between rewards and losses.
- The **lowest loss** is achieved with a **learning rate of 0.0001**, but this results in significantly lower rewards.

## How to Use

To run the experiments or generate the graphs, navigate to the corresponding folder and execute the provided scripts. Ensure that you have the required dependencies installed (e.g., `numpy`, `matplotlib`, `pickle`) and update the file paths if necessary.


