import imageio
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from gymnasium.wrappers import FlattenObservation
from evaluator import _make_car_env

def plot_results(log_dir, env_id, algo_id):        
    data = np.load(f"{log_dir}/evaluations.npz")
    steps = data['timesteps']
    results = data['results'] # This is a 2D array [num_evals, n_eval_episodes]
    
    mean = np.mean(results, axis=1)
    std = np.std(results, axis=1)
    
    plt.figure(figsize=(10, 5))
    plt.fill_between(steps, mean - std, mean + std, alpha=0.2)
    plt.plot(steps, mean, label="Mean Eval Return")
    plt.title(f"Learning Curve for {algo_id} on {env_id}")
    plt.xlabel("Training Steps")
    plt.ylabel("Undiscounted Return")
    plt.grid(True)
    plt.savefig(f"{log_dir}/learning_curve.png")
    plt.close()

def compare_results(results, env_id):
    environment = env_id.replace("dm_control/", "")
    plt.figure(figsize=(10,5))
    
    for res in results:
        data = np.load(f"{res['path']}/evaluations.npz")
        steps = data['timesteps']
        eval_data = data['results']
        
        mean_rewards = np.mean(eval_data, axis=1)
        plt.plot(steps, mean_rewards, label=f"{res['algo']}")

    plt.title(f"Algorithm Comparison: {environment}")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results[-1]['path']}/../comparison_{environment}.png")
    plt.close()