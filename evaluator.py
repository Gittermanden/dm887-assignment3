import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from gymnasium.wrappers import FlattenObservation, GrayscaleObservation, ResizeObservation
from gymnasium.spaces import Box



def _make_car_env(render_mode=None):
    env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode, lap_complete_percent=0.95)
    env = ResizeObservation(env, (64, 64))
    env = GrayscaleObservation(env, keep_dim=True)

    return env

def run_experiment(env_id, algo_id, total_steps=100000, seed=42, n_envs=1, eval_freq=20000, buffer_size=200000):
    log_path = f"./eval_results/seed_{seed}/{env_id}/{algo_id}".replace("dm_control/", "")
    os.makedirs(log_path, exist_ok=True)
    print(f"--- Starting Experiment: {env_id}, using Model: {algo_id} ---")
    

    if "dm_control" in env_id:
        policy_type = "MlpPolicy"
        env = make_vec_env(env_id, n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv, wrapper_class=FlattenObservation)
        eval_env = make_vec_env(env_id, n_envs=1, seed=seed, wrapper_class=FlattenObservation)

    elif "CarRacing" in env_id:
        policy_type = "CnnPolicy"
        env = make_vec_env(lambda:_make_car_env(), n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv)
        eval_env = make_vec_env(lambda: _make_car_env(), n_envs=1, seed=seed)
                
        env = VecFrameStack(env, n_stack=4)
        eval_env = VecFrameStack(eval_env, n_stack=4)
        
        env = VecTransposeImage(env) # Changes [64, 64, 4] to [4, 64, 64]
        eval_env = VecTransposeImage(eval_env)

    adjusted_eval_freq = max(eval_freq // n_envs, 1)
    
    eval_callback = EvalCallback(
            eval_env, 
            best_model_save_path=log_path,
            log_path=log_path, 
            eval_freq=adjusted_eval_freq, 
            n_eval_episodes=10,
            deterministic=True,
            # render=True # Uncomment to watch the evaluation unfold for testing purposes
        )
    tb_log_dir = f"./tb_logs/{seed}/{env_id}/{algo_id}/".replace("dm_control/", "")


    if algo_id == "PPO":
        model = PPO(policy_type, env, tensorboard_log=tb_log_dir, seed=seed)
    elif algo_id == "SAC":
        model = SAC(policy_type, env, tensorboard_log=tb_log_dir, buffer_size=100000, seed=seed, optimize_memory_usage=True, replay_buffer_kwargs={"handle_timeout_termination": False}, learning_rate=1e-4, learning_starts=10000)
    elif algo_id == "TD3":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3*np.ones(n_actions))
        model = TD3(policy_type, env, tensorboard_log=tb_log_dir, buffer_size=400000, action_noise=action_noise, seed=seed, optimize_memory_usage=True, replay_buffer_kwargs={"handle_timeout_termination": False}, learning_rate=1e-4)
        
    model.learn(total_timesteps=total_steps, callback=eval_callback, progress_bar=True)
    env.close()
    model.save(log_path + "/final_model")
    del model
    print(f"Finished {env_id}!")
    
    return log_path


