import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from gymnasium.wrappers import FlattenObservation, GrayscaleObservation, ResizeObservation, RescaleAction
from gymnasium.spaces import Box


class ForceRGBChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = self.observation_space.low.min()
        high = self.observation_space.high.max()
        self.observation_space = Box(low=low, high=high, shape=(64, 64, 1), dtype=np.uint8)

    def observation(self, observation):
        return observation.reshape((64, 64, 1))

def _make_car_env(render_mode=None):
    env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode)
    env = MaxAndSkipEnv(env, skip=4)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=True)

    #env = gym.wrappers.ReshapeObservation(env, (64, 64, 1))
    #env = ForceRGBChannel(env) 
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
        
        #env = make_vec_env(env_id, n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv, env_kwargs={"continuous": True})
        #eval_env = make_vec_env(env_id, n_envs=1, seed=seed, env_kwargs={"continuous": True})
        
        # Stacking and Transposing
        #env = VecFrameStack(env, n_stack=4, channels_order="last")
        #eval_env = VecFrameStack(eval_env, n_stack=4, channels_order="last")
        
        env = VecFrameStack(env, n_stack=4)
        eval_env = VecFrameStack(eval_env, n_stack=4)
        
        env = VecTransposeImage(env) # Changes [64, 64, 4] to [4, 64, 64]
        eval_env = VecTransposeImage(eval_env)

    adjusted_eval_freq = max(eval_freq // n_envs, 1) # Adjust evaluation frequency according to # of environments
    #buffer_kwargs = {"optimize_memory_usage": True, "handle_timeout_termination": False}
    # 3. Setup the Evaluation Callback (for your plots!)
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
    # 4. Initialize & Train
    if algo_id == "PPO":
        #model = PPO(policy_type, env, tensorboard_log=tb_log_dir, ent_coef=0.0075, seed=seed)
        model = PPO(policy_type, env, tensorboard_log=tb_log_dir, seed=seed)
    elif algo_id == "SAC":
        model = SAC(policy_type, env, tensorboard_log=tb_log_dir, buffer_size=buffer_size, seed=seed, optimize_memory_usage=True, replay_buffer_kwargs={"handle_timeout_termination": False})
    elif algo_id == "TD3":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions)) # This configuration gives a bias towards the gas factor when CarRacing.
        model = TD3(policy_type, env, tensorboard_log=tb_log_dir, buffer_size=buffer_size, action_noise=action_noise, seed=seed, optimize_memory_usage=True, replay_buffer_kwargs={"handle_timeout_termination": False})
        
    model.learn(total_timesteps=total_steps, callback=eval_callback, progress_bar=True)
    env.close()
    model.save(log_path + "/final_model")
    del model
    print(f"Finished {env_id}!")
    
    return log_path


#if __name__ == "__main__":
    #run_experiment("dm_control/cartpole-swingup-v0", "SAC", 50000, eval_freq=5000)
    #run_experiment("dmc-acrobot-swingup", 100000)
    #run_experiment("CarRacing-v3", "SAC", total_steps=1000)


