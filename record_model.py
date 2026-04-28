from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy

import os

from platform import python_version
from importlib.metadata import version
from evaluator import _make_car_env




# Recording a video for a model
def main(log_dir, env_id, seed, algo_id):

    wrapper = FlattenObservation if "dm_control" in env_id else None

    # Create Evaluation environment
    if "CarRacing" in env_id:
        
        env = make_vec_env(lambda: _make_car_env(render_mode="rgb_array"), n_envs=1, seed=seed, wrapper_class=wrapper)
        env = VecFrameStack(env, n_stack=4)
        env = VecTransposeImage(env)
    else:
        env = make_vec_env(env_id, n_envs=1, seed=seed, wrapper_class=wrapper)
    # Load the best model
    best_model_path = os.path.join(log_dir, "best_model.zip")
    if algo_id == "SAC":
        best_model = SAC.load(best_model_path, env=env)
    elif algo_id == "TD3":
        best_model = TD3.load(best_model_path, env=env)
    elif algo_id == "PPO":
        best_model = PPO.load(best_model_path, env=env)

    #mean_reward, std_reward = evaluate_policy(best_model, env, n_eval_episodes=20)
    #print(f"Best Model - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Record video of the best model playing 
    best_model_file_name = "best_model_{}".format(env_id.replace("dm_control/", ""))
    env = VecVideoRecorder(env,
                        log_dir,
                        video_length=5_000,
                        record_video_trigger=lambda x: x == 0,
                        name_prefix=best_model_file_name)

    obs = env.reset()
    for _ in range(5_000):
        action, _states = best_model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            break

    env.close()
    del best_model

    # Create Evaluation environment
    if "CarRacing" in env_id:
        
        env = make_vec_env(lambda: _make_car_env(render_mode="rgb_array"), n_envs=1, seed=seed, wrapper_class=wrapper)
        env = VecFrameStack(env, n_stack=4)
        env = VecTransposeImage(env)
    else:
        env = make_vec_env(env_id, n_envs=1, seed=seed, wrapper_class=wrapper)


    final_model_path = os.path.join(log_dir, "final_model.zip")
    if algo_id == "SAC":
        final_model = SAC.load(final_model_path, env=env)
    elif algo_id == "TD3":
        final_model = TD3.load(final_model_path, env=env)
    elif algo_id == "PPO":
        final_model = PPO.load(final_model_path, env=env)
        
        
    #mean_reward, std_reward = evaluate_policy(final_model, env, n_eval_episodes=20)
    #print(f"Final Model - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


    # Record video of the final model playing 
    final_model_file_name = "final_model_{}".format(env_id.replace("dm_control/", ""))
    env = VecVideoRecorder(env,
                        log_dir,
                        video_length=5_000,
                        record_video_trigger=lambda x: x == 0,
                        name_prefix=final_model_file_name)

    obs = env.reset()
    for _ in range(5_000):
        action, _states = final_model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            break

    env.close()
    del final_model

#main("./eval_results/seed_54/CarRacing-v3/PPO", 54, "CarRacing-v3")