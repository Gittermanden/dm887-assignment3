from plotter import plot_results, compare_results
from record_model import main as record_model
from evaluator import run_experiment
from gymnasium.envs.registration import registry
import gymnasium as gym
import shimmy
from shimmy import dm_control_compatibility


if __name__ == "__main__":
    """
    environments = [
        #{"id":"dm_control/cartpole-swingup-v0", "steps": 100000, "eval_freq": 10000}, 
        #{"id": "dm_control/acrobot-swingup-v0", "steps": 200000, "eval_freq": 10000}, 
        #{"id":"CarRacing-v3", "steps": 1000000,"eval_freq": 50000}
    ]
    """
        
    environments = [
        #{"id":"dm_control/cartpole-swingup-v0", "steps": 2000, "eval_freq": 200}, 
        #{"id": "dm_control/acrobot-swingup-v0", "steps": 2000, "eval_freq": 200}, 
        {"id":"CarRacing-v3", "steps": 500000,"eval_freq": 50000}
    ]
    
    #algorithms = ["SAC", "TD3", "PPO"]
    algorithms = ["TD3"]
    #seed_pool = [5, 8, 32, 67, 154]
    seed_pool = [56]

    for seed in seed_pool:
        for env in environments:
            #results = []
            for algo in algorithms:
                try:
                    log_path = run_experiment(
                        env_id=env['id'], 
                        algo_id=algo,
                        total_steps=env["steps"],
                        seed=seed,
                        n_envs=8, # PPO is the only algorithm that is built around multiple environments, but, whatever...
                        eval_freq=env["eval_freq"],
                    )

                    # Record a video of the best model encountered during training in the environment
                    try:
                        record_model(log_path, env['id'], seed, algo)
                    except Exception as e:
                        print(f"Failed to record {algo} on {env['id']}': {e}")
                    
                    # Plot the learning curve for the environment
                    try:
                        plot_results(log_path, env['id'], algo)
                        #results.append({"algo": algo, "path": log_path}) # For comparison plot
                    except Exception as e:
                        print(f"Failed to graph {algo} on {env['id']}': {e}")
                except Exception as e:
                    print(f"Failed to run {algo} on {env['id']}': {e}")
            # Compare the algorithms on the environment
            #try:
            #    compare_results(results, env['id'])
            #except Exception as e:
            #    print(f"Failed to compare algorithms on {env['id']}': {e}")