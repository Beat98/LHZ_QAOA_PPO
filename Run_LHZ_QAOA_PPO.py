import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import sys
import time
from pathlib import Path

sys.path.append(f'{Path(__file__).parent.joinpath("ENV").joinpath("lhz")}')
sys.path.append(f'{Path(__file__).parent.joinpath("ENV").joinpath("qaoa")}')

from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from ENV.LHZ_QAOA_env import QaoaEnv
from functions import generate_setup_file, save_data, mask_fn, load_data_from_csv, ProgressBarCallback


def run_lhz_qaoa_ppo(config=None):
    """
    Runs several executions of a complete PPO learning process with 'n_timesteps' timesteps. Stores learning data of
    each execution separately in data folder. Stores important experiment information (the setup) in setup.txt together
    with a csv and the stable baslines model.

    Parameters:
    - config (dict, optional): Configuration dictionary that overrides default parameters.
      The following keys are expected:

      - n_max_gates (int): Maximal gate sequence length. Default is 6.
      - n_steps (int): Number of steps between policy updates. Default is 50.
      - n_timesteps (int): Number of environment steps. Default is 1000.
      - num_executions (int): Number of experiment executions. Default is 1.
      - gamma (float): Discount factor for rewards. Default is 0.99.
      - qaoa_optimizer (str): The classical optimizer for QAOA. Default is "SLSQP".
      - reps (int): Number of random initializations of the QAOA angles. Default is 10.
      - num_qubits (int): Number of parity qubits. Here: 6 or 10. Default is 6.
      - constraint_strength (int): Constraint strength. Only relevant if the optimization target is "physical_energy".
      Default is 3.
      - optimization_target (str): The optimization target for QAOA optimization: "physical_energy" or "logical_energy".
      Default is "logical_energy".
      - final_decoding (str): The method for final decoding. Default is "logical_mean".
      - additional_unitaries (str): The gate pool from which the additional unitaries are used. Here: None, "l1" or "l2".
      Default is "l1".

    Example usage:
    run_lhz_qaoa_ppo(config={"n_timesteps": 2000, "num_executions": 5})
    """

    # Set default parameters
    default_config = {
        "n_max_gates": 6,
        "n_steps": 50,
        "n_timesteps": 1000,
        "num_executions": 1,
        "gamma": 0.99,
        "qaoa_optimizer": "SLSQP",
        "reps": 10,
        "num_qubits": 6,
        "constraint_strength": 3,
        "optimization_target": "logical_energy",
        "final_decoding": "logical_mean",
        "additional_unitaries": "l1"
    }

    # Update defaults with provided config
    if config is not None:
        default_config.update(config)

    n_max_gates = default_config["n_max_gates"]
    n_steps = default_config["n_steps"]
    n_timesteps = default_config["n_timesteps"]
    num_executions = default_config["num_executions"]
    gamma = default_config["gamma"]
    qaoa_optimizer = default_config["qaoa_optimizer"]
    reps = default_config["reps"]
    num_qubits = default_config["num_qubits"]
    constraint_strength = default_config["constraint_strength"]
    optimization_target = default_config["optimization_target"]
    final_decoding = default_config["final_decoding"]
    additional_unitaries = default_config["additional_unitaries"]

    # selection of gates for QAOA
    if additional_unitaries == "l_1":
        actions = ['X', 'Z', 'C', 'A', 'B']
    elif additional_unitaries == "l_2":
        actions = ['X', 'Z', 'C', 'A', 'B', 'H', 'I', 'J', 'K', 'L']
    else:
        actions = None

    # Selected hard problem instances for 6 and 10 qubits
    if num_qubits == 6:
        constraints = [[0, 1, 3], [1, 2, 3, 4], [3, 4, 5]]
        local_fields = [-1, -1, -1, 1, -1, -1]
    elif num_qubits == 10:
        constraints = [[0, 1, 4], [1, 2, 4, 5], [2, 3, 5, 6], [4, 5, 7], [5, 6, 7, 8], [7, 8, 9]]
        local_fields = [-1, -1, -1, -1, 1, -1, 1, -1, 1, -1]
    else:
        raise "check number of qubits"

    # folder where the data dictionary is stored
    data_set_name = "test_data_set"
    experiment_name = "experiment_1"

    problem_dict = {"num_qubits": num_qubits, "constraints": constraints, "constraint_strength": constraint_strength,
                    "local_fields": local_fields}

    for i in range(num_executions):
        folder_name = os.path.join(data_set_name, experiment_name, f"execution_{i}")
        path = Path(__file__).parent.joinpath("data", folder_name)

        # Dictionary to store important data
        data = {"problem_dict": problem_dict,
                "n_steps": n_steps,
                "n_timesteps": n_timesteps,
                "n_max_gates": n_max_gates,
                "reps": reps,
                "additional_unitaries": additional_unitaries,
                "qaoa_optimizer": qaoa_optimizer,
                "optimization_target": optimization_target,
                "final_decoding": final_decoding,
                "actions": actions}

        env = QaoaEnv(problem_dict=problem_dict,
                      reps=reps,
                      n_max_gates=n_max_gates,
                      qaoa_optimizer=qaoa_optimizer,
                      final_decoding=final_decoding,
                      optimization_target=optimization_target,
                      actions=actions)

        # set up the folders for logging and saving
        log_dir = os.path.join(path, "logs")
        models_dir = os.path.join(path, "models")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        # stores additional training data in monitor.csv
        env = Monitor(env, log_dir, info_keywords=("min_energy", "gate_sequence", "qaoa_time"))

        # enable action masking with mask function
        env = ActionMasker(env, mask_fn)

        # set up the model and train it
        data["policy"] = policy = MaskableActorCriticPolicy
        model = MaskablePPO(policy, env, verbose=0, n_steps=n_steps, gamma=gamma)

        # define the progress bar callback
        callback = ProgressBarCallback(total_timesteps=n_timesteps)

        # record the start time
        start_time = time.time()

        # learn with the callback
        model.learn(total_timesteps=n_timesteps, reset_num_timesteps=False, callback=callback)

        # calculate the learning time
        learning_time = time.time() - start_time
        data["learning_time"] = learning_time

        model.save(path.joinpath(models_dir, f"{n_timesteps}"))

        # save data
        csv_path = os.path.join(log_dir, "monitor.csv")
        csv_data = load_data_from_csv(csv_path)
        generate_setup_file(folder_name, data)
        data.update(csv_data)
        save_data(folder_name=os.path.join(data_set_name, experiment_name), filename=f"execution_{i}", data=data)


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="LHZ QAOA PPO")
    parser.add_argument('--n_max_gates', type=int, help='Maximal gate sequence length. Default is 6.')
    parser.add_argument('--n_steps', type=int, help='Number of steps between policy updates. Default is 50.')
    parser.add_argument('--n_timesteps', type=int, help='Number of environment steps. Default is 1000.')
    parser.add_argument('--num_executions', type=int, help='Number of experiment executions. Default is 1.')
    parser.add_argument('--gamma', type=float, help='Discount factor for rewards. Default is 0.99.')
    parser.add_argument('--qaoa_optimizer', type=str, help='The classical optimizer for QAOA. Default is "SLSQP".')
    parser.add_argument('--reps', type=int, help='Number of random initial of the QAOA angles. Default is 10.')
    parser.add_argument('--num_qubits', type=int, help='Number of parity qubits. Here: 6 or 10. Default is 6.')
    parser.add_argument('--constraint_strength', type=int,
                        help='Constraint strength. Only relevant if the optimization target is "physical_energy". '
                             'Default is 3.')
    parser.add_argument('--optimization_target', type=str,
                        help='The target for QAOA optimization: "physical_energy" or "logical_energy". Default is '
                             '"logical_energy".')
    parser.add_argument('--final_decoding', type=str, help='The method for final decoding. Default is "logical_mean".')
    parser.add_argument('--additional_unitaries', type=str,
                        help='The gate pool from which the additional unitaries are used. Here: None, "l1" or "l2". '
                             'Default is "l1".')

    args = parser.parse_args()

    cmd_args = {key: value for key, value in vars(args).items() if value is not None}

    custom_config = {
        "n_timesteps": 500
    }

    combined_config = {**custom_config, **cmd_args}

    run_lhz_qaoa_ppo(config=combined_config)
