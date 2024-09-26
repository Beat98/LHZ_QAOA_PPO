from itertools import product

import sys
from pathlib import Path
import os
import csv
from qutip import expect, tensor, basis
from tqdm import tqdm

sys.path.append(f'{Path(__file__).parent.joinpath("ENV").joinpath("lhz")}')
sys.path.append(f'{Path(__file__).parent.joinpath("ENV").joinpath("qaoa")}')

from lhz.qutip_hdicts import hdict_physical_not_full
from lhz.core import n_logical
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from sb3_contrib.ppo_mask import MaskablePPO

import gym
import numpy as np
import pandas as pd
import copy

import json
from pathlib import Path

from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps):  # total_timesteps is the total number of steps the model will be trained for
        super(ProgressBarCallback, self).__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self):
        """ This method is called before the first rollout starts. """
        if self.pbar is not None:
            self.pbar.close()
        self.pbar = tqdm(total=self.total_timesteps)

    def _on_step(self):
        """ This method is called for each step in training. """
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        """ This method is called after training ends. """
        self.pbar.close()


def rename_sequence(sequence):
    mapping = {
        'X': '\hat{\\text{X}}',
        'C': '\hat{\\text{C}}',
        'Z': '\hat{\\text{Z}}',
        'A': '\hat{\\text{U}}_1',
        'B': '\hat{\\text{U}}_2',
        'H': '\hat{\\text{U}}_3',
        'I': '\hat{\\text{U}}_4',
        'J': '\hat{\\text{U}}_5',
        'K': '\hat{\\text{U}}_6',
        'L': '\hat{\\text{U}}_7',
    }

    for items in mapping.items():
        sequence = sequence.replace(items[0], items[1])
    return sequence


def save_data_json(folder_name: str, filename: str, data):
    """Creates the folder data (if it doesn't exist) in the parent directory and stores a json file of
    the 'data' with name filename.json"""

    folder_path = Path(__file__).parent.joinpath("data", f"{folder_name}")
    folder_path.mkdir(parents=True, exist_ok=True)

    with open(f'{folder_path.joinpath(f"{filename}.json")}', 'w') as json_file:
        json.dump(data, json_file)


def save_data(folder_name: str, filename: str, data):
    """Creates the folder data (if it doesn't exist) in the parent directory and stores a numpy file of
    the 'data' with name filename.npy"""

    folder_path = Path(__file__).parent.joinpath("data", f"{folder_name}")
    folder_path.mkdir(parents=True, exist_ok=True)
    np.save(f'{folder_path.joinpath(f"{filename}")}', data)


def load_data(folder_name: str, filename: str):
    folder_path = Path(__file__).parent.joinpath("data", f"{folder_name}")
    data = np.load(f'{folder_path.joinpath(f"{filename}.npy")}', allow_pickle=True)
    return data.item()


def get_data_list(folder_name, num_executions):
    data_list = []
    for i in range(num_executions):
        file_name = f"execution_{i}"
        folder_path = Path(__file__).parent.joinpath("data", f"{folder_name}")
        try:
            data = np.load(f'{folder_path.joinpath(f"{file_name}.npy")}', allow_pickle=True)
        except FileNotFoundError as e:
            print(f"{e}")
            assert f"{e}"
            continue
        data_list.append(data.item())
    return data_list


def generate_setup_file(folder_name: str, data):
    """Creates the folder data (if it doesn't exist) in the parent directory and sets up a txt file with important
    parameters and the problem description"""

    data = copy.copy(data)

    folder_path = Path(__file__).parent.joinpath("data", f"{folder_name}")
    folder_path.mkdir(parents=True, exist_ok=True)

    with open(f"{folder_path.joinpath('setup.txt')}", "w") as f:
        f.write(str(data))


def get_standard_sequence(gate_length):
    pattern = "CZXCZX"
    sequence = (pattern * (gate_length // len(pattern))) + pattern[:gate_length % len(pattern)]
    return sequence[:gate_length]


def get_triv_sequence(gate_length):
    pattern = "ZXZXZX"
    sequence = (pattern * (gate_length // len(pattern))) + pattern[:gate_length % len(pattern)]
    return sequence[:gate_length]


def moving_average(list, window_lenght=100):
    list_nan_padding = np.zeros(len(list) + 2 * (window_lenght // 2))
    for i in range(len(list_nan_padding)):
        if i >= window_lenght // 2 and i < len(list_nan_padding) - window_lenght // 2:
            list_nan_padding[i] = list[i - window_lenght // 2]
        else:
            list_nan_padding[i] = np.NaN

    moving_avg = np.zeros(len(list))
    for i in range(len(list)):
        moving_avg[i] = np.nanmean(list_nan_padding[i:i + window_lenght])
    return moving_avg


def get_data_from_csv(path, filename):
    data_list = []
    full_path = Path(path) / filename

    with open(full_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for row in reader:
            step = int(row[1])
            value = float(row[2])
            data_list.append([step, value])

    return data_list


def load_data_from_csv(path):
    # Read the CSV data using pandas
    data = pd.read_csv(path, skiprows=1)

    # Extract columns as NumPy arrays
    reward = data['r'].to_numpy()
    gate_length = data['l'].to_numpy()
    # qaoa_time = data['qaoa_time'].to_numpy()
    min_energy = data['min_energy'].to_numpy()
    gate_sequences = data['gate_sequence'].to_numpy()

    # Create a data dictionary with the specified keys and values
    data_dict = {
        "reward": reward,
        "gate_length": gate_length,
        # "qaoa_time": qaoa_time,
        "min_energy": min_energy,
        "gate_sequences": gate_sequences
    }
    return data_dict


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()


def load_masked_ppo(model, env, num_episodes=10):
    data = {
        "qaoa_time": [],
        "reward": [],
        "min_energy": [],
        "gate_sequences": []
    }

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action_mask = env.valid_action_mask()
            action, _ = model.predict(obs, action_masks=action_mask)
            obs, reward, done, info = env.step(action)
            if done:
                data["gate_sequences"].append(env.program_str)
                data["min_energy"].append(info["min_energy"])
                data["qaoa_time"].append(info["qaoa_time"])
                data["reward"].append(reward)

    return data


def run_qaoa_optimization(problem_instance, program_str, reps=100, qaoa_optimizer="SLSQP"):
    problem_instance.set_programstring(program_str)

    result_container = problem_instance.repeated_scipy_optimization(
        optimization_method=qaoa_optimizer, reps=reps)
    best_result = result_container.get_lowest_objective_value()

    return best_result


# def get_gates(obs):
#     gate_sequence = ""
#     for i, int_gate in enumerate(obs):
#         gate_sequence += ["_", "X", "C", "Z"][int(int_gate)]
#     return gate_sequence


# def get_problem_hdict(num_qubits):
#     """Problem configurations for group retreat presentation"""
#     if num_qubits == 3:
#         constraints = [[0, 1, 2]]
#         constraint_strength = 3.0
#         local_fields = [1, -1, -1]
#     elif num_qubits == 4:
#         constraints = [[0, 1, 2, 3]]
#         constraint_strength = 3.0
#         local_fields = [1, -1, -1, -1]
#     elif num_qubits == 6:
#         constraints = [[0, 1, 3], [1, 3, 4, 5], [1, 2, 4]]
#         constraint_strength = 3.0
#         local_fields = [1, -1, 1, -1, -1, 1]
#     else:
#         raise ValueError("We have problems for 3, 4 and 6 qubits")
#
#     h_dict = hdict_physical_not_full(jij=local_fields, constraints=constraints, cstrength=constraint_strength)
#
#     return h_dict, constraints, constraint_strength, local_fields


def calculate_E_min_E_max(data=None, problem_dict=None):
    if problem_dict is None:
        problem_dict = data["problem_dictionary"]

    def generate_states(num_qubits):
        qubit_0 = basis(2, 0)
        qubit_1 = basis(2, 1)

        qubit_states = [qubit_0, qubit_1]

        all_states = []

        for qubit_combination in product(range(2), repeat=num_qubits):
            state = tensor([qubit_states[i] for i in qubit_combination])
            all_states.append(state)

        return all_states

    hdict = hdict_physical_not_full(jij=problem_dict["local_fields"], constraints=problem_dict["constraints"],
                                    cstrength=problem_dict["constraint_strength"])

    state_list = generate_states(len(problem_dict["local_fields"]))
    expectation_val_list = [expect(hdict['P'], state) for state in state_list]

    return min(expectation_val_list), max(expectation_val_list)


def generate_configs(num_qubits):
    configs = list(product([-1, 1], repeat=num_qubits))
    return [np.array(config) for config in configs]


def filter_equivalent_sequences(sequence_list):
    filtered_sequences = []
    for sequence in sequence_list:
        normalized_sequence = sequence.replace("ZC", "CZ")
        filtered_sequences.append(normalized_sequence)

    filtered_sequences = np.array(filtered_sequences)
    filtered_sequences = np.unique(filtered_sequences)

    return filtered_sequences


def calculate_E_min_E_max_logical(data=None, problem_dict=None, jij=None):
    def logical_energy(conf, Jij):
        N = len(conf)
        i_upper, j_upper = np.triu_indices(N, k=1)
        E = np.einsum('i,i,i', conf[i_upper], Jij, conf[j_upper])
        return E

    if problem_dict is None and jij is None:
        problem_dict = data["problem_dictionary"]

    if data is None and jij is None:
        jij = problem_dict["local_fields"]
    Np = len(jij)
    Nl = n_logical(Np)

    config_list = generate_configs(Nl)
    expectation_val_list = np.array([logical_energy(config, jij) for config in config_list])

    return min(expectation_val_list), max(expectation_val_list)


def calculate_approximation_ratio(energy, std_dev=None, problem_dict=None, logical_qaoa=True):
    rescaled_std_dev = None
    if logical_qaoa:
        E_min, E_max = calculate_E_min_E_max_logical(problem_dict=problem_dict)
    else:
        E_min, E_max = calculate_E_min_E_max(problem_dict=problem_dict)

    rescaled_energy = energy / E_min
    if std_dev is not None:
        rescaled_std_dev = abs(std_dev / E_min)

    return rescaled_energy, rescaled_std_dev


def calculate_approximation_ratio_simple(E, E_min):
    return E / E_min

# def calculate_one_minus_residual_energy_and_std(energy, std_dev=None, problem_dict=None, logical_qaoa=True):
#     rescaled_std_dev = None
#     if logical_qaoa:
#         E_min, E_max = calculate_E_min_E_max_logical(problem_dict=problem_dict)
#     else:
#         E_min, E_max = calculate_E_min_E_max(problem_dict=problem_dict)
#
#     rescaled_energy = 1 - (energy - E_min) / (E_max - E_min)
#     if std_dev is not None:
#         rescaled_std_dev = std_dev / (E_max - E_min)
#
#     return rescaled_energy, rescaled_std_dev
