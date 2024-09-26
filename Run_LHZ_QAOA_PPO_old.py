import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import matplotlib.pyplot as plt
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
from functions import generate_setup_file, save_data, mask_fn, load_data_from_csv

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

# --n_max_gates --n_timesteps --reps --num_qubits --commutator

# setup parser
parser = argparse.ArgumentParser(description="QAOA PPO")
parser.add_argument('--env_setup', default="fixed_length_reward", type=str, help="Which reward option?")
parser.add_argument('--n_max_gates', default=6, type=int, help="maximal sequence length")
parser.add_argument('--n_steps', default=50, type=int, help="steps between policy updates")
parser.add_argument('--n_timesteps', default=100, type=int, help="number of environment steps")
parser.add_argument('--num_executions', default=1, type=int, help="number of experiments")
parser.add_argument('--gamma', default=0.99, type=float, help="")
parser.add_argument('--qaoa_optimizer', default="SLSQP", type=str, help="")
parser.add_argument('--reps', default=1, type=int, help="")
parser.add_argument('--num_qubits', default=6, type=int, help="")
parser.add_argument('--constraint_strength', default=3, type=int, help="")
parser.add_argument('--optimization_target', default="logical_energy", type=str, help="")
parser.add_argument('--final_decoding', default="logical_mean", type=str, help="")
parser.add_argument('--commutator', default="l1", type=str, help="")

args = parser.parse_args()

num_executions = args.num_executions
env_setup = args.env_setup
n_max_gates = args.n_max_gates
n_steps = args.n_steps
n_timesteps = args.n_timesteps
gamma = args.gamma
qaoa_optimizer = args.qaoa_optimizer
reps = args.reps
num_qubits = args.num_qubits
constraint_strength = args.constraint_strength
optimization_target = args.optimization_target
final_decoding = args.final_decoding
commutator = args.commutator

if commutator == "l_1":
    actions = ['X', 'Z', 'C', 'A', 'B']
elif commutator == "l_2":
    actions = ['X', 'Z', 'C', 'A', 'B', 'H', 'I', 'J', 'K', 'L']
else:
    actions = None

if num_qubits == 6:
    constraints = [[0, 1, 3], [1, 2, 3, 4], [3, 4, 5]]
    local_fields = [-1, -1, -1, 1, -1, -1]
elif num_qubits == 10:
    constraints = [[0, 1, 4], [1, 2, 4, 5], [2, 3, 5, 6], [4, 5, 7], [5, 6, 7, 8], [7, 8, 9]]
    local_fields = [-1, -1, -1, -1, 1, -1, 1, -1, 1, -1]
else:
    raise "check number of qubits"


# folder where the data dictionary is stored
data_set_name = f"test_data_set"
experiment_name = "experiment_1"

problem_dict = {"num_qubits": num_qubits, "constraints": constraints, "constraint_strength": constraint_strength,
                "local_fields": local_fields}

for i in range(num_executions):

    folder_name = os.path.join(data_set_name, experiment_name, f"execution_{i}")
    path = Path(__file__).parent.joinpath("data", folder_name)

    # Dictionary to store important data
    data = {"problem_dict": problem_dict, "n_steps": n_steps, "n_timesteps": n_timesteps, "n_max_gates": n_max_gates,
            "reps": reps, "commutator": commutator,
            "env_setup": env_setup, "qaoa_optimizer": qaoa_optimizer, "optimization_target": optimization_target,
            "final_decoding": final_decoding,
            "actions": actions}

    if env_setup == "energy_threshold_reward":
        data["reward_threshold"] = reward_threshold = 0

    env = QaoaEnv(problem_dict=problem_dict, reps=reps,
                  n_max_gates=n_max_gates, qaoa_optimizer=qaoa_optimizer,
                  final_decoding=final_decoding,
                  optimization_target=optimization_target, actions=actions)

    # set up the folders for logging and saving
    log_dir = os.path.join(path, "logs")
    models_dir = os.path.join(path, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # stores additional training data in monitor.csv
    env = Monitor(env, log_dir, info_keywords=("min_energy", "gate_sequence", "qaoa_time"))

    # enable action masking
    env = ActionMasker(env, mask_fn)

    # set up the model and train it
    data["policy"] = policy = MaskableActorCriticPolicy
    model = MaskablePPO(policy, env, verbose=0, n_steps=n_steps, gamma=gamma) #TODO: , batch_size=None ??

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
    data.update(csv_data)
    generate_setup_file(folder_name, data)
    save_data(folder_name=os.path.join(data_set_name, experiment_name), filename=f"execution_{i}", data=data)

    # plt.plot(data["min_energy"], "r.")
    # plt.show()
