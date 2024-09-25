from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
import seaborn as sns

pp = pprint.PrettyPrinter(indent=1, width=100)
from functions import get_data_list, moving_average, save_data, calculate_approximation_ratio, load_data, \
    get_triv_sequence, get_standard_sequence, calculate_approximation_ratio_simple

sns.set_style('ticks')
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.grid'] = False
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0.02


def get_visualize_and_process_data(ax, color, data_set_name, experiment_name, window_length=100, plot=False, save=False,
                                   end=-1):

    # load data
    data_list = get_data_list(os.path.join(data_set_name, experiment_name), num_executions=num_executions)
    # selection of data
    if selection is not None:
        data_list = np.array(data_list)
        data_list = data_list[selection]

    best_sequences, final_sequences, final_energies, best_energies, learning_times, num_unique_seq = [], [], [], [], [], []
    energy_matrix = []
    for i, data in enumerate(data_list):
        qaoa_energy = data['min_energy'][:end]
        energy_matrix.append(qaoa_energy)

        sequences = data['gate_sequences'][:end]
        unique_seq = np.unique(sequences)
        num_unique_seq.append(len(unique_seq))

        # Track the best gate sequence in this learning process
        min_energy_idx = np.argmin(qaoa_energy)

        best_sequence = sequences[min_energy_idx]
        best_energy = qaoa_energy[min_energy_idx]
        best_sequences.append(best_sequence)
        best_energies.append(best_energy)

        # Track the final gate sequence in this learning process
        final_sequences.append(sequences[-1])
        final_energies.append(qaoa_energy[-1])

        learning_times.append(data["learning_time"])

    # Count occurrences of each sequence
    best_sequences_counts = dict(Counter(best_sequences))
    final_sequences_counts = dict(Counter(final_sequences))


    mean_energies = np.mean(energy_matrix, axis=0)
    if plot:

        # for n, mean_energies in enumerate(energy_matrix):

        if optimization_target == "logical_energy":
            logical_qaoa = True
        else:
            logical_qaoa = False

        mean_energies, _ = calculate_approximation_ratio(mean_energies,
                                                     problem_dict=data_list[0]["problem_dict"],
                                                     logical_qaoa=logical_qaoa)
        xs = np.arange(len(mean_energies))
        ax.plot(xs, mean_energies, linestyle="None", marker=".", alpha=0.3, markersize=2)  # ,color=color)#, label='PPO agent mean')

        moving_averages = moving_average(mean_energies, window_length)
        ax.plot(xs, moving_averages, label=f'l = {l}', color=color)  # label=f'execution {n}'
        # ax.set_xticks(xs)
        # ax.set_xticklabels(sequences, rotation='vertical')
        # ax.set_title(rf"$K= {num_qubits}$")
        ax.set_xlabel('episodes')
        ax.set_ylabel(r'approximation ratio $r$')
        # ax.set_ylabel(r'$E_{qaoa}/E_{GS}$')

        # ax.set_ylim([0.05,0.7])

        ax.grid(True)
        plt.tight_layout(pad=0.1)
        if save:
            plt.savefig(os.path.join("data", data_set_name, experiment_name), dpi=400)

    experiment_info = {key: data_list[0][key] for key in data_list[0].keys() if
                       key not in ['min_energy', 'gate_sequences', 'gate_length', 'reward']}

    data_dict = {
        'num_unique_seq_with_final_energy_and_sequence': list(
            zip(num_unique_seq, np.round(final_energies, 3), final_sequences)),
        'avg_quantum_calls': np.mean(num_unique_seq),
        'avg_learning_time': np.mean(learning_times),
        'best_sequences_counts': best_sequences_counts,
        'final_sequences_counts': final_sequences_counts,
        'final_seq_with_energy': {
            seq: [energy for f_seq, energy in zip(final_sequences, final_energies) if f_seq == seq]
            for seq in final_sequences
        },
        'best_seq_with_energy': {
            seq: [energy for b_seq, energy in zip(best_sequences, best_energies) if b_seq == seq]
            for seq in best_sequences
        },
        'experiment_info': experiment_info
    }
    if save:
        data_dict['experiment_info']['policy'] = str(data_dict['experiment_info']['policy'])
        save_data(folder_name=os.path.join(data_set_name, experiment_name), filename="processed_data",
                  data=data_dict)
    return data_dict


optimization_target = "logical_energy"
final_decoding = "logical_mean"

num_qubits = 6
num_executions = 10
reps = 10

if num_qubits == 6:
    constraints = [[0, 1, 3], [1, 2, 3, 4], [3, 4, 5]]
    local_fields = [-1, -1, -1, 1, -1, -1]
    n_max_gates = 9
elif num_qubits == 10:
    constraints = [[0, 1, 4], [4, 5, 7], [7, 8, 9], [1, 2, 4, 5], [2, 3, 5, 6], [5, 6, 7, 8]]
    local_fields = [-1, -1, -1, -1, 1, -1, 1, -1, 1, -1]
    # local_fields = [-1, -1, -1, -1, 1, -1, 1, 1, 1, 1]
    n_max_gates = 6

# n_max_gates = 3
l_s = [0,1,2]
# l = 1
# n_steps_s = [25, 50, 100, 250, 500]

data_set_name = f"interesting_{num_qubits}_qubits_instance"
# data_set_name = "hyperparameter_optimization"

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

selection = None
end = 2500

fig, ax = plt.subplots(figsize=(9, 5))

for i, l in enumerate(l_s):
# for i, n_steps in enumerate(n_steps_s):
    if l == 0:
        commutator = None
    elif l == 1:
        commutator = "both"
    elif l == 2:
        commutator = "second_order_v2"
    experiment_name = os.path.join(f"commutators_{commutator}", f"num_qubits_{num_qubits}_n_max_gates_{n_max_gates}",
                                   f"local_fields_{''.join([str(f) for f in local_fields])}_num_executions_{num_executions}_reps_{reps}")  # _optimization_target_{optimization_target}_final_decoding_{final_decoding}
    # experiment_name = os.path.join("n_steps", f"num_qubits_{num_qubits}_commutator_{commutator}_num_"
    #                                           f"executions_{num_executions}_reps_{reps}", f"n_steps_{n_steps}")
    result = get_visualize_and_process_data(ax, colors[i], data_set_name, experiment_name, plot=True, save=True,
                                            end=end, window_length=200)
    pp.pprint(result)

file_name = f"num_qubits_{num_qubits}_gate_length_{n_max_gates}_reps_100_num_constraints_violated_None"
folder_name = os.path.join("non_triv_local_fields", f"symmetry_cleaned_standard_better_then_trivial")
data_non_triv_instances = load_data(folder_name, file_name)
triv_seq_energy = data_non_triv_instances[tuple(local_fields)][get_triv_sequence(n_max_gates)]
standard_seq_energy = data_non_triv_instances[tuple(local_fields)][get_standard_sequence(n_max_gates)]
E_min = data_non_triv_instances[tuple(local_fields)]["E_min"]

triv_seq_energy = calculate_approximation_ratio_simple(triv_seq_energy, E_min)
standard_seq_energy = calculate_approximation_ratio_simple(standard_seq_energy, E_min)
xs = np.arange(end)
ax.plot(xs, [triv_seq_energy for e in range(end)], 'k', linestyle="dashed", label="trivial seq.")
ax.plot(xs, [standard_seq_energy for e in range(end)], 'r', linestyle="dashed", label="standard seq.")
ax.legend()

plt.savefig(os.path.join("data", "final_plots", f"comparison_ls_num_qubits_{num_qubits}"), dpi=400)
# plt.savefig(os.path.join("data", "final_plots", f"comparison_n_steps_num_qubits_{num_qubits}"), dpi=400)

plt.show()
