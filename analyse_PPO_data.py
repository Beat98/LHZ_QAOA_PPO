import os
import pprint
from collections import Counter
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from functions import (
    get_data_list, moving_average, save_data, calculate_approximation_ratio
)

# Plot configuration
sns.set_style('ticks')
plt.rcParams.update({
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 18,
    'axes.grid': False,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'axes.xmargin': 0,
    'axes.ymargin': 0.02
})

pp = pprint.PrettyPrinter(indent=1, width=100)


def load_and_select_data(data_set_name, num_executions, selection=None):
    """
    Load experiment data and optionally select certain experiment executions for processing.

    Args:
        data_set_name (str): The name of the data set.
        num_executions (int): Number of executions of the experiment.
        selection (list, optional): List of indices to select specific experiment executions.

    Returns:
        list: Processed list of data entries.
    """

    # List of the data of the selected experiment executions. Type List[Dict[...], Dict[...]]
    data_list = get_data_list(data_set_name, num_executions=num_executions)
    if selection is not None:
        data_list = np.array(data_list)[selection]
    return data_list


def extract_and_process_data(data_list, end, save):
    """
    Extract energies and sequences from the data and extract interesting statistics.

    Args:
        data_list (list): List of experiment data.
        end (int): The endpoint to slice data.

    Returns:
        tuple:
            * data_dict (dict): including interesting experiment data
            * energy_matrix (np.array): matrix of the QAOA energies from all selected experiments

    """
    best_sequences, final_sequences = [], []
    final_energies, best_energies = [], []
    learning_times, num_unique_seq = [], []
    energy_matrix = []

    # Iterate through each experiment execution data
    for data in data_list:
        # Extract and slice the QAOA energy and sequences data up to the specified 'end' index
        qaoa_energy = data['min_energy'][:end]
        energy_matrix.append(qaoa_energy)
        sequences = data['gate_sequences'][:end]

        # Identify unique sequences and record their count
        unique_seq = np.unique(sequences)
        num_unique_seq.append(len(unique_seq))

        # Find the index of the minimum QAOA energy (best energy) in the sliced data
        min_energy_idx = np.argmin(qaoa_energy)

        # Record the best sequence and corresponding energy
        best_sequences.append(sequences[min_energy_idx])
        best_energies.append(qaoa_energy[min_energy_idx])

        # Record the final (last) sequence and corresponding energy
        final_sequences.append(sequences[-1])
        final_energies.append(qaoa_energy[-1])

        # Track the learning time for this experiment
        learning_times.append(data["learning_time"])

    # Count the occurrences of each best sequence and final sequence across all experiments
    best_sequences_counts = dict(Counter(best_sequences))
    final_sequences_counts = dict(Counter(final_sequences))

    # Extract additional experiment information
    experiment_info = {key: data_list[0][key] for key in data_list[0].keys() if
                       key not in ['min_energy', 'gate_sequences', 'gate_length', 'reward', 'learning_time']}

    # Constructing the data_dict with metrics and statistics
    data_dict = OrderedDict({
        'experiment_info': experiment_info,
        'avg_quantum_calls': np.mean(num_unique_seq),
        'avg_learning_time': np.mean(learning_times),
        'final_seq_with_energy': {
            seq: [energy for f_seq, energy in zip(final_sequences, final_energies) if f_seq == seq]
            for seq in final_sequences
        },
        'final_sequences_counts': final_sequences_counts,
        'best_seq_with_energy': {
            seq: [energy for b_seq, energy in zip(best_sequences, best_energies) if b_seq == seq]
            for seq in best_sequences
        },
        'best_sequences_counts': best_sequences_counts,
        'num_unique_seq_with_final_energy_and_sequence': list(
            zip(num_unique_seq, np.round(final_energies, 3), final_sequences)),
    })

    if save:
        data_dict['experiment_info']['policy'] = str(data_dict['experiment_info']['policy'])
        save_data(
            folder_name=data_set_name,
            filename="processed_data",
            data=data_dict
        )
    return data_dict, energy_matrix


def plot_data(ax, energy_matrix, color, window_length, save):
    """
    Plot experiment data and save it if specified.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot data on.
        color (str): Color for the plot.
        window_length (int): Window length for moving average.
        save (bool): Whether to save the plot.

    Returns:
        dict: Processed data dictionary.
    """

    # Calculate mean energies
    mean_energies = np.mean(energy_matrix, axis=0)
    logical_qaoa = optimization_target == "logical_energy"

    # Calculate approximation ratio
    mean_energies, _ = calculate_approximation_ratio(
        mean_energies, problem_dict=data_list[0]["problem_dict"], logical_qaoa=logical_qaoa
    )

    xs = np.arange(len(mean_energies))
    ax.plot(xs, mean_energies, linestyle="None", marker=".", alpha=0.3, markersize=2)

    # Calculate and plot the moving average
    moving_averages = moving_average(mean_energies, window_length)
    ax.plot(xs, moving_averages, label=f'moving average', color=color)

    # Set axis labels and grid
    ax.set_xlabel('Episodes')
    ax.set_ylabel(r'Approximation ratio $r$')
    ax.grid(True)
    plt.tight_layout(pad=0.1)
    plt.legend()

    if save:
        plt.savefig(os.path.join("data", data_set_name), dpi=400)
    plt.show()


# --- Main Execution Block ---
if __name__ == "__main__":
    # Parameters and configurations
    optimization_target = "logical_energy"
    final_decoding = "logical_mean"
    num_qubits = 6
    num_executions = 1
    reps = 10
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    selection = None
    end = 2500

    # Plot initialization
    fig, ax = plt.subplots(figsize=(9, 5))
    data_set_name = "test_data_set"

    # Load and select data
    data_list = load_and_select_data(data_set_name, num_executions, selection)

    # Process data and extract interesting statistics
    data_dict, energy_matrix = extract_and_process_data(data_list, end, save=True)

    # Plot data
    plot_data(ax, energy_matrix, colors[0], window_length=200, save=True)

    # Print the data_dict including statistics and experiment information
    pp.pprint(data_dict)


