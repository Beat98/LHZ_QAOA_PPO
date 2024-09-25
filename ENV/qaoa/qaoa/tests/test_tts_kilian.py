from qaoa.QiskitQaoa import QiskitQaoa, single_lhz_energy
from qaoa.operator_dicts.qiskit_gates import r4p_lhz_circuits
from functools import partial
import qiskit.providers.aer.noise as noise
from time import time
import numpy as np
from qaoa.QaoaBase import Result


def simulate_experiment(n_phys, constraints, local_fields, one_qubit_error_rate, four_qubit_error_rate, qaoa_sequence,
                        num_measurements, optimization_iterations, output_filename='result.json', cstr=3.0,
                        split_two_four_angle=False, optimization_repetitions=1):
    t0 = time()
    noisemodel = noise.NoiseModel()

    # Depolarizing quantum errors
    error_1 = noise.depolarizing_error(one_qubit_error_rate, 1)
    error_4 = noise.depolarizing_error(four_qubit_error_rate, 4)

    # Add errors to noise model
    noisemodel.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noisemodel.add_all_qubit_quantum_error(error_4, ['r4p'])

    efun = partial(single_lhz_energy, local_fields, constraints, cstr)
    subcircs = r4p_lhz_circuits(n_phys, local_fields, constraints, split_two_four_angle, bind_four_to_two_function=None)

    q = QiskitQaoa(n_phys, qaoa_sequence, efun, subcircs, num_measurements=num_measurements)
    q.noise_model = noisemodel

    ress = []
    # e_fin_best = np.inf
    # i_best = 0
    for _ in range(optimization_repetitions):
        t1 = time()
        q.program.set_all_parameters(0)
        res = q.mc_optimization(optimization_iterations, return_timelines=True, do_convergence_checks=False,
                                measurement_functions=[np.std, np.min, np.mean, lambda x: x],
                                measurement_labels=['std', 'min', 'energy explore', 'all_energies'])

        res.save_to_file(output_filename + '_' + str(_))
        t1 = time() - t1
        print('simulation %d/%d took %.2fs' % (_+1, optimization_repetitions, t1))
        ress += [res]

        # if ress[_].final_objective_value < e_fin_best:
        #     e_fin_best = ress[_].final_objective_value
        #     i_best = _

    # save_counter = 0
    # for ir, res in enumerate(ress):
    #     if ir == i_best:
    #         res.save_to_file(output_filename + 'b')
    #     else:
    #         res.save_to_file(output_filename + str(save_counter))
    #         save_counter += 1

    t0 = time() - t0
    print('total simulation for %s took %.2fs' % (output_filename, t0))

    return ress


def tts_evaluator(results, groundstate_energy, program_string, gate_time_dict, desired_groundstate_probability=0.99):
    """
    results must include all measurements to be able to calculate fidelity
    :param desired_groundstate_probability:
    :param results:
    :param groundstate_energy:
    :param program_string:
    :param gate_time_dict:
    :return:
    """
    unique_letters = list(set(program_string))

    for ul in unique_letters:
        if ul not in gate_time_dict.keys():
            gate_time_dict[ul] = 0

    if 'all_energies' not in results[0].measurement_labels:
        print('all energies not in result, tts evaluation not possible')
        return None

    i_all_es = results[0].measurement_labels.index('all_energies')

    num_opt_steps = len(results[0].objective_values)
    num_measurements_per_energy_expectationvalue_estimation = len(results[0].measurements[i_all_es][0])

    time_for_one_circuit_execution = sum([gate_time_dict[l] for l in program_string])

    ttss = []
    for i in range(num_opt_steps):
        counts = 0
        for res in results:
            counts += np.count_nonzero(res.measurements[i_all_es][i] == groundstate_energy)
        p_gs = counts / (num_measurements_per_energy_expectationvalue_estimation * len(results))

        tts_optimizing = (i + 1) * num_measurements_per_energy_expectationvalue_estimation * time_for_one_circuit_execution
        if p_gs == 0:
            tts_find_sol = np.inf
        else:
            tts_find_sol = time_for_one_circuit_execution * np.log(1 - desired_groundstate_probability) / np.log(1 - p_gs)
        ttss += [(tts_optimizing, tts_find_sol)]

    return ttss


if __name__ == '__main__':
    from qaoa.result_plotter import plot_results
    import matplotlib.pyplot as plt

    csdict = {9: [[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7], [4, 5, 7, 8]],
              16: [[0, 1, 4, 5], [1, 2, 5, 6], [2, 3, 6, 7],
                   [4, 5, 8, 9], [5, 6, 9, 10], [6, 7, 10, 11],
                   [8, 9, 12, 13], [9, 10, 13, 14], [10, 11, 14, 15]],
              20: [[0, 1, 4, 5], [1, 2, 5, 6], [2, 3, 6, 7],
                   [4, 5, 8, 9], [5, 6, 9, 10], [6, 7, 10, 11],
                   [8, 9, 12, 13], [9, 10, 13, 14], [10, 11, 14, 15],
                   [12, 13, 16, 17], [13, 14, 17, 18], [14, 15, 18, 19]],
              25: [[0, 1, 5, 6], [1, 2, 6, 7], [2, 3, 7, 8], [3, 4, 8, 9],
                   [5, 6, 10, 11], [6, 7, 11, 12], [7, 8, 12, 13], [8, 9, 13, 14],
                   [10, 11, 15, 16], [11, 12, 16, 17], [12, 13, 17, 18], [13, 14, 18, 19],
                   [15, 16, 20, 21], [16, 17, 21, 22], [17, 18, 22, 23], [18, 19, 23, 24]]}

    # n_p = 20
    n_p = 9
    cs = csdict[n_p]
    cstr = 4.20

    # jij = np.random.choice([-1, 1], size=(n_p,))
    jij = [-1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 1]
    n_p = len(jij)
    gs_energy = -12.
    # gs_energy = -4.452156241007393

    # jij = [-1, -1, 1, 1, 1, 1, -1, 1, -1]
    # gs_energy = -5.

    program_length = 1
    qaoa_sequence = 'ZXCX' * program_length

    four_qubit_error_rate = 0.000
    one_qubit_error_rate = 0.000

    num_measurements = 500
    optimization_iterations = 50
    optimization_repetitions = 50
    # fname = '../reg_run_01/res_KPS_'
    # fname = '/Users/kili/Documents/qaoadata/reg_run_02/data/res_uniform_0_3_'
    fname = 'test_'

    calculate = True

    if calculate:
        rs = simulate_experiment(n_p, cs, jij, one_qubit_error_rate, four_qubit_error_rate,
                                 qaoa_sequence, num_measurements, optimization_iterations, output_filename=fname,
                                 optimization_repetitions=optimization_repetitions)
    else:
        rs = []
        from qaoa.QaoaBase import Result
        for _ in range(2):
            rs += [Result.load_from_file(fname+str(_))]

    plot_results(rs, skip_measurements=[3])

    ttss = tts_evaluator(rs, gs_energy, qaoa_sequence, {'C': 1}, desired_groundstate_probability=0.99)
    opts = [b[0] for b in ttss]
    find = [x[1] for x in ttss]
    tot_time = [a + b for a, b in ttss]
    plt.figure()
    plt.plot(range(optimization_iterations), opts, range(optimization_iterations), find, range(optimization_iterations), tot_time)
    plt.legend(['optimization', 'find sol', 'total'])
    plt.xlabel('num optimization steps')
    plt.ylabel('TTS')
    plt.show()
