from qaoa.QaoaBase import Result
from qaoa.QiskitQaoa import QiskitQaoa, single_lhz_energy
from qaoa.operator_dicts.qiskit_gates import r4p_lhz_circuits
from functools import partial
import qiskit.providers.aer.noise as noise
from time import time
import numpy as np
import json_tricks
import os


def simulate_experiment(n_phys, constraints, local_fields, one_qubit_error_rate, four_qubit_error_rate, qaoa_sequence,
                        num_measurements, optimization_iterations, output_filename='result.json', cstr=3.0,
                        split_two_four_angle=False, optimization_repetitions=1, maxthreads=0,
                        start_parameters=None):
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

    q = QiskitQaoa(n_phys, qaoa_sequence, efun, subcircs,
                   num_measurements=num_measurements,
                   backend_options={"max_parallel_threads": maxthreads})
    q.noise_model = noisemodel

    ress = []
    for _ in range(optimization_repetitions):
        t1 = time()
        if start_parameters is None:
            q.program.set_all_parameters(0)
        else:
            q.program.linear_parameters = start_parameters.copy()

        res = q.mc_optimization(optimization_iterations, return_timelines=True, do_convergence_checks=False,
                                measurement_functions=[np.std, np.min, np.mean, lambda x: x],
                                measurement_labels=['std', 'min', 'energy explore', 'all_energies'])

        if optimization_repetitions == 1:
            res.save_to_file(output_filename)
        else:
            res.save_to_file(output_filename + '_' + str(_))

        if optimization_repetitions > 1:
            t1 = time() - t1
            print('simulation %d/%d took %.2fs' % (_ + 1, optimization_repetitions, t1))
        ress += [res]

    t0 = time() - t0
    print('total simulation for %s took %.2fs' % (output_filename, t0), flush=True)

    return ress


def tts_evaluator(results, groundstate_energy, program_string, gate_time_dict, desired_groundstate_probability=0.99):
    """
    results must include all_energies measurements to be able to calculate fidelity
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

    num_opt_steps = len(results[0].measurements[i_all_es])
    num_measurements_per_energy_expectationvalue_estimation = len(results[0].measurements[i_all_es][0])

    time_for_one_circuit_execution = sum([gate_time_dict[l] for l in program_string])

    ttss = []
    fidelities = []
    p_gss = []
    for i in range(num_opt_steps):
        counts = []

        for res in results:
            counts += [np.count_nonzero(res.measurements[i_all_es][i] == groundstate_energy)]
            # counts += np.count_nonzero(res.measurements[i_all_es][i] < 0.9*groundstate_energy)
        p_gs_arr = np.array(counts) / num_measurements_per_energy_expectationvalue_estimation
        p_gs_mean = np.mean(p_gs_arr)
        fidelities += [p_gs_mean]
        p_gss += [p_gs_arr]

        # should this be (i) or (i+1)   (I think i, because the 0th in optimization actually does not optimize)
        # also check with this weird line if its a refined_result or a normal one
        # although I did that before also already by taking to opt-times from the not refined results
        num_opt_step = results[0].opt_steps[i] if 'opt_steps' in res.__dict__.keys() else i  # + 1
        tts_optimizing = num_opt_step * num_measurements_per_energy_expectationvalue_estimation * time_for_one_circuit_execution
        if p_gs_mean == 0:
            tts_find_sol = np.inf
        else:
            tts_find_sol = time_for_one_circuit_execution * np.log(1 - desired_groundstate_probability) / np.log(1 - p_gs_mean)
        ttss += [(tts_optimizing, tts_find_sol)]

    return ttss, fidelities, p_gss


def refine_result(n_phys, constraints, local_fields, one_qubit_error_rate, four_qubit_error_rate, qaoa_sequence,
                  num_measurements, input_filename, opt_steps_to_repeat, cstr=3.0,
                  split_two_four_angle=False, maxthreads=0):
    t0 = time()
    if os.path.exists(input_filename + '_refined'):
        print('already refined')
        with open(input_filename + '_refined') as f:
            str_temp = f.read()
        return json_tricks.loads(str_temp)
    noisemodel = noise.NoiseModel()

    # Depolarizing quantum errors
    error_1 = noise.depolarizing_error(one_qubit_error_rate, 1)
    error_4 = noise.depolarizing_error(four_qubit_error_rate, 4)

    # Add errors to noise model
    noisemodel.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noisemodel.add_all_qubit_quantum_error(error_4, ['r4p'])

    efun = partial(single_lhz_energy, local_fields, constraints, cstr)
    subcircs = r4p_lhz_circuits(n_phys, local_fields, constraints, split_two_four_angle, bind_four_to_two_function=None)

    q = QiskitQaoa(n_phys, qaoa_sequence, efun, subcircs,
                   num_measurements=num_measurements,
                   backend_options={"max_parallel_threads": maxthreads})
    q.noise_model = noisemodel

    with open(input_filename) as f:
        res_string = f.read()

    res = json_tricks.loads(res_string)

    refined_result = Result()
    refined_result.opt_steps = list(opt_steps_to_repeat)
    refined_result.measurement_labels = ['all_energies']

    all_energies = []
    for opt_step in opt_steps_to_repeat:
        parameters = res.all_parameters[opt_step]
        all_energies += [q.execute_circuit(parameters)]

    refined_result.measurements = []
    refined_result.measurements += [all_energies]

    refined_result.save_to_file(input_filename + '_refined')

    t0 = time() - t0
    print('total refinement for %s took %.2fs' % (input_filename, t0), flush=True)

    return refined_result


def residual_energy_evaluation(results, e_min, e_max):
    if 'all_energies' not in results[0].measurement_labels:
        print('all energies not in result, tts evaluation not possible')
        return None

    i_all_es = results[0].measurement_labels.index('all_energies')

    if 'opt_steps' in results[0].__dict__.keys():
        # refined result
        opt_steps = results[0].opt_steps
    else:
        # normal result
        opt_steps = list(range(len(results[0].measurements[i_all_es])))

    means_normalized = []
    percentiles_normalized = []
    all_energies = []
    for i_o, optstep in enumerate(opt_steps):
        temp_means = []
        temp_percentiles = []
        temp_alles = []
        for result in results:
            temp_percentiles += [np.percentile(
                result.measurements[i_all_es][i_o], q=[5, 10, 30, 50, 70, 90, 95])
            ]
            temp_means += [np.mean(result.measurements[i_all_es][i_o])]
            temp_alles += [result.measurements[i_all_es][i_o]]

        all_energies += [(np.array(temp_alles)-e_min)/(e_max-e_min)]
        temp_percentiles = np.array(temp_percentiles)
        e_means = np.array(temp_means)
        means_normalized += [(e_means-e_min)/(e_max-e_min)]
        percentiles_normalized += [(temp_percentiles-e_min)/(e_max-e_min)]

    return means_normalized, percentiles_normalized, all_energies


if __name__ == '__main__':
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

    n_p = 16
    cs = csdict[n_p]
    cstr = 4.20
    jij = np.random.choice([-1, 1], size=(n_p,))

    program_length = 2
    qaoa_sequence = 'ZXCX' * program_length

    four_qubit_erate = 0.000
    one_qubit_erate = 0.000

    n_measurements = 500
    optimization_iterations = 100
    optimization_repetitions = 2
    fname = '../reg_run_01/res_KPS_'
    fname = '/Users/kili/Documents/qaoadata/reg_run_02/data/res_uniform_0_3_'
    fname = 'test_'
    fn = 'H:\\qaoa_data\\05_run\\res_discrete_1_0_0'

    jij = [1., 1., -1., -1., 1., 1., 1., 1., 1., -1., -1., 1., -1.,
           1., 1., -1.]

    opt_steps_to_rep = [35, 40, 45, 50]

    rres = refine_result(n_p, cs, jij, 0.001, 0.001, 'ZCX' * 3, 1000, fn, opt_steps_to_rep)
