# import psutil
import argparse
from lhz.core import standard_constraints
from qaoa.QiskitQaoa import QiskitQaoa, single_lhz_energy
from qaoa.operator_dicts.qiskit_gates import standard_lhz_circuits
from functools import partial
from time import time
import numpy as np
import qiskit.providers.aer.noise as noise
import ray
import os


# handle console input
parser = argparse.ArgumentParser(description='parallel qiskit test')

# positional args
# parser.add_argument('seed', nargs='*', default=7, type=int, help='seed used for random and filename')

# optional args
parser.add_argument('-nl', '--nlogical', type=int, default=4)
parser.add_argument('-nm', '--nmeasure', type=int, default=1)
parser.add_argument('-mt', '--maxthreads', type=int, default=1)
parser.add_argument('-rep', '--repetitions', type=int, default=1)
parser.add_argument('-n', '--noise', action='store_true', default=False)
parser.add_argument('-ni', '--ninstances', type=int, default=1)

args = parser.parse_args()

max_threads = args.maxthreads
n_meas = args.nmeasure
n_l = args.nlogical
n_runs = args.repetitions
noise_flag = args.noise
n_instances = args.ninstances
n_cpus = args.ncpus

n_p = n_l * (n_l - 1) // 2
cs = standard_constraints(n_l)

# cs += [[20, 21, 22], [22, 23, 24]]
# n_p = 25
cstr = 3.0

jij = [1, 2, 1, -2, 1, -1] * 5
jij = jij[:n_p]
pstr = 'ZXCX' * 2
efun = partial(single_lhz_energy, jij, cs, cstr)
subcircs = standard_lhz_circuits(n_p, jij, cs)
parameters = 3 * (np.random.random(len(pstr)) - 0.5)

# Error probabilities
prob_1 = 0.001  # 1-qubit gate
prob_2 = 0.01  # 2-qubit gate

# Depolarizing quantum errors
error_1 = noise.depolarizing_error(prob_1, 1)
error_2 = noise.depolarizing_error(prob_2, 2)

# Add errors to noise model
noise_model = noise.NoiseModel()
noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
# print(noise_model)

temp_timing = []

bo = {"max_parallel_threads": max_threads}

# print(args)
os.environ['OMP_NUM_THREADS'] = str(1)


@ray.remote(num_cpus=n_cpus)
def run(i):
    for _ in range(n_runs):
        q = QiskitQaoa(len(jij), pstr, efun, subcircs,
                       num_measurements=n_meas, backend_options=bo)
        if noise_flag:
            q.noise_model = noise_model
        t0 = time()
        q.execute_circuit(parameters)
        t_end = time() - t0
        temp_timing.append(t_end)

    if n_runs > 1:
        print("%i: mean: %.2fs, std: %.2fs" % (i, np.mean(temp_timing), np.std(temp_timing)))
    else:
        print("%i: duration: %.2fs" % (i, temp_timing[0]))
    print('----------------------------')

    return


ray.init()
futures = [run.remote(i) for i in range(n_instances)]
data = ray.get(futures)

# data = [run(i) for i in range(n_instances)]
# print(data)
print('end')
# print("%.2f MB" % (psutil.Process().memory_info().vms/1e6))
# print("%.2f MB" % (psutil.Process().memory_info().rss/1e6))
