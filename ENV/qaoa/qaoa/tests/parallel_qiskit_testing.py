# import psutil
import argparse
from lhz.core import standard_constraints
from qaoa.QiskitQaoa import QiskitQaoa, single_lhz_energy
from qaoa.operator_dicts.qiskit_gates import standard_lhz_circuits
from functools import partial
from time import time
import numpy as np
import qiskit.providers.aer.noise as noise

# import os
# what? TODO does this not fuck up everything?
# os.environ['OMP_NUM_THREADS'] = str(1)
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

args = parser.parse_args()

max_threads = args.maxthreads
n_meas = args.nmeasure
n_l = args.nlogical
n_runs = args.repetitions
noise_flag = args.noise

n_p = n_l * (n_l - 1) // 2
cs = standard_constraints(n_l)

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

# cs += [[20, 21, 22], [22, 23, 24]]
# n_p = 25
# n_p = 20
# cs = csdict[n_p]
cstr = 3.0

jij = [1, 2, 1, -2, 1, -1] * 5
jij = jij[:n_p]
pstr = 'ZXCX' * 2
pstr = 'ZCX' * 3
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

print(args)

for _ in range(n_runs):
    q = QiskitQaoa(len(jij), pstr, efun, subcircs, num_measurements=n_meas, backend_options=bo)
    if noise_flag:
        q.noise_model = noise_model
    t0 = time()
    q.execute_circuit(parameters)
    t_end = time() - t0
    temp_timing.append(t_end)

if n_runs > 1:
    print("mean: %.2fs, std: %.2fs" % (np.mean(temp_timing), np.std(temp_timing)))
else:
    print("duration: %.2fs" % temp_timing[0])
print('----------------------------')

# print("%.2f MB" % (psutil.Process().memory_info().vms/1e6))
# print("%.2f MB" % (psutil.Process().memory_info().rss/1e6))
