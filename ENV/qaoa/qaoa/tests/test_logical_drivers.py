from functools import partial
from lhz.spinglass_utility import create_qutip_wf
from qaoa.operator_dicts.qiskit_logical_drivers import logical_driver_subcircuit
from qaoa.operator_dicts.qiskit_gates import standard_lhz_circuits, prepare_computational_state_circuit
from qaoa.QiskitQaoa import QiskitQaoa, single_lhz_energy
from qaoa.QutipQaoa import QutipQaoa
from lhz.qutip_hdicts import hdict_physical_not_full
from qutip import expect
import numpy as np

# n=4 logical example
#   2
#  1 4
# 0 3 5
from qaoa.result_plotter import plot_results

n_qub = 6
constraints = [[0, 1, 3], [1, 2, 3, 4], [3, 4, 5]]
cstr = 3.0
lqind = [[0, 1, 2], [0, 3, 4], [1, 3, 5], [2, 4, 5]]

jij = np.random.choice([-1, 1], size=n_qub)
jij = np.array([1, -1, 1, -1, -1, -1])

standard_circuits = standard_lhz_circuits(n_qub, jij, constraints)
log_driver = logical_driver_subcircuit(n_qub, lqind)
init = prepare_computational_state_circuit([1] * n_qub)
standard_circuits['L'] = log_driver
standard_circuits['I'] = init

# %%

# pstri = 'ZCX' * 2
pstri = 'I' + 'ZLZL'
pstri = 'CXCXCX' + 'ZLZLZLZL'
# pstri = 'ZXCZXCZXC'

efun = partial(single_lhz_energy, jij, constraints, cstr)
q = QiskitQaoa(6, pstri, efun, standard_circuits, )
# res = q.mc_optimization(100, return_timelines=True,
#                         measurement_functions=[np.mean, np.min],
#                         measurement_labels=['mean', 'min'])
# plot_results(res)

p0 = None
if pstri[0] == 'I':
    p0 = create_qutip_wf([1] * n_qub)
pstrit = pstri.strip('I')

hd = hdict_physical_not_full(jij, constraints, cstrength=cstr, log_line_indizes=lqind)
qq = QutipQaoa(h_dict=hd,
               program_string=pstrit, psi0=p0)

print('programstring: ', pstri)
pars = 2 * np.random.random(size=(len(pstrit),)) - 1
q.n_shots = 1e5
print('E Qiskit: ', q.objective_function(q.execute_circuit(pars)))
used_params_qutip = [pars[i] / 2 if l in ['X', 'Z', 'L']
                     else -pars[i] / cstr
                     for i, l in enumerate(pstrit)]
print('E Qutip: ', qq.objective_function(qq.execute_circuit(used_params_qutip)))
