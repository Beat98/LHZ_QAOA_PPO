from qaoa.legacy_qaoa_classes.QutipGateLHZQaoa import QutipGateLHZQaoa
from qaoa.result_plotter import plot_results
import numpy as np

from lhz.spinglass_utility import all_confs, create_qutip_wf, translate_configuration_logical_to_physical


# def observation_result_as_string(program_string, measurement_list):
#     strr = 'program:         ' + '       '.join('0' + program_string)
#     for mline in measurement_list:
#         strr += '\nmeasurement:  ' + ', '.join(['%+.3f' % p for p in mline])
#     return strr

check = lambda x: 'same' if not np.any(np.abs(x.data.toarray()) > 0) else 'different'
# %%
nl = 4
n = nl * (nl - 1) // 2

all_log = [create_qutip_wf(translate_configuration_logical_to_physical(c)) for c in all_confs(nl)]
suplog = sum(all_log)
suplog = suplog / suplog.norm()

Jij = [1, -0.56, 2, -1, 1, -1] * 5
# Jij = [0, 0, 0, 0, 0, 0]
Jij = Jij[:n]

pstr = 'ZXCVZV'

pstr = 'CXCXCXLXLX'
pstr = 'VZVZVZ'
# pstr = 'LXLXLX'
pstr = 'CLZCLZCLZ'
# pstr = 'L'

qL = QutipGateLHZQaoa(pstr, Jij, psi0=suplog)
qL = QutipGateLHZQaoa(pstr, Jij)
mfs = [qL.expectation_value_function('Nc'), qL.expectation_value_function('P')]
r = qL.mc_optimization(n_maxiterations=900, temperature=0.002, par_change_range=1, return_timelines=True,
                       measurement_functions=[qL.expectation_value_function('Nc')])

plot_results(r)

state, measurements, strr = qL.execute_and_observe(mfs, return_string=True)  #, parameters=[1, 0, 0, 0, 0, 0])

print(strr)

for st in all_log:
    pass

x0 = qL._parallel_logical_x(0).data.toarray()
x1 = qL._parallel_logical_x(np.pi/2).data.toarray()

c0 = qL._logical_x(0).data.toarray()
c1 = qL._logical_x(np.pi/3).data.toarray()

