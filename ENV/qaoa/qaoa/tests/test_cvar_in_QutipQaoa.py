from qaoa.QutipQaoa import QutipQaoa
import matplotlib.pyplot as plt
import lhz.qutip_hdicts as lq
from qaoa.result_plotter import plot_result
import numpy as np

nl = 4
n_p = nl*(nl-1)//2
jij = 2*np.random.random(size=(n_p,))-1

hd = lq.Hdict_physical(jij)

pstr = 'ZCX' * 2

q1 = QutipQaoa(hd, pstr, jij, optimization_target='cvar_50a')
q2 = QutipQaoa(hd, pstr, jij, optimization_target='cvar_10')

plt.hist(q1.eigsysP[0], bins=15)
plt.show()


r = q1.mc_optimization(n_maxiterations=500, return_timelines=True, measurement_functions=[q1.energy_function, q1.fidelity_function, lambda x: q1.energy_function(x)-q1.cvar_expectation_value(x)])
r2 = q2.mc_optimization(n_maxiterations=500, return_timelines=True, measurement_functions=[q2.energy_function, q2.fidelity_function, lambda x: q2.energy_function(x)-q2.cvar_expectation_value(x)])

plot_result(r, 'cvar 50')
plot_result(r2, 'cvar 10')




