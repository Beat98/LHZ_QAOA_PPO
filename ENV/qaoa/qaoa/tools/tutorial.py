from qaoa import QutipQaoa, QiskitQaoa
from qaoa.result_plotter import plot_results
from lhz.qutip_hdicts import hdict_physical_not_full
# not full is for arbitrary graphs, the Hdict_physical is just for the fully connected triangle

from qaoa.operator_dicts.qiskit_gates import standard_lhz_circuits
from qaoa.energy_functions.qiskit_efuns.physical_energy import create_efun_physical


# graph
# 0 - 1 - 2
# |   | /
# 3 - 4

num_physical_qubits = 5
constraints = [[0, 1, 3, 4], [1, 2, 4]]
constraint_strength = 3.0
local_fields = [-1, 1, 0.4, 0.8, -0.9]

# program string = sequence of unitaries
pstr = 'ZCX' * 3

# qutip
hdict = hdict_physical_not_full(jij=local_fields, constraints=constraints, cstrength=constraint_strength)
q_qutip = QutipQaoa(h_dict=hdict, program_string=pstr,
                    do_prediagonalization=True,   # for bigger systems (n_phys > 10) its better/faster to not do this
                    psi0=None,  # you can define the starting wavefunction, in qutip this is done at this point
                    )

# qiskit
sub_circuits = standard_lhz_circuits(n_qubits=num_physical_qubits, local_fields=local_fields,
                                     constraints=constraints, costr=constraint_strength)
optimization_target = create_efun_physical(jij=local_fields, constraints=constraints, costr=constraint_strength)
q_qiskit = QiskitQaoa(n_qubits=num_physical_qubits, programstring=pstr, energy_function=optimization_target,
                      subcircuits=sub_circuits,
                      noise_model=None,
                      do_default_state_init=True,  # if you want to define the starting wavefunction the sub_circuits should include a circuit for 'I'
                      )

# optimization (same for qutip/qiskit as this is defined in the QaoaBase class)
result = q_qiskit.mc_optimization(n_maxiterations=10)
result_container = q_qiskit.multiple_random_init_mc_opt(reps=3, n_maxiterations=10,
                                                        return_timelines=True,  # needed if you want to plot
                                                        measurement_functions=None,  # easier to look at in qutip for now
                                                        )

result_container_qutip = q_qutip.mc_optimization(n_maxiterations=200,
                                                 return_timelines=True,
                                                 measurement_functions=[q_qutip.expectation_value_function('Nc')],
                                                 measurement_labels=['number of violated constraints'])

plot_results(result_container, title='qiskit sim', legend=['1', '2', '3'])
plot_results(result_container_qutip, title='qutip simulation',
             filename=None,  # if you want to save to a picture directly
             show=True,
             )
