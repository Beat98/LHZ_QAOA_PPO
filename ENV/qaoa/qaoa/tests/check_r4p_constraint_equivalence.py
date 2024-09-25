from qaoa.operator_dicts.qiskit_gates import R4PGate, single_cnot_constraint_circuit, r4p_constraint_circuit
from qiskit import Aer, execute

b = Aer.get_backend('unitary_simulator')
cos = [[0, 1, 2, 3]]

c, p = single_cnot_constraint_circuit(4)
cb = c.bind_parameters({p: 1})
uc = execute(cb, b).result()
ucc = uc.get_unitary()
phases_c = ucc.diagonal()

g = R4PGate(0, 0, 1)
ugr = g.to_matrix()
phases_g = ugr.diagonal()
r = r4p_constraint_circuit(4, cos, split_two_four=False)
rb = r.bind_parameters({list(r.parameters)[0]: 1})
ur = execute(rb, b).result()
urr = ur.get_unitary()
phases_r = urr.diagonal()
