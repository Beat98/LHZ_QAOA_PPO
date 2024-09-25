from qaoa.legacy_qaoa_classes.LinalgGates import cnot_sequence, rz, single_gate_in_n

from qutip import sigmaz, tensor, qeye

import numpy as np

N = 10
N = 4
Cs = [[0, 1, 2], [1, 3], [3, 1], [1, 2, 3], [1, 3, 2]]
# Cs = [[0, 1, 2]]
# Cs = standard_constraints(Nlogical(N))
# Cs += [[1, 2, 5]]
angle = 0.2*3

# qutip
sz = sigmaz()
szs = []
for i in range(N):
    blub = [qeye(2)] * N
    blub[i] = sz
    szs.append(tensor(blub))

Cexps = []
CQds = []
for c in Cs:
    t = 1
    for ic in c:
        t *= szs[ic]
    Cexps += [(-1j*angle*(0-t)).expm()]
    CQds += [Cexps[-1].data.toarray()]

Cexp = (-1j*angle*szs[0]*szs[1]*szs[2]).expm()
# C2 = (-1j*angle*(1 + szs[0]*szs[1]*szs[2])*3).expm()
Cdata = Cexp.data.toarray()


# linalg
CLs = []
for c in Cs:
    # CLs += [cnot_sequence(list(reversed(c)), N) @ single_gate_in_n(rz(angle), c[-1], N) @ cnot_sequence(c, N)]
    CLs += [cnot_sequence(c, N, backwards=True) * single_gate_in_n(rz(-angle), c[-1], N) * cnot_sequence(c, N)]

for i in range(len(Cs)):
    d = CQds[i] - CLs[i]
    print(np.mean(abs(d)), np.std(abs(d)))

