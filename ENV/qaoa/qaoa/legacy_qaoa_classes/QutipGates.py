from qutip.qip import cnot
from qutip import qeye, tensor

eye = qeye(2)


def n_body_gate(gate, n):
    return tensor([gate] * n)


def single_gate_in_n(gate, i, n):
    return tensor(*[qeye(2)] * i, gate, *[qeye(2)] * (n - 1 - i))


def cnot_sequence(indices, n, backwards=False, switch_control_target=False):
    pairs = [(i, j) for i, j in zip(indices[:-1], indices[1:])]

    ret = n_body_gate(qeye(2), n)
    for i1, i2 in pairs if not backwards else reversed(pairs):
        if switch_control_target:
            ret = cnot(n, i2, i1) * ret
        else:
            ret = cnot(n, i1, i2) * ret

    return ret
