import numpy as np
from scipy.sparse import csr_matrix, kron

H = csr_matrix(np.array(((1, 1), (1, -1))) / np.sqrt(2))
eye = csr_matrix(np.array(((1, 0), (0, 1))))
X = csr_matrix(np.array(((0, 1), (1, 0))))
Y = csr_matrix(np.array(((0, -1j), (1j, 0))))
Z = csr_matrix(np.array(((1, 0), (0, -1))))

# states, convention:
# u = |0>, Z u = +u
# d = |1>, Z d = -d
u = csr_matrix(np.array(((1,), (0,))))
d = X @ u
# projectors
Pu = u @ u.transpose()
Pd = d @ d.transpose()

cnot12 = csr_matrix(np.array(
    ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0))
))

cnot21 = csr_matrix(np.array(
    ((1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0))
))


# rotation gates not as usual in literature (there angle should be dividied by 2)
# but such that rz(a) = exp(-1j*Z*a) and so on.


def rz(alpha):
    return csr_matrix(np.array(((np.exp(-1j * alpha), 0), (0, np.exp(1j * alpha)))))


def rx(alpha):
    return csr_matrix(np.array(((np.cos(alpha), -1j * np.sin(alpha)), (-1j * np.sin(alpha), np.cos(alpha)))))


def ry(alpha):
    return csr_matrix(np.array(((np.cos(alpha), -np.sin(alpha)), (np.sin(alpha), np.cos(alpha)))))


def list_kronecker(*operators):
    if len(operators) == 1:
        if type(operators[0]) is list:
            if len(operators[0]) > 1:
                operators = operators[0]  # unpack list (if more than 1 element inside)
            else:
                return operators[0][0]  # return the single element of list
        elif type(operators[0]) is np.ndarray:
            return operators[0]  # single operator, return directly

    # if type(operators) == tuple and len(operators) == 1:  # check if the input was a list
    #     operators = operators[0]
    #
    # if len(operators) == 1:  # single operator, return
    #     return operators

    ret = kron(operators[0], operators[1])

    for e in operators[2:]:
        ret = kron(ret, e)

    return ret


def cnot(i1, i2, n):
    gate_list_u = [eye] * n
    gate_list_d = [eye] * n

    gate_list_u[i1] = Pu
    gate_list_d[i1] = Pd
    gate_list_d[i2] = X

    return list_kronecker(gate_list_u) + list_kronecker(gate_list_d)

    # ret_u = 1
    # ret_d = 1
    #
    # for gu, gd in zip(gate_list_u, gate_list_d):
    #     ret_u = np.kron(ret_u, gu)
    #     ret_d = np.kron(ret_d, gd)

    # return ret_d + ret_u


def n_body_gate(gate, n):
    return list_kronecker([gate] * n)
    # ret = 1
    #
    # for _ in range(n):
    #     ret = np.kron(ret, gate)
    #
    # return ret


def single_gate_in_n(gate, i, n):
    return list_kronecker(*[eye] * i, gate, *[eye] * (n - 1 - i))
    # ret = n_body_gate(eye, i)
    # ret = np.kron(ret, gate)
    # return np.kron(ret, n_body_gate(eye, n-1-i))


def cnot_sequence2(indices, n):
    dists = np.diff(indices)
    ret = n_body_gate(eye, n)

    for i, di in enumerate(dists):
        if di == 1:
            t = single_gate_in_n(cnot12, indices[i], n - 1)
            # t = n_body_gate(eye, indices[i])
            # t = np.kron(t, cnot12)
            # t = np.kron(t, n_body_gate(eye, n-1-indices[i+1]))
        elif di == -1:
            t = single_gate_in_n(cnot12, indices[i + 1], n - 1)
            # t = n_body_gate(eye, indices[i + 1])
            # t = np.kron(t, cnot21)
            # t = np.kron(t, n_body_gate(eye, n - 1 - indices[i]))
        else:
            i1, i2 = sorted(indices[i:i + 2])
            t = cnot(i1, i2, n)

        ret = t @ ret

    return ret


def cnot_sequence(indices, n, backwards=False):
    pairs = [(i, j) for i, j in zip(indices[:-1], indices[1:])]

    ret = n_body_gate(eye, n)
    for i1, i2 in pairs if not backwards else reversed(pairs):
        ret = cnot(i1, i2, n) @ ret

    return ret
