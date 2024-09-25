from copy import deepcopy
from typing import Tuple, List, Union

from lhz.core import standard_constraints, get_hockeystick_indices
from ncvl_class import NonConstraintViolatingLine

from qutip import qeye, sigmax, tensor


def get_crossing_qubit(line_a: NonConstraintViolatingLine, line_b: NonConstraintViolatingLine) \
        -> Union[int, None]:
    """
    Returns the qubit at which NCVLs line_a and line_b are crossing. If they do not cross, the
    function will return None. If there are multiple intersection points, return value is -1.
    :param line_a:
    :param line_b:
    :return:
    """
    intersection = list(set(line_a.qubits) & set(line_b.qubits))
    if len(intersection) == 1:
        return intersection[0]
    elif len(intersection) == 0:
        return None
    else:
        print('ERROR: multiple intersection points between lines!')
        return -1


def set_rotation_qubits(ncvls: List[NonConstraintViolatingLine],
                        a_line: NonConstraintViolatingLine) -> List[NonConstraintViolatingLine]:
    """
    Sets the rotation qubits for the s_z-rotation such that the ground-state can be prepared.
    :param ncvls: A list of NCVLs.
    :param a_line: The a-line of the set of NCVLs. This is the line that can be constructed from the
        others by linear combination and does not occur in the driver explicitly.
    :return:
    """

    ncvls = deepcopy(ncvls)
    for ncvl in ncvls:
        if get_crossing_qubit(ncvl, a_line) is not None:
            ncvl.line_type = 1
            ncvl.rotation_qubit = get_crossing_qubit(ncvl, a_line)
        else:
            ncvl.line_type = 2

    current_line_type = 2

    current_lines = [ncvl for ncvl in ncvls if ncvl.line_type == current_line_type]

    while len(current_lines):
        for line in current_lines:
            crossing_qubits = []
            for second_line in current_lines:
                if second_line != line and get_crossing_qubit(line, second_line) is not None:
                    crossing_qubits.append(get_crossing_qubit(line, second_line))

            complement = list(set(line.qubits) - set(crossing_qubits))
            if len(complement):
                line.rotation_qubit = complement[0]
            else:
                line.line_type = current_line_type + 1

        current_line_type += 1
        current_lines = [ncvl for ncvl in ncvls if ncvl.line_type == current_line_type]

    return ncvls


def get_lhz_ncvls(n_l: int, manual_constraint_indices: List[int]) \
        -> Union[None, Tuple[List[NonConstraintViolatingLine], List[int]]]:
    """
    Creates the ncvls (non-constraint-violating lines) for a fully connected LHZ-architecture,
    where some constraints may be violated, i. e. they have to be enfocred manually via the
    constraint term.

    :param n_l: Then number of logical qubits in the problem (integer)
    :param manual_constraint_indices: A list of indices representing the constraints shall not be
        enforced via the driver. The indices must correspond to the array returned by
        standard_constraints().
    :return: A list of lists containing the ncvls (i. e. the driver terms) and another list of
        lists containing the constraints that have to be enforced manually.
    """

    constraints = standard_constraints(n_l)
    manual_constraints = [constraints[i] for i in manual_constraint_indices]

    enforced_constraints = [constraint for constraint in constraints
                            if constraint not in manual_constraints]

    ncvls = [get_hockeystick_indices(i, n_l)[0] for i in range(1, n_l)]

    for constraint in manual_constraints:
        # construct all neighboring index pairs along the constraint-plaquette
        pairs = list(zip(constraint, constraint[1:])) + [tuple([constraint[0], constraint[-1]])]

        # this works because the qubits in a line are sorted in ascending order for the all-to-all
        # LHZ-architecture
        pairs = [sorted(list(pair)) for pair in pairs]
        # print(pairs)
        for pair in pairs:
            split_line = None
            for line in ncvls:

                if set(pair).issubset(set(line)):
                    is_in_enforced_constraints = False
                    for enforced_constraint in enforced_constraints:
                        if set(pair).issubset(set(enforced_constraint)):
                            is_in_enforced_constraints = True
                            break

                    if not is_in_enforced_constraints:
                        part_a = line[:line.index(pair[1])]  # split line
                        part_b = line[line.index(pair[1]):]
                        if part_a not in ncvls and part_a != []:
                            ncvls.append(part_a)
                        if part_b not in ncvls and part_b != []:
                            ncvls.append(part_b)
                        split_line = line
                        break

            if split_line is not None:  # remove the original line if it was split
                ncvls.remove(split_line)

    # if there are too many or too few lines, notify the user:
    if len(ncvls) != n_l - 1 + len(manual_constraints):
        print('ERROR: The set of NCVLs is either not complete or overcomplete!')
        return

    ncvls = [NonConstraintViolatingLine(ncvl) for ncvl in ncvls]
    a_line = NonConstraintViolatingLine(get_hockeystick_indices(0, n_l)[0], line_type=0)
    ncvls = set_rotation_qubits(ncvls, a_line)

    return ncvls, manual_constraints


def construct_driver_hamiltonian(n_l: int, ncvls: List[NonConstraintViolatingLine],
                                 convention: str = 'standard'):
    """

    :param n_l: Number of logical qubits
    :param ncvls: List of NCVLs
    :param convention: Either 'standard' or 'qiskit'. Specifies whether qubits are ordered in
        stanard convention (q0q1q2...) or qiskit-convention (....q3q2q1q0).
    :return:
    """

    n_p = n_l * (n_l - 1) // 2

    si = qeye(2)
    sx = sigmax()
    sx_list = []
    hx = 0

    for n in range(n_p):
        op_list = [si] * n_p
        op_list[n] = sx
        sx_list.append(tensor(op_list))

    for ncvl in ncvls:
        term = 1
        for qubit in ncvl.qubits:
            if convention == 'qiskit':
                term *= sx_list[n_p-1-qubit]  # qiskit qubit ordering convention!
            else:
                term *= sx_list[qubit]
        hx += term
    return hx


if __name__ == '__main__':
    n_logical = 7
    manual_constr_indices = [3, 4, 8, 12]
    # manual_constr_indices = [0, 1, 2, 3, 4, 5]

    ncvl_list, constr = get_lhz_ncvls(n_logical, manual_constr_indices)
    print(ncvl_list)
    print(len(ncvl_list)-(n_logical-1+len(manual_constr_indices)))
