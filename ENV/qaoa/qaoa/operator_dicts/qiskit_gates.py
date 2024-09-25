from typing import List
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterExpression, Gate
from sympy import Symbol
import numpy as np


class R4PGate(Gate):
    """4-body Rydberg parity phase gate"""

    def __init__(self, even_2, even_4=None, odd=None, label=None):
        """Create new R4P gate."""
        if even_4 is None:
            even_4 = even_2
        if odd is None:
            odd = 0
        if label is None:
            label = 'r4p'
        super().__init__(name='r4p', num_qubits=4, params=[even_2, even_4, odd], label=label)

    # def inverse(self):
    #     r"""Return inverted R4P gate.
    #
    #     :math:`R4P(\theta,\phi,\lambda)^{\dagger} =R4P(-\theta,-\phi,-\lambda)`)
    #     """
    #     return R4PGate(-self.params[0], -self.params[1], -self.params[2])

    def to_matrix(self):
        """Return a Numpy.array for the R4P gate."""
        even_2, even_4, odd = self.params
        even_2, even_4, odd = float(even_2), float(even_4), float(odd)  # why is this?
        # 0000 0001 0010 0011
        # 0100 0101 0110 0111
        # 1000 1001 1010 1011
        # 1100 1101 1110 1111
        return np.diag([
            np.exp(1j*even_4), np.exp(1j*odd),    np.exp(1j*odd),    np.exp(1j*even_2),
            np.exp(1j*odd),    np.exp(1j*even_2), np.exp(1j*even_2), np.exp(1j*odd),
            np.exp(1j*odd),    np.exp(1j*even_2), np.exp(1j*even_2), np.exp(1j*odd),
            np.exp(1j*even_2), np.exp(1j*odd),    np.exp(1j*odd),    np.exp(1j*even_4)
        ])

    def _define(self):
        """Wrap UnitaryGate"""
        from qiskit.extensions import UnitaryGate
        q = QuantumRegister(self.num_qubits, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (UnitaryGate(self.to_matrix(), label=self.label), q[:], [])
        ]
        qc._data = rules
        self.definition = qc


class ExchangeGate(Gate):
    """
    A Gate implementing the hopping term for implementing sum constraints in QAOA

    exp(-i*theta*H) where H = sp_i * sm_j + sm_i * sp_j
    sp and sm denote the creation and annihilation operator, respectively.
    """

    def __init__(self, theta, label=None):
        if label is None:
            label = 'exchange'
        super().__init__(name='exchange', num_qubits=2, params=[theta], label=label)

    def to_matrix(self) -> np.ndarray:
        theta = float(self.params[0])
        return np.array([[1., 0.,                     0.,                     0.],
                         [0., np.cos(theta/2),        0.-1.j*np.sin(theta/2), 0.],
                         [0., 0.-1.j*np.sin(theta/2), np.cos(theta/2),        0.],
                         [0., 0.,                     0.,                     1.]])

    def _define(self):
        """Wrap UnitaryGate"""
        from qiskit.extensions import UnitaryGate
        q = QuantumRegister(self.num_qubits, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (UnitaryGate(self.to_matrix(), label=self.label), q[:], [])
        ]
        qc._data = rules
        self.definition = qc


class GGate(Gate):
    """
    A gate with parameter p implementing the matrix
        [[1-sqrt(p) -sqrt(p)],
         [sqrt(p),  sqrt(1-p)]]
    needed to prepare a 3-qubit W-state.
    """
    def __init__(self, p, label=None):
        if label is None:
            label = 'G'
        super().__init__(name='G', num_qubits=1, params=[p], label=label)

    def to_matrix(self) -> np.ndarray:
        p = float(self.params[0])
        return np.array([[np.sqrt(1-p), -np.sqrt(p)],
                         [np.sqrt(p),   np.sqrt(1-p)]])

    def _define(self):
        """Wrap UnitaryGate"""
        from qiskit.extensions import UnitaryGate
        q = QuantumRegister(self.num_qubits, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (UnitaryGate(self.to_matrix(), label=self.label), q[:], [])
        ]
        qc._data = rules
        self.definition = qc


def single_constraint_circuit(n_qubits, cstr=1.):
    c = Parameter('c')
    co_circ = QuantumCircuit(n_qubits, name='co')
    qubits = list(range(n_qubits))

    chain = QuantumCircuit(n_qubits)
    for i in range(n_qubits - 1):
        chain.cx(i, i + 1)

    co_circ.append(chain.to_instruction(), range(n_qubits))
    co_circ.rz(phi=c*cstr, qubit=qubits[-1])
    co_circ.append(chain.inverse().to_instruction(), range(n_qubits))

    return co_circ, c


def r4p_constraint_circuit(n_qubits, constraints, parameter_name='pname', split_two_four=False,
                           bind_four_to_two_function=None):
    """
    TODO
    :param n_qubits:
    :param constraints:
    :param parameter_name:
    :param split_two_four:
    :param bind_four_to_two_function:
    :return:
    """
    circ = QuantumCircuit(n_qubits)

    if bind_four_to_two_function is not None:
        p2 = Parameter(parameter_name)
        x = Symbol('x')
        p4 = ParameterExpression({p2: x}, bind_four_to_two_function(x))
    elif split_two_four and bind_four_to_two_function is None:
        p2 = Parameter(parameter_name + '2')
        p4 = Parameter(parameter_name + '4')
    else:
        p2 = Parameter(parameter_name)
        p4 = p2

    for co in constraints:
        r4g = R4PGate(p2, p4)
        circ.append(r4g, co)

    circ.name = 'r4p_co'
    return circ


def constraint_circuit(n_qubits, constraints, costr=1., parameter_name='pname'):
    circ = QuantumCircuit(n_qubits)
    parameter = Parameter(parameter_name)

    if type(costr) is float or type(costr) is int:
        costr_list = [costr] * len(constraints)
    else:
        assert len(constraints) == len(costr), "length of constraints must match length of costr!"
        costr_list = costr

    for j, cons in enumerate(constraints):
        temp_co, param = single_constraint_circuit(len(cons), costr_list[j])
        circ.append(temp_co.to_instruction({param: parameter}), cons)

    circ.name = 'constraints'
    return circ


def sum_constraint_exchange_circuit(n_qubits, sum_constraints, parameter_name='param'):

    circ = QuantumCircuit(n_qubits)
    parameter = Parameter(parameter_name)
    for cons in sum_constraints:
        hubbard_gate = ExchangeGate(parameter)
        circ.append(hubbard_gate, cons)

    return circ


def sum_constraint_transverse_field_circuit(n_qubits, sum_constraints, parameter_name='param'):
    circ = QuantumCircuit(n_qubits)
    parameter = Parameter(parameter_name)
    qubits_in_constraints = [q for interaction in sum_constraints for q in interaction]
    circ.rx(parameter, qubit=[q for q in range(n_qubits) if q not in qubits_in_constraints])
    return circ


def sum_constraint_circuit(n_qubits, sum_constraints, parameter_name='param'):
    circ = QuantumCircuit(n_qubits)
    parameter = Parameter(parameter_name)

    qubits_in_constraints = [q for interaction in sum_constraints for q in interaction]

    circ.rx(parameter, qubit=[q for q in range(n_qubits) if q not in qubits_in_constraints])
    for cons in sum_constraints:
        hubbard_gate = ExchangeGate(parameter)
        circ.append(hubbard_gate, cons)

    return circ


def single_rotation_circuit(n_qubits, direction='X'):
    circ = QuantumCircuit(n_qubits)
    parameter = Parameter('p' + direction)

    if direction == 'Z':
        circ.rz(parameter, qubit=range(n_qubits))
    elif direction == 'X':
        circ.rx(parameter, qubit=range(n_qubits))

    circ.name = 'rot' + direction
    return circ


def local_field_circuit(jij):
    circ = QuantumCircuit(len(jij))

    param = Parameter('pZ')
    for i, field in enumerate(jij):
        circ.rz(param*field, qubit=i)

    circ.name = 'loc_fields'
    return circ


def interaction_circuit(n_qubits, jij, interactions):
    circ = QuantumCircuit(n_qubits)
    param = Parameter('pP')
    assert len(jij) == len(interactions), "Number of couplings does not coincide with length of jij!"
    for i, int in enumerate(interactions):
        # consider higher order interactions
        for j in range(len(int)-1):
            circ.cx(int[j], int[j+1])
        circ.rz(jij[i]*param, qubit=int[-1])
        for j in range(len(int)-1)[::-1]:
            circ.cx(int[j], int[j+1])

    circ.name = 'interactions'
    return circ


def prepare_w_state_circuit(n_qubits, qubits):
    """

    :param n_qubits: the total number of qubits involved in the circuit
    :param qubits: The indices of qubits wanted to be in a W-state (must be a list of 3 qubits)
    :return: The corresponding QuantumCircuit
    """
    # cf. https://physics.stackexchange.com/questions/311743/quantum-circuit-for-a-3-qubit-w-rangle-state

    circ = QuantumCircuit(n_qubits)
    circ.append(GGate(1/3.), [qubits[0]])
    circ.x(qubits[0])
    circ.ch(qubits[0], qubits[1])
    circ.x(qubits[1])
    circ.toffoli(*qubits)
    circ.x(qubit=[qubits[0], qubits[1]])
    return circ


def prepare_computational_state_circuit(desired_state: List) -> QuantumCircuit:
    """
    :param desired_state: Desired computational state to prepare. Format: [-1, 1, 0, 1, ...];
        -1 will prepare groundstate of sigma_z, (bit=1 in qiskit),
         1 the excited state of sigma_z (bit=0 in qiskit),
         0 the symmetric superposition |+>
    :return: Circuit preparing the desired state
    """
    qc = QuantumCircuit(len(desired_state))

    for i, x in enumerate(desired_state):
        if x == 1:
            pass
        elif x == -1:
            qc.x(i)
        else:
            qc.h(i)
    return qc


def standard_lhz_circuits(n_qubits, local_fields, constraints, costr=1.):
    return {'X': single_rotation_circuit(n_qubits, 'X'),
            'Z': local_field_circuit(local_fields),
            'C': constraint_circuit(n_qubits, constraints, costr)}


def standard_gatemodel_circuits(n_qubits, interaction_strengths, couplings):
    return{'X': single_rotation_circuit(n_qubits, 'X'),
           'P': interaction_circuit(n_qubits, interaction_strengths, couplings)}


def r4p_lhz_circuits(n_qubits, local_fields, constraints, split_two_four=False, bind_four_to_two_function=None):
    return {'X': single_rotation_circuit(n_qubits, 'X'),
            'Z': local_field_circuit(local_fields),
            'C': r4p_constraint_circuit(n_qubits, constraints,
                                        split_two_four=split_two_four,
                                        bind_four_to_two_function=bind_four_to_two_function)}


def sumconstraints_lhz_circuits(n_qubits, sum_constraints, local_fields, constraints, init_state=None):
    if init_state is None:
        init_state = [-1]*n_qubits
    return {'I': prepare_computational_state_circuit(init_state),
            'X': sum_constraint_circuit(n_qubits, sum_constraints),
            'E': sum_constraint_exchange_circuit(n_qubits, sum_constraints),
            'T': sum_constraint_transverse_field_circuit(n_qubits, sum_constraints),
            'Z': local_field_circuit(local_fields),
            'C': constraint_circuit(n_qubits, constraints)}


if __name__ == '__main__':
    n_p = 6
    cos = [{0, 1, 3}, {1, 2, 3, 4}, {3, 4, 5}]
    jij = [1, 2, 3, 4, 5, 6]

    larger_qc = constraint_circuit(n_p, cos, 'blub')
    larger_qc.append(local_field_circuit(jij), range(n_p))

    larger_qc.measure_all()

    bound = larger_qc.bind_parameters({list(larger_qc.parameters)[0]: 0.1})

    print(bound.draw())

    print(larger_qc.decompose().decompose().draw())
    print(larger_qc.decompose().decompose().count_ops(), larger_qc.depth())
