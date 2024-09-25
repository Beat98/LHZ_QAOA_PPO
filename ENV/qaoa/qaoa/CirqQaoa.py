import numpy as np
from lhz.core import n_logical, standard_constraints, qubit_dict
from qaoa.QaoaBase import QaoaBase
from qaoa.programs import SimpleProgram
from cirq import Circuit, LineQubit, Simulator, resolve_parameters, measure, is_measurement, Moment
from cirq import H
from sympy import Symbol


class CirqQaoa(QaoaBase):
    def __init__(self, n_qubits, programstring, energy_function, subcircuits,
                 num_measurements=10000):
        self.n_qubits = n_qubits
        self.n_shots = num_measurements

        self.subcircuits = subcircuits
        self.circuit = Circuit()
        self.qubits = [LineQubit(i) for i in range(self.n_qubits)]

        self.program = SimpleProgram(program_string=programstring)

        self.circuit_parameters = []
        self.set_programstring(programstring)

        self.noise_model = None

        self.measurements_present = False

        self.energy_function = energy_function

        super(CirqQaoa, self).__init__(np.mean)

    def energy_from_counts(self, counts):
        """
        :param counts: A dictionary that maps the possible state configurations to the
                        number of times they were measured
        :return:    The average energy that was measured
        """
        total = 0
        energy = 0

        # if data comes directly from res.final_state and has not been prepared yet
        if type(counts) is not dict:
            counts = dict(zip([x for x in range(2 ** self.n_qubits)], counts))
            for i in counts.keys():
                counts[i] = float((abs(counts[i]) ** 2))
        print("energy_from_counts", counts)
        for bin_conf, counts in counts.items():

            if type(bin_conf) is int:
                bin_array = [x for x in bin(bin_conf)[2:]]
                while len(bin_array) < self.n_qubits:
                    bin_array.insert(0, '0')
                bin_conf = "".join(bin_array)
            energy += self.energy_function(bin_conf) * counts
            total += counts
        energy /= total

        return energy

    def all_energies_from_counts(self, counts):
        """

        :param counts: A dictionary that maps the possible state configurations to the
                        number of times they were measured
        :return:    An array of all single energeies that were measured
        """
        # print("all_energies_from_counts cirq")
        energies = []

        for bin_conf, count in counts.items():
            if type(bin_conf) is int:
                bin_array = [x for x in bin(bin_conf)[2:]]
                while len(bin_array) < self.n_qubits:
                    bin_array.insert(0, '0')
                bin_conf = "".join(bin_array)
            energies += [self.energy_function(bin_conf)] * count

        return np.array(energies)

    def execute_circuit(self, parameters=None, do_measurement=True, return_result=False):
        """

        :param parameters: The Qaoa-parameters the circuit shall be executed with.
        :param do_measurement: Specifies whether a measurement shall be performed after executing the circuit.
        If False, state-collapse due to measurement is avoided (i. e. no sampling is necessary).
        :param return_result: Specifies whether the whole output of the Cirq simulator shall be returned or
        only the objective value (which is the energy of the output state)
        :return: Either the output of the circuit simulator or the energy expectation value for the output state,
        depending on return_result.
        """
        if parameters is not None:
            params = parameters
        else:
            params = self.program.linearparameters

        # circuit = self.circuit
        sml = Simulator()

        parameter_mapping = {par: val for par, val in zip(self.circuit_parameters, params)}
        # print(parameter_mapping)
        if do_measurement:
            if not self.measurements_present:
                self.circuit.append(measure(*self.qubits, key='final_measurement'))
                self.measurements_present = True
            # qubits = circuit.all_qubits()
            # circuit.append(measure(*qubits, key='final_measurement'))
            result = sml.run(self.circuit, parameter_mapping, repetitions=self.n_shots)
            # print(result.histogram(key='final_measurement'))

            counts = result.histogram(key='final_measurement')

        else:

            if self.measurements_present:
                # print("removing measurement gates")
                self.circuit = without_measurements(self.circuit)
                self.measurements_present = False

            result = sml.simulate(self.circuit, parameter_mapping)
            # print(result.final_state)
            counts = dict(zip([x for x in range(2 ** self.n_qubits)], result.final_state))
            for i in counts.keys():
                counts[i] = float((abs(counts[i]) ** 2))
            print(counts)

        if return_result:
            return result

        return self.all_energies_from_counts(counts)

    def execute_and_observe(self, parameters, measurement_functions):
        pass

    def set_programstring(self, programstring):
        """

        :param programstring: A sequence of letters determining the subcircuits being applied.
        :return: None.
        """
        # print("set programstring cirq")
        super().set_programstring(programstring)

        self.circuit_parameters = []
        for i in range(len(self.program)):
            self.circuit_parameters += [Symbol('par' + str(i))]

        self.circuit = Circuit()
        for q in self.qubits:
            self.circuit.append(H(q))  # construct initial state |+>

        for i, letter in enumerate(programstring):
            self.circuit.append(resolve_parameters(
                self.subcircuits[letter],
                {get_parameter_name(letter): self.circuit_parameters[i]}))


def get_parameter_name(program_letter):
    # there might be a better way to do that mapping
    if program_letter == 'X':
        return 'pX'
    elif program_letter == 'Z':
        return 'pZ'
    elif program_letter == 'C':
        return 'pname'

def without_measurements(circuit: Circuit) -> Circuit:
    """
    Helper function to remove measurement gates
    ensures there are no measurement gates since rotation gates
    must still be added
    Args:
        circuit: cirq.Circuit type which contains measurements
    Returns:
        cirq.Circuit, circuit without measurements
    """
    cir2 = Circuit()
    for moment in circuit:
        new_moment = []
        for op in moment.operations:
            if not is_measurement(op):
                new_moment.append(op)

        cir2.append(Moment(new_moment))

    return cir2


def single_lhz_energy_cirq(jij, constraints, costr, bin_conf):
    """

    :param jij: Local field strengths of the LHZ-Hamiltonian (Interaction-strengths of Spinglass Hamiltonian)
    (array)
    :param constraints: Array of tuples containing the qubit indices of physical qubit in a constraint.
    :param costr: strength of constraints (float)
    :param bin_conf: The binary configuration of the state (string containing 0 and 1)
    :return: returns the energy of the state given by bin_conf according to the setting given by the
    other parameters.
    """

    # conf = [1 if b == '0' else -1 for b in bin_conf]
    conf = [-1 if b == '0' or b == 0 else 1 for b in bin_conf]
    e = np.sum(np.array(jij) * np.array(conf))

    for c in constraints:
        e += -costr * (np.prod(np.array(conf)[c]) - 1) / 2
        # e += -costr * (np.prod(np.array(conf)[c]))

    return e


if __name__ == '__main__':
    from qaoa.operator_dicts.cirq_gates import standard_lhz_circuits
    from lhz.core import standard_constraints
    from functools import partial

    n_l = 4
    n_p = n_l * (n_l - 1) // 2

    cstr = 3.0
    # cs = [[1,2,3], [0,5,4,2]]
    cs = standard_constraints(n_l)
    pstr = 'ZCX'
    jij = [1, 2, 1, -2, 1, -1] * 6
    jij = jij[:n_p]

    efun = partial(single_lhz_energy_circ, jij, cs, cstr)

    subcircs = standard_lhz_circuits(n_p, jij, cs)
    qsim = CirqQaoa(len(jij), pstr, efun, subcircs)
    #
    # print(qsim.circuit)
    # print(qsim.program.linearparameters)

    res = qsim.execute_circuit(return_result=True, do_measurement=False)
    print("energy:", qsim.energy_from_counts(res.final_state))
    print("successfully executed circuit!")
    # print(qsim.circuit)
    mc_res = qsim.mc_optimization(n_maxiterations=100, return_timelines=True,
                                  do_convergence_checks=False)
    res = qsim.execute_circuit(mc_res.parameters, return_result=True, do_measurement=False)
    print("energy:", qsim.energy_from_counts(res.final_state))
    # print(qsim.circuit)
    # print(mc_res)
