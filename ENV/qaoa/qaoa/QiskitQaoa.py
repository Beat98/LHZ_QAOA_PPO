import abc
from typing import Callable

from qaoa.QaoaBase import QaoaBase
from qaoa.programs import SimpleProgram
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter
from functools import lru_cache
import numpy as np

# hack for import compatibility, can we raise a warning if this is used?
from qaoa.energy_functions.qiskit_efuns.logical_energy import energy_of_logical_configuration as efun_gatemodel
from qaoa.energy_functions.qiskit_efuns.physical_energy import energy_of_physical_configuration as single_lhz_energy

# something like this?
# @deprecated
# def efun_gatemodel(*args):
#     return efun_gatemodel_depr(*args)

"""
conventions used in this library:
- in the LHZ-mapping: 0 -> spins are parallelly aligned
                      1 -> spins are antiparallelly aligned
- when translating states, 0 maps to +1, 
                           1 maps to -1
- in Qiskit, the highest-valued bit is the one on the outer right side of the bit string,
  i. e. they have to be read in reversed order (in histogram plots, the strings have to be read downwards)

"""


class QiskitQaoa(QaoaBase):
    def __init__(self, n_qubits, programstring, energy_function, subcircuits,
                 num_measurements=1000, backend_options=None, simulation_method='statevector',
                 noise_model=None, basis_gates=None, do_default_state_init=True):
        """

        :param n_qubits: number of qubits in the circuit
        :param programstring: sequence of subcircs to be used for QAOA
        :param energy_function: takes binary representation returned from qiskit to calculate energy
        :param subcircuits: parametrized subcircuits where each circuit is for a given letter from the programstring,
                            letter I is reserved for initializing circuit
        :param num_measurements: number of repetitions for qiskit ciruit execution
        :param backend_options:
        :param noise_model:
        :param basis_gates:
        :param do_default_state_init:
        """

        self.circuit = None
        self.circuit_parameters = list()
        self.program = None
        self.noise_model = None
        self.last_executed_job = None

        if noise_model is not None:
            self.noise_model = noise_model

        self.basis_gates = basis_gates

        self.n_qubits = n_qubits
        self.n_shots = num_measurements
        self.do_default_state_init = do_default_state_init

        self.subcircuits = subcircuits

        self.set_programstring(programstring)

        self.backend = AerSimulator(method=simulation_method, noise_model=self.noise_model)
        self.backend_options = {"max_parallel_threads": 0,
                                "max_parallel_shots": 0}

        if backend_options is not None:
            self.backend_options.update(backend_options)

        self.energy_function = energy_function

        super(QiskitQaoa, self).__init__(np.mean)

    @lru_cache(maxsize=None)
    def cached_energy(self, bin_conf):
        return self.energy_function(bin_conf)

    def energy_from_counts(self, counts):
        total = 0
        energy = 0
        for bin_conf, counts in counts.items():
            energy += self.cached_energy(bin_conf) * counts
            total += counts
        energy = energy / total

        return energy

    def all_energies_from_counts(self, counts):
        energies = []
        for bin_conf, counts in counts.items():
            energies += [self.cached_energy(bin_conf)] * counts

        return np.array(energies)

    def execute_circuit(self, parameters=None, measure=True, return_result=False):
        if parameters is not None:
            params = parameters
        else:
            params = self.program.linearparameters

        bound_circuit = self.circuit.bind_parameters({par: val for par, val in zip(self.circuit_parameters, params)})
        if measure:
            self.last_executed_job = execute(bound_circuit, self.backend,
                                             noise_model=self.noise_model, shots=self.n_shots,
                                             basis_gates=self.basis_gates, **self.backend_options)
        else:
            bound_circuit.remove_final_measurements(inplace=True)
            bound_circuit.save_state()
            self.last_executed_job = execute(bound_circuit, self.backend, noise_model=self.noise_model,
                                             shots=self.n_shots, basis_gates=self.basis_gates, **self.backend_options)
        result = self.last_executed_job.result()

        if return_result:
            return result
        return self.all_energies_from_counts(result.get_counts())

    def set_programstring(self, programstring):
        # redo program string in program
        for letter in programstring:
            assert letter.isupper(), 'lower case letters are reserved internally for gates with multiple parameters'
            assert letter in self.subcircuits.keys(), 'gate %s not defined in subcircuits' % letter

        expanded_programstring = list(programstring.replace('I', ''))  # remove initializing part of programstring
        for key, subcircuit in self.subcircuits.items():
            if len(subcircuit.parameters) > 1:
                while key in expanded_programstring:
                    position = expanded_programstring.index(key)
                    expanded_programstring[position] = key.lower() + '0'
                    for i in range(len(subcircuit.parameters) - 1):
                        expanded_programstring.insert(position + i + 1, key.lower() + str(i + 1))

        self.program = SimpleProgram(expanded_programstring)

        self.circuit_parameters = list()

        # redo qiskit circuit
        self.circuit = QuantumCircuit(self.n_qubits)

        if 'I' not in programstring and self.do_default_state_init:
            self.state_init_default()

        parameter_count = 0
        for i, letter in enumerate(programstring):
            temp_dict = {}
            for original_param in sorted(self.subcircuits[letter].parameters, key=lambda x: x.name):
                self.circuit_parameters.append(Parameter('p_%d' % parameter_count))
                temp_dict[original_param] = self.circuit_parameters[-1]
                parameter_count += 1
            self.circuit.append(self.subcircuits[letter].to_instruction(temp_dict), range(self.n_qubits))

        self.circuit.measure_all(inplace=True)

    def set_transpiled_circuit(self, transpiled_circuit):
        """
        Set circuit.
        :param transpiled_circuit: Transpiled circuit which has to have the SAME parameters as the original circuit
        created by set_programstring!
        :return: None
        """
        # TODO check if parameters of self.circuit are the same as those of transpiled_circuit
        # "jo des funktioniert glab i" - Michael F., 09.06.2021
        self.circuit = transpiled_circuit.copy()

    def state_init_default(self):
        self.circuit.h(range(self.n_qubits))



