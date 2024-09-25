import numpy as np
from lhz.core import n_logical, qubit_dict
from lhz.spinglass_utility import create_qutip_wf, get_lowest_states, \
    translate_configuration_logical_to_physical
from qaoa.QaoaBase import QaoaBase
from qaoa.programs import CombinedLetterProgram
from qaoa.additional_functions import decompose_state, logical_energy, get_spanning_tree_configs, get_configs
from qutip import Qobj
from qutip import expect, sesolve


class QutipQaoa(QaoaBase):
    def __init__(self, h_dict, program_string, jij=None, get_groundstate_energy=True,
                 optimization_target='physical_energy', final_decoding='physical', combined_letters=None,
                 psi0=None, do_prediagonalization=False, num_spanningtrees=None):
        """

        :param h_dict: dictionary of qutip-hamiltonians which are
                       applied onto psi0 given by program_string
        :param program_string: gate sequence, keys must be in h_dict
        :param jij: if given (must fit to problem in h_dict['P']) this is used
                    instead of solving eigensystem
        :param get_groundstate_energy:
        :param optimization_target:
        :param combined_letters:
        :param psi0: starting vector for QAOA, usually groundstate of H_X
        :param do_prediagonalization:
        """
        # ToDo: add optimization target logical_energy

        assert program_string.strip(''.join(h_dict.keys())) == '', 'program contains unknown stuff'
        assert 'P' in ''.join(h_dict.keys()), 'Problem must be defined (Hdict[\'P\'])'

        self.jij = jij
        self.Hdict = h_dict

        self.N = int(np.log2(list(h_dict.values())[0].shape[0]))
        self.Np = len(self.jij)

        self.finiterange = None

        if num_spanningtrees is not None:
            self.num_spanningtrees = num_spanningtrees
        else:
            self.num_spanningtrees = n_logical(len(jij))

        if jij is not None:
            # compare jij matrix to dimension of given hamiltonians to decide if we are in the lhz-encoding
            if n_logical(len(jij)) == self.N:
                self.logical = True
            elif len(jij) == self.N:
                self.logical = False
            else:
                assert False, 'format of jij is wrong'

        self.program = CombinedLetterProgram(program_string, combined_letters)

        if psi0 is None:
            self.psi0 = create_qutip_wf(['+'] * self.N)
        else:
            self.psi0 = psi0

        if do_prediagonalization:
            self.run_qaoa = self.run_qaoa_prediag

            unique_keys_in_program = list(set(program_string))

            self.eigenenergies = {}
            self.Us = {}
            self.Udags = {}
            self.UUds = {}

            self.diagkeys = ['Z', 'C', 'P']
            # self.diagkeys = []

            temp = []

            for key in unique_keys_in_program:
                if key not in self.diagkeys:
                    temp += [key]
                    self.eigenenergies[key], self.Us[key] = np.linalg.eigh(self.Hdict[key].data.toarray())
                    self.Udags[key] = self.Us[key].conj().T

            from itertools import permutations
            pairs = permutations(temp, 2)


            for pair in pairs:
                self.UUds[pair] = self.Udags[pair[0]] @ self.Us[pair[1]]

        else:
            self.run_qaoa = self.run_qaoa_qutip

        self.fidlist = []
        self.eigsysP = None


        # if either diagonalization was done or jij matrix was given set the groundstate energy
        if self.fidlist:
            es = list()
            for s in self.fidlist:
                es.append(expect(h_dict['P'], s))

            self.gs_energy = es[0]

        if type(optimization_target) == str:
            if optimization_target == 'fidelity':
                def temp_objective_function(state):
                    return -self.fidelity_function(state)

                super().__init__(temp_objective_function)
            elif optimization_target.startswith('cvar'):
                if self.eigsysP is None:
                    self.eigsysP = self.Hdict['P'].eigenstates()

                try:
                    fraction = int(optimization_target.split('_')[1])
                except ValueError as e:
                    print(e, 'fraction set to default (50)')
                    fraction = 50

                cvar_e_lim = fraction / 100 * (np.max(self.eigsysP[0]) - np.min(self.eigsysP[0])) / 2
                self.cvar_proj = 0

                for i, st in enumerate(self.eigsysP[1]):
                    if self.eigsysP[0][i] < cvar_e_lim:
                        self.cvar_proj += st * st.dag()
                super().__init__(self.cvar_expectation_value)

            elif optimization_target == 'physical_energy' and final_decoding == 'physical':
                super().__init__(objective_function=self.energy_function,
                                 final_energy_function=lambda state: self.final_energy(state, option=final_decoding))

            elif optimization_target == 'physical_energy' and final_decoding in ["logical_min", "logical_mean"]:
                super().__init__(objective_function=self.energy_function,
                                 final_energy_function=lambda state: self.final_energy(state, option=final_decoding))
                self.decoded_E = self.calculate_decoded_E()

            elif optimization_target == 'logical_energy' and final_decoding in ["logical_min", "logical_mean"]:
                super().__init__(objective_function=self.energy_function_logical,
                                 final_energy_function=lambda state: self.final_energy(state, option=final_decoding))
                self.decoded_E = self.calculate_decoded_E()
            else:
                super().__init__(objective_function=self.energy_function)
                assert "qaoa with physical energy without decoding is used (not very representative)"


        elif callable(optimization_target):
            super().__init__(optimization_target)

    def calculate_decoded_E(self):
        config_list = get_configs(self.Np)
        decoded_E = np.zeros((2 ** self.Np, self.num_spanningtrees))
        for i, conf in enumerate(config_list):
            logical_config_list = get_spanning_tree_configs(conf, self.num_spanningtrees)
            decoded_E[i, :] = np.array([logical_energy(logical_conf, self.jij) for logical_conf in logical_config_list])

        return decoded_E

    def fidelity_function(self, state):
        return sum([abs(state.overlap(gs)) ** 2 for gs in self.fidlist])

    def energy_function(self, state):
        # for debugging:
        # Np = int(np.log2(state.shape[0]))
        # decomposed_state = decompose_state(state, Np)
        # print(decomposed_state)
        return expect(self.Hdict['P'], state)

    def energy_function_logical(self, state):

        probabilities = np.zeros(2 ** self.Np)
        for i, amplitude in enumerate(state.data.toarray()):
            probabilities[i] = np.abs(amplitude) ** 2

        mean_decoded_E = np.mean(self.decoded_E, axis=1)

        mean_energy = np.dot(probabilities, mean_decoded_E)
        # for debugging:
        # print(get_configs(self.Np)[np.argmax(probabilities)])

        return mean_energy

    def final_energy(self, state, option="logical_min"):

        probabilities = np.zeros(2 ** self.Np)

        for i, amplitude in enumerate(state.data.toarray()):
            probabilities[i] = np.abs(amplitude) ** 2

        if option == "logical_min":
            min_decoded_E = np.min(self.decoded_E, axis=1)
            energy = np.dot(probabilities, min_decoded_E)
        elif option == "logical_mean":
            mean_decoded_E = np.mean(self.decoded_E, axis=1)
            energy = np.dot(probabilities, mean_decoded_E)
        elif option == "physical":
            energy = expect(self.Hdict['P'], state)
        else:
            energy = None
            assert "no valid final energy option given"

        # storing some interesting stuff
        max_prob = np.max(probabilities)
        max_prob_idx = np.argmax(probabilities)
        best_config_physical = get_configs(self.Np)[max_prob_idx]

        logical_config_list = get_spanning_tree_configs(best_config_physical, self.num_spanningtrees)
        logical_energy_list = [logical_energy(log_conf,self.jij) for log_conf in logical_config_list]
        best_config_logical_idx = np.argmin(logical_energy_list)
        best_config_logical = logical_config_list[best_config_logical_idx]

        return energy, max_prob, best_config_physical, best_config_logical



    def expectation_value_function(self, letter):
        def exp_fun(state):
            return expect(self.Hdict[letter], state)

        return exp_fun

    def cvar_expectation_value(self, state):
        projected_state = self.cvar_proj * (state.copy())
        projected_state = projected_state.unit()

        return expect(self.Hdict['P'], projected_state)

    def run_qaoa_qutip(self, parameters=None):
        """
        does the time evolution for a given program with qutip
        """

        if parameters is not None:
            assert len(parameters) == len(self.program)
            # temporarily save parameters if run gets called with parameters
            tempstorage = self.program.linearparameters.copy()
            self.program.linearparameters = parameters.copy()

        try:
            state = self.psi0.copy()
            for pType, param in self.program.zipped_program():
                res = sesolve(self.Hdict[pType], state, [0, param], [])
                state = res.states[-1]
        except Exception as e:
            print('error during run_qaoa', e)
            return self.psi0

        # restore parameters if run got called with parameters
        if parameters is not None:
            self.program.linearparameters = tempstorage.copy()

        return state

    def run_qaoa_prediag(self, parameters=None):
        """
        does the time evoultion for a given program with the prediagonalized hamiltonians
        :param parameters:
        :return:
        """

        if parameters is not None:
            assert len(parameters) == len(self.program)
            # temporarily save parameters if run gets called with parameters
            tempstorage = self.program.linearparameters.copy()
            self.program.linearparameters = parameters.copy()

        state = self.psi0.data.toarray()

        previousbasis = 'Z'  # because the state is saved in computational basis

        for pType, param in self.program.zipped_program():
            if pType == previousbasis and pType not in self.diagkeys:  # no change: 1
                state = np.exp(-1j * param * self.eigenenergies[pType][:, np.newaxis]) * state
            elif previousbasis in self.diagkeys:
                if pType in self.diagkeys:  # diag -> diag: 1
                    state = state * np.exp(-1j * param * self.Hdict[pType].diag()[:, np.newaxis])
                else:  # diag -> nondiag: Udag[pType]
                    state = np.exp(-1j * param * self.eigenenergies[pType][:, np.newaxis]) * (self.Udags[pType] @ state)
            elif previousbasis not in self.diagkeys:
                if pType in self.diagkeys:  # nondiag -> diag: U[previousbasis]
                    state = np.exp(-1j * param * self.Hdict[pType].diag()[:, np.newaxis]) * (
                            self.Us[previousbasis] @ state)
                else:  # nondiag -> nondiag UUds[(pType, previousbasis)]
                    state = np.exp(-1j * param * self.eigenenergies[pType][:, np.newaxis]) * (
                            self.UUds[(pType, previousbasis)] @ state)
            else:
                raise (Exception(
                    'this should never happen'))  # , mismatch in prediagonalized keys and program - only change program over set-programstring and dont include new keys'))

            previousbasis = pType
            # if pType not in self.diagkeys:  # D->X  or X->X
            #     if previousbasis in self.diagkeys: # make that all D->X come from Z
            #         previousbasis = 'Z'
            #     state = np.exp(-1j * param * self.eigenenergies[pType][:, np.newaxis]) * (self.UUds[(pType, previousbasis)] @ state)
            # else: # D->D or X->D
            #     if previousbasis in self.diagkeys: # D-> D
            #         state = state * np.exp(-1j * param * self.Hdict[pType].diag()[:, np.newaxis])
            #     else: # X->D
            #         state = np.exp(-1j * param * self.Hdict[pType].diag()[:, np.newaxis]) * (self.Us[previousbasis] @ state)
            #
            # previousbasis = pType

        # transform back to computational basis if the last programtype was not diagonal
        if pType not in self.diagkeys:
            state = self.Us[pType] @ state

        # restore parameters if run got called with parameters
        if parameters is not None:
            self.program.linearparameters = tempstorage.copy()

        return Qobj(state, dims=self.psi0.dims, shape=self.psi0.shape)

    def execute_circuit(self, parameters=None):
        return self.run_qaoa(parameters)

    def get_probabilities(self, state, cutoff=0.05):
        probs = abs(state.data.toarray() ** 2).flatten()
        print_inds = np.where(probs >= cutoff)[0]
        ret_list = []
        for i, ind in enumerate(print_inds):
            ret_list += [(np.round(probs[ind], 3), np.binary_repr(ind, width=self.N))]

        return ret_list
