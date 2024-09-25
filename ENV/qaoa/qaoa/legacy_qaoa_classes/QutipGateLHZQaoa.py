from qutip import basis, tensor, expect
from qutip.qip import rx, rz
import numpy as np

from lhz.core import n_logical, standard_constraints, qubit_dict
from lhz.qutip_hdicts import Hdict_physical
from qaoa.legacy_qaoa_classes.QutipGates import n_body_gate, cnot_sequence, eye
from qaoa.QaoaBase import QaoaBase
from qaoa.programs import SimpleProgram


class QutipGateLHZQaoa(QaoaBase):
    def __init__(self, program_string, jij, cstrength=3.0, optimization_target='energy', psi0=None):
        self.program = SimpleProgram(program_string)
        self.jij = jij
        self.n = len(jij)
        self.nlogical = n_logical(self.n)
        assert self.nlogical < 6, 'more than 8 GB RAM needed, make timeevolution better first'

        if psi0 is None:
            self.psi0 = tensor([(basis(2, 0) + basis(2, 1)).unit()]*self.n)
        else:
            assert psi0.shape == (2**self.n, 1), 'psi0 has wrong dimension'
            self.psi0 = psi0.copy()

        self.cstr = cstrength
        self.constraints = standard_constraints(self.nlogical)

        self.forward_constraint_cnot_sequence = []
        self.backward_constraint_cnot_sequence = []

        for c in self.constraints:
            self.forward_constraint_cnot_sequence += [cnot_sequence(c, self.n)]
            self.backward_constraint_cnot_sequence += [cnot_sequence(c, self.n, backwards=True)]

        qd = qubit_dict(self.nlogical)
        self.logical_x_inds = []
        for i in range(self.nlogical):
            thv = []
            for (k, v), ind in qd.items():
                if k == i or v == i:
                    thv += [ind]
            self.logical_x_inds += [thv]

        self.forward_x_cnot_sequence = []
        self.backward_x_cnot_sequence = []

        for x_chain in self.logical_x_inds:
            self.forward_x_cnot_sequence += [cnot_sequence(x_chain, self.n, switch_control_target=True)]
            self.backward_x_cnot_sequence += [cnot_sequence(x_chain, self.n, backwards=True, switch_control_target=True)]

        self.Us = {'X': self._global_x, 'Z': self._global_z, 'C': self._full_constraints, 'H': self._half_constraints, 'V': self._logical_x, 'L': self._parallel_logical_x}
        # self.energies = np.array(all_lhz_energies(jij, self.constraints, cstr=self.cstr)[::-1]).reshape((1, 2**self.n))
        self.h_dict = Hdict_physical(self.jij, self.cstr, self.constraints)
        self.gs_energy = min(self.h_dict['P'].diag())

        if optimization_target == 'fidelity':
            def temp_objective_function(state):
                return -self.groundstate_fidelity(state)

            super().__init__(temp_objective_function)
        else:
            super().__init__(self.energy_expecation_value)

    def execute_circuit(self, parameters=None):
        if parameters is not None:
            assert len(parameters) == len(self.program)
            # temporarily save parameters if run gets called with parameters
            tempstorage = self.program.linearparameters.copy()
            self.program.linearparameters = parameters.copy()

        state = self.psi0.copy()
        for pType, param in self.program.zipped_program():
            state = self.Us[pType](param) * state

        # restore parameters if run got called with parameters
        if parameters is not None:
            self.program.linearparameters = tempstorage.copy()

        # print(sum((abs(state)**2).flatten()))
        return state

    def execute_and_observe(self, measurement_functions, parameters=None, return_string=False):
        if parameters is not None:
            assert len(parameters) == len(self.program)
            # temporarily save parameters if run gets called with parameters
            tempstorage = self.program.linearparameters.copy()
            self.program.linearparameters = parameters.copy()

        state = self.psi0.copy()

        if not isinstance(measurement_functions, list):
            measurement_functions = [measurement_functions]

        measurements = []
        for _ in measurement_functions:
            measurements += [np.zeros(len(self.program)+1)]

        for im, mf in enumerate(measurement_functions):
            measurements[im][0] = mf(state)

        for ip, (pType, param) in enumerate(self.program.zipped_program()):
            state = self.Us[pType](param) * state

            for im, mf in enumerate(measurement_functions):
                measurements[im][ip + 1] = mf(state)

        # restore parameters if run got called with parameters
        if parameters is not None:
            self.program.linearparameters = tempstorage.copy()

        if return_string:
            strr = 'program:         ' + '       '.join('0' + self.program.program_string)
            for mline in measurements:
                strr += '\nmeasurement:  ' + ', '.join(['%+.3f' % p for p in mline])

            return state, measurements, strr

        else:
            return state, measurements

    def energy_expecation_value(self, state):
        return expect(self.h_dict['P'], state)
        # return (self.energies @ abs(state)**2)[0][0]

    def expectation_value_function(self, operator_key):
        return lambda st: expect(self.h_dict[operator_key], st)

    def groundstate_fidelity(self, state):
        raise Warning('not implemented')
        pass

    # def _global_x_old(self, par):
    #     return tensor([rx(2*par)] * self.n)
        # return n_body_gate(rx(2*par), self.n)

    def _global_x(self, par):
        globx = 1
        for i in range(self.n):
            globx = rx(2*par, N=self.n, target=i) * globx
        return globx

    #
    # def _global_z2(self, par):
    #     gate = 1  # n_body_gate(eye, self.n)
    #     for iq, j in enumerate(self.jij):
    #         gate = rz(j*par*2, self.n, iq) * gate
    #         # gate = single_gate_in_n(rz(j*par), iq, self.n)*gate
    #     return gate
    # def _global_z_old(self, par):
    #     return tensor([rz(2*j*par) for j in self.jij])

    def _global_z(self, par):
        globz = 1
        for i, j in enumerate(self.jij):
            globz = rz(2*j*par, N=self.n, target=i) * globz
        return globz

    # in full constraints and halfconstraints the rotation should be *2 the angle but as in hdict_physical we have a 1/2 this falls away
    def _full_constraints(self, par):
        gate_seq = 1  # n_body_gate(eye, self.n)

        for ic, c in enumerate(self.constraints):
            # gate_seq = self.backward_constraint_cnot_sequence[ic] @ single_gate_in_n(rz(-self.cstr*par), c[-1], self.n) @ self.forward_constraint_cnot_sequence[ic] @ gate_seq
            gate_seq = self.backward_constraint_cnot_sequence[ic] * rz(-self.cstr*par, self.n, c[-1]) * self.forward_constraint_cnot_sequence[ic] * gate_seq

        return gate_seq

    def _half_constraints(self, par):
        gate_seq = n_body_gate(eye, self.n)

        for ic, c in enumerate(self.constraints):
            # gate_seq = single_gate_in_n(rz(-self.cstr*par), c[-1], self.n) @ self.forward_constraint_cnot_sequence[ic] @ gate_seq
            gate_seq = rz(-self.cstr*par, self.n, c[-1]) * self.forward_constraint_cnot_sequence[ic] * gate_seq

        return gate_seq

    def _logical_x(self, par):
        gate_seq = 1

        for ix, inds in enumerate(self.logical_x_inds):
            gate_seq = self.backward_x_cnot_sequence[ix] * rx(2*par, self.n, inds[-1]) * self.forward_x_cnot_sequence[ix] * gate_seq

        return gate_seq

    def _parallel_logical_x(self, par):
        gate_seq = 1

        for ix, inds in enumerate(self.logical_x_inds[:-1]):
            gate_seq = self.forward_x_cnot_sequence[ix] * gate_seq

        for ix, inds in enumerate(self.logical_x_inds[:-1]):
            gate_seq = rx(2*par, self.n, inds[-1]) * gate_seq

        for ix, inds in enumerate(self.logical_x_inds[:-1]):
            gate_seq = self.backward_x_cnot_sequence[ix]

        gate_seq = self.backward_x_cnot_sequence[-1] * rx(2*par, self.n, self.logical_x_inds[-1][-1]) * self.forward_x_cnot_sequence[-1] * gate_seq

        return gate_seq


if __name__ == '__main__':
    from qaoa.result_plotter import plot_results

    Jij = [1, 1, -1, 1, 2, 1]*1
    Jij = 2*np.random.random((6,))-1

    tqs = [(QutipGateLHZQaoa('ZXCX'*2, Jij), 'full gate')] + \
          [(QutipGateLHZQaoa('ZXHX'*2, Jij), 'half gate, less gates')] + \
          [(QutipGateLHZQaoa('ZXHX'*4, Jij), 'half gate, same CNOT count')]

    mc_opti_opts = {'n_maxiterations': 50, 'return_timelines': True, 'do_convergence_checks': True, 'par_change_range': 0.5}
    ress = []
    tits = []
    for tq, tit in tqs:
        ress += [tq.mc_optimization(**mc_opti_opts, measurement_functions=[tq.energy_expecation_value])]
        tits += [tit]
        # plot_result(ress[-1], tit)

    plot_results(ress, title='comparison', legend=tits)
