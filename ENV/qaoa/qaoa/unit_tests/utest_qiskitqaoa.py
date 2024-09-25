import functools
import matplotlib.pyplot as plt
import numpy as np
import unittest
from qiskit import Aer
from qiskit.providers.aer import AerSimulator

import qaoa.operator_dicts.qiskit_gates as qiskitgates
from qaoa.QiskitQaoa import single_lhz_energy, QiskitQaoa
from qaoa.result_plotter import plot_results


class TestQiskitQaoa(unittest.TestCase):
    def test_everything(self):
        # n_l = 4
        # n_p = n_l * (n_l - 1) // 2
        # cs = standard_constraints(n_l)
        cstr = [3.0, 2.]

        n_p = 5
        cs = [[0, 1, 2, 3], [1, 2, 3, 4]]

        jij = [1, 2, 1, -2, 1, -1] * 6
        jij = jij[:n_p]
        # jij = [1, 2, 1.3, -2]

        pstr = 'ZXCX' * 3

        # qiskit class
        efun = functools.partial(single_lhz_energy, jij, cs, cstr)
        subcircs = qiskitgates.standard_lhz_circuits(n_p, jij, cs, cstr)
        subcircs = qiskitgates.r4p_lhz_circuits(n_p, jij, cs, True)
        q = QiskitQaoa(len(jij), pstr, efun, subcircs)
        # print(q.program)
        mc_res = q.mc_optimization(n_maxiterations=50, return_timelines=True, do_convergence_checks=False)

        used_params = q.program.linearparameters

        res_qasm = q.execute_circuit(used_params, measure=True, return_result=True)
        # q.backend = Aer.get_backend('statevector')
        q.backend = AerSimulator(method='statevector')
        res_state_vec = q.execute_circuit(used_params, measure=False, return_result=True)
        # q.backend = Aer.get_backend('qasm_simulator')
        wf_qiskit = res_state_vec.get_statevector()

        print('energy qiskit statevector: ', q.energy_from_counts(res_state_vec.get_counts()))
        print('energy qiskit qasm simulator: ', q.energy_from_counts(res_qasm.get_counts()))

        # %% qutip class
        do_qutip_comparison = False
        if do_qutip_comparison:
            from qaoa.QutipQaoa import QutipQaoa
            from lhz.qutip_hdicts import hdict_physical_not_full
            from qutip import expect

            hd = hdict_physical_not_full(jij, constraints=cs, cstrength=cstr)
            qq = QutipQaoa(hd, pstr, get_groundstate_energy=True, do_prediagonalization=False)
            resq = qq.mc_optimization(50, return_timelines=True, do_convergence_checks=False)

            plot_results([mc_res, resq], legend=['qiskit', 'qutip'])
            # translation of parameters from qiskit to qutip because of different conventions in rotations
            # 1/2 because RZ(a) = exp(-i Z a/2) for X and Z
            # the qiskit constraint circuit does not know constraint strength, the hamiltonian does!
            # C = (cstrength * (1 - ZZZZ) / 2) -> gives factor -1/cstr
            used_params_qutip = [used_params[i] / 2 if l in ['X', 'Z']
                                 else -used_params[i] / cstr
                                 for i, l in enumerate(pstr)]
            wf_qutip = qq.run_qaoa(used_params_qutip)

            print('energy qutip: ', expect(hd['P'], wf_qutip))

        do_cirq_comparison = False
        if do_cirq_comparison:
            from qaoa.operator_dicts.qiskit_gates import standard_lhz_circuits
            from lhz.core import standard_constraints
            from qaoa.QutipQaoa import QutipQaoa
            from lhz.qutip_hdicts import hdict_physical_not_full
            from qutip import expect
            from functools import partial
            from qaoa.CirqQaoa import CirqQaoa, single_lhz_energy_cirq
            import qaoa.operator_dicts.cirq_gates as cirq_gates
            from time import time

            n_l = 4
            n_p = n_l * (n_l - 1) // 2
            cs = standard_constraints(n_l)
            cstr = 3.0

            # n_p = 4
            # cs = [[0, 1, 2, 3]]

            jij = [1, 2, 1, -2, 1, -1] * 6
            jij = jij[:n_p]
            # jij = [1, 2, 1.3, -2]

            pstr = 'ZXCX' * 1
            # pstr = 'ZXCX'

            # qiskit class
            efun = partial(single_lhz_energy, jij, cs, cstr)
            subcircs = standard_lhz_circuits(n_p, jij, cs)
            q = QiskitQaoa(len(jij), pstr, efun, subcircs)

            print(q.program.linearparameters)

            n_mcsteps = 100

            t1 = time()
            mc_res = q.mc_optimization(n_maxiterations=n_mcsteps, return_timelines=True, do_convergence_checks=False,
                                       measurement_functions=[np.std])
            t2 = time()
            t_qiskit = t2 - t1
            # cirq class
            efun = partial(single_lhz_energy_cirq, jij, cs, cstr)
            subcircs = cirq_gates.standard_lhz_circuits(n_p, jij, cs)
            q_cirq = CirqQaoa(len(jij), pstr, efun, subcircs)

            t1 = time()
            mc_res_cirq = q_cirq.mc_optimization(n_maxiterations=n_mcsteps, return_timelines=True, do_convergence_checks=False,
                                                 measurement_functions=[np.std])
            t2 = time()
            t_cirq = t2 - t1
            # qutip class
            hd = hdict_physical_not_full(jij, constraints=cs, cstrength=cstr)
            qq = QutipQaoa(hd, pstr, get_groundstate_energy=True, do_prediagonalization=True)

            t1 = time()
            resq = qq.mc_optimization(n_mcsteps, return_timelines=True, do_convergence_checks=False,
                                      measurement_functions=[np.std])
            t2 = time()
            t_qutip = t2 - t1

            # plot_results([mc_res, mc_res_cirq, resq], legend=['qiskit', 'cirq', 'qutip'])

            # %%
            #               Z    X    C    X
            # used_params = [0.3, 0.1, 0.5, 0.0]
            used_params = q.program.linearparameters
            print(used_params)

            # translation of parameters from qiskit to qutip because of different conventions in rotations
            # 1/2 because RZ(a) = exp(-i Z a/2) for X and Z
            # the qiskit constraint circuit does not know constraint strength, the hamiltonian does!
            # C = (cstrength * (1 - ZZZZ) / 2) -> gives factor -1/cstr
            used_params_qutip = [used_params[i] / 2 if l in ['X', 'Z']
                                 else -used_params[i] / cstr
                                 for i, l in enumerate(pstr)]

            res_qasm = q.execute_circuit(used_params, measure=True, return_result=True)
            q.backend = Aer.get_backend('statevector_simulator')
            res_state_vec = q.execute_circuit(used_params, measure=False, return_result=True)
            q.backend = Aer.get_backend('qasm_simulator')
            wf_qiskit = res_state_vec.get_statevector()
            wf_qutip = qq.run_qaoa(used_params_qutip)

            res_state_vec_cirq = q_cirq.execute_circuit(used_params, do_measurement=False, return_result=True)
            # for i, (binconf, prob) in enumerate(res_state_vec.get_counts().items()):
            #     bluub = create_qutip_wf([1 if b == '0' else -1 for b in binconf[::-1]])
            #     ovlap = bluub.overlap(wf_qutip)
            #
            #     if i == 0:
            #         global_phase = - np.angle(ovlap) + np.angle(wf_qiskit[i])
            #
            #     print('%s: %.2f * exp(%.2f * i), \t %s: %.2f * exp(%.2f * i), \t <HZ>: %.2f \t' %
            #           (binconf, np.abs(wf_qiskit[i]), np.angle(wf_qiskit[i])/1 % (2 * np.pi),
            #            'qutip', np.abs(ovlap), (np.angle(ovlap)/1 + global_phase) % (2 * np.pi),
            #            expect(hd['Z'], bluub)))

            print('energy qutip: ', expect(hd['P'], wf_qutip))
            print('energy qiskit statevector: ', q.energy_from_counts(res_state_vec.get_counts()))

            print('energy cirq statevector', q_cirq.energy_from_counts(res_state_vec_cirq.final_state))

            # print(sum(res_state_vec.get_counts().values()))
            # print(res_state_vec_cirq.final_state)
            # print(res_qasm.get_counts())
            print('energy qiskit qasm simulator: ', q.energy_from_counts(res_qasm.get_counts()))

            print('parameters after mc-optimization (qiskit): ', mc_res.parameters)
            print('parameters after mc-optimization (cirq): ', mc_res_cirq.parameters)
            print('parameters after mc-optimization (qutip): ', resq.parameters)
            print('energy mc-optimized state (qiskit): ', mc_res.final_objective_value)
            print('energy mc-optimized state (cirq): ', mc_res_cirq.final_objective_value)
            print('energy mc-optimized state (qutip): ', resq.final_objective_value)

            print('runtime qiskit:', t_qiskit)
            print('runtime cirq:', t_cirq)
            print('runtime qutip:', t_qutip)
            # print(q.circuit.decompose().decompose().decompose())
            # print(q_cirq.circuit)

            # measure simulation time vs system size
            n_arr = [3, 4, 5, 6, 7, 8, 9]
            jij = [1, 2, 1, -2, 1, -1] * 6
            cstr = 3.0
            pstr = 'ZXCX' * 1

            n_mcsteps = 100

            t_qutip = []
            t_qiskit = []
            t_cirq = []

            for n_l in n_arr:
                n_p = n_l * (n_l - 1) // 2
                cs = standard_constraints(n_l)
                jij = [1, 2, 1, -2, 1, -1] * 6
                jij = jij[:n_p]

                # qiskit class
                # efun = partial(single_lhz_energy_circ, jij, cs, cstr)
                # subcircs = standard_lhz_circuits(n_p, jij, cs)
                print('performing', n_l)
                # q = QiskitQaoa(len(jij), pstr, efun, subcircs)
                #
                # print(q.program.linearparameters)
                # print("qiskit")
                # t1 = time()
                # mc_res = q.mc_optimization(n_maxiterations=n_mcsteps, return_timelines=True, do_convergence_checks=False,
                #                            measurement_functions=[np.std])
                # t2 = time()
                # t_qiskit.append(t2 - t1)
                # cirq class
                # efun = partial(single_lhz_energy_cirq, jij, cs, cstr)
                # subcircs = cirq_gates.standard_lhz_circuits(n_p, jij, cs)
                # q_cirq = CirqQaoa(len(jij), pstr, efun, subcircs)

                # print("cirq")
                # t1 = time()
                # mc_res_cirq = q_cirq.mc_optimization(n_maxiterations=n_mcsteps, return_timelines=True,
                #                                      do_convergence_checks=False,
                #                                      measurement_functions=[np.std])
                # t2 = time()
                # t_cirq.append(t2 - t1)
                # # qutip class
                hd = hdict_physical_not_full(jij, constraints=cs, cstrength=cstr)

                print("qutip")
                t1 = time()
                qq = QutipQaoa(hd, pstr, get_groundstate_energy=False, do_prediagonalization=False)
                resq = qq.mc_optimization(n_mcsteps, return_timelines=True, do_convergence_checks=False,
                                          measurement_functions=[np.std])
                t2 = time()
                t_qutip.append(t2 - t1)

            plt.plot(n_arr[0:len(t_qutip)], t_qutip, label='qutip')
            # plt.plot(n_arr, t_qiskit, label='qiskit')
            # plt.plot(n_arr, t_cirq, label='cirq')
            plt.legend()
            plt.xlabel('n_l')
            plt.ylabel('runtime')
