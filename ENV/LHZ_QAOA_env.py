from ENV.qaoa.qaoa import QutipQaoa

import numpy as np
import gym
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from gym import spaces
from lhz.core import n_logical
from lhz.qutip_hdicts import hdict_physical_not_full


class QaoaEnv(gym.Env):
    def __init__(self, problem_dict, actions=None, reps=5, n_maxiterations_monte_carlo=100, reward_threshold=0,
                 n_max_gates=10,
                 env_setup="fixed_length_reward", qaoa_optimizer="SLSQP", optimization_target="energy",
                 final_decoding="physical", num_spanningtrees=None):

        self.constraints = problem_dict["constraints"]
        self.constraint_strength = problem_dict["constraint_strength"]
        self.local_fields = problem_dict["local_fields"]
        self.h_dict = hdict_physical_not_full(jij=self.local_fields, constraints=self.constraints,
                                              cstrength=self.constraint_strength)
        self.num_qubits = len(self.h_dict['X'].dims[0])
        self.num_qubits_logical = n_logical(self.num_qubits)
        self.reps = reps
        self.n_maxiterations_monte_carlo = n_maxiterations_monte_carlo
        self.n_max_gates = n_max_gates
        self.threshold = reward_threshold
        if actions is not None:
            self.actions = actions
        else:
            self.actions = ["X", "C", "Z"]
        self.n_actions = len(self.actions)
        self.action_space = Discrete(self.n_actions)
        self.observation_space = Box(low=0, high=self.n_actions, shape=(self.n_max_gates,), dtype=np.int32)
        self.env_setup = env_setup
        self.qaoa_optimizer = qaoa_optimizer
        self.optimization_target = optimization_target

        if self.optimization_target == "logical_energy" and num_spanningtrees is None:
            self.num_spanningtrees = self.num_qubits_logical
            assert "the number of spanningtrees of the decoding is set to the number of logial qubits"
        elif self.optimization_target == "logical_energy" and num_spanningtrees is not None:
            self.num_spanningtrees = num_spanningtrees
        else:
            self.num_spanningtrees = None

        self._action_to_gate = {i: self.actions[i - 1] for i in range(1, self.n_actions + 1)}

        self.program_str = ""
        self.observation = np.zeros(self.n_max_gates)

        self.step_counter = 0
        # Initialize qutip qaoa
        self.problem_instance = QutipQaoa(h_dict=self.h_dict, jij=self.local_fields,
                                          program_string=''.join(self.actions),
                                          do_prediagonalization=True,
                                          psi0=None, optimization_target=self.optimization_target,
                                          final_decoding=final_decoding,
                                          num_spanningtrees=self.num_spanningtrees
                                          )

        self.program_str_data = {}

    def reset(self):
        self.step_counter = 0
        self.program_str = ""
        self.observation = np.zeros(self.n_max_gates)
        return self.observation

    def step(self, action):
        """
        One gate is appended to the gate sequence.
        Reward options (to be set at initialization):
        "fixed_length": The agent is rewarded after each step with the negative energy.
                        An episode is finished (done=True) if a maximal sequence length is reached.
        "fixed_length_reward":    The agent is rewarded with the negative energy and the episode is finished
                    if a maximal sequence length is reached.
        "energy_threshold_reward":   The agent is rewarded with reward=1 if the energy is below a certain threshold.
                            An episode is finished (done=True) if a maximal sequence length is reached.
        """
        action_str = self._action_to_gate[action + 1]
        self.program_str += action_str
        self.observation[self.step_counter] = action + 1  # actions are 1,2 and 3

        min_obj_value = None
        qaoa_time = None
        done = False
        reward = 0

        if self.env_setup == "fixed_length_reward":
            if len(self.program_str) >= self.n_max_gates:
                min_obj_value, qaoa_time = self.calculate_min_energy_qaoa()
                reward = -min_obj_value / self.num_qubits  # negative energy density
                done = True
        # elif self.env_setup == "energy_threshold_reward":
        #     if min_obj_value < self.threshold:
        #         reward = 1
        #         done = True
        #     if len(self.program_str) >= self.n_max_gates:
        #         done = True

        self.step_counter += 1

        info = {"min_energy": min_obj_value, "qaoa_time": qaoa_time, "gate_sequence": self.program_str}

        return self.observation, reward, done, info

    def render(self):
        pass

    def calculate_min_energy_qaoa(self):
        self.problem_instance.set_programstring(self.program_str)

        scipy_optimizers = [
            "BFGS",  # Broyden-Fletcher-Goldfarb-Shanno
            "L-BFGS-B",  # Limited-memory BFGS with box constraints
            "COBYLA",  # Constrained Optimization BY Linear Approximations
            "SLSQP",  # Sequential Least Squares Programming
        ]
        if self.program_str not in self.program_str_data.keys():
            if self.qaoa_optimizer in scipy_optimizers:
                result_container = self.problem_instance.repeated_scipy_optimization(
                    optimization_method=self.qaoa_optimizer,
                    param_ranges=None, reps=self.reps)
            elif self.qaoa_optimizer == "monte_carlo":
                result_container = self.problem_instance.multiple_random_init_mc_opt(reps=self.reps,
                                                                                     n_maxiterations=self.n_maxiterations_monte_carlo)
            else:
                raise ValueError("Optimizer not available")

            best_result = result_container.get_lowest_objective_value()
            min_obj_value = best_result.final_objective_value
            parameters = best_result.parameters
            time_taken_total = result_container.time_taken_total

            self.program_str_data[self.program_str] = {"energy": min_obj_value, "time": time_taken_total,
                                                       "parameters": parameters}

        else:
            min_obj_value = self.program_str_data[self.program_str]["energy"]
            time_taken_total = self.program_str_data[self.program_str]["time"]

        return min_obj_value, time_taken_total

    def valid_action_mask(self):
        action_mask = np.ones(self.n_actions)

        # first action should not be "X"
        if len(self.program_str) == 0:
            action_mask[self.actions.index("X")] = 0

        # last action should not be "C" or "Z"
        # if len(self.program_str) == self.n_max_gates-1:
        #     action_mask[self.actions.index("C")] = 0
        #     action_mask[self.actions.index("Z")] = 0
        # crashes because of conflicts with other constraints

        # restrict the repetition of actions
        if len(self.program_str) != 0:
            last_gate = self.program_str[-1]
            action_mask[self.actions.index(last_gate)] = 0

        # maskes gates because of commutation
        gate_mapping = {
            "ZC": "Z",
            "CZ": "C",
        }

        if len(self.program_str) >= 2:
            last_two_gates = self.program_str[-2:]
            if last_two_gates in gate_mapping:
                action_mask[self.actions.index(gate_mapping[last_two_gates])] = 0

        return action_mask
