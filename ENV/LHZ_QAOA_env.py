from ENV.qaoa.qaoa import QutipQaoa

import numpy as np
import gym
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from lhz.core import n_logical
from lhz.qutip_hdicts import hdict_physical_not_full


class QaoaEnv(gym.Env):
    DEFAULT_ACTIONS = ["X", "C", "Z"]
    SCIPY_OPTIMIZERS = ["BFGS", "L-BFGS-B", "COBYLA", "SLSQP"]

    def __init__(self, problem_dict, actions=None, reps=5,
                 n_max_gates=10, qaoa_optimizer="SLSQP", optimization_target="logical_energy",
                 final_decoding="logical_mean", num_spanningtrees=None):

        """
            Initializes the gym conform Qaoa environment.

            Args:
                problem_dict (dict):
                    Description of the problem instance to be optimized, containing constraints,
                    constraint strength, and local fields.
                actions (list of str):
                    The list of actions (gates) available in the environment. If None, defaults to DEFAULT_ACTIONS.
                reps (int, optional):
                    Number of random initializations of the QAOA angles. Default is 5.
                n_max_gates (int):
                    Maximum number of gates allowed in the QAOA sequence. Default is 10.
                qaoa_optimizer (str):
                    The classical optimizer to be used for QAOA. Default is "SLSQP".
                optimization_target (str):
                    The optimization target for QAOA optimization: "physical_energy" or "logical_energy".
                    Default is "logical_energy".
                final_decoding (str):
                    The method for final decoding. Default is "logical_mean".
                num_spanningtrees (int):
                    The number of spanning trees to use in the decoding process.
                    If None, it defaults to the number of logical qubits.

            """

        # Validate problem_dict
        if not isinstance(problem_dict, dict) or not all(
                key in problem_dict for key in ["constraints", "constraint_strength", "local_fields"]):
            raise ValueError("problem_dict must contain 'constraints', 'constraint_strength', and 'local_fields'.")

        self._initialize_problem_parameters(problem_dict, n_max_gates)
        self._initialize_action_and_observation_spaces(actions)
        self._initialize_qaoa_optimizer(reps, qaoa_optimizer, optimization_target, num_spanningtrees,
                                        final_decoding)

        # Initialize learning parameters
        self.step_counter = 0
        self.program_str = ""
        self.observation = np.zeros(self.n_max_gates)

        # Reset the environment
        self.reset()

    def _initialize_problem_parameters(self, problem_dict, n_max_gates):
        self.constraints = problem_dict["constraints"]
        self.constraint_strength = problem_dict["constraint_strength"]
        self.local_fields = problem_dict["local_fields"]
        self.h_dict = hdict_physical_not_full(jij=self.local_fields, constraints=self.constraints,
                                              cstrength=self.constraint_strength)
        self.num_qubits = len(self.h_dict['X'].dims[0])
        self.num_qubits_logical = n_logical(self.num_qubits)

        self.n_max_gates = n_max_gates

    def _initialize_action_and_observation_spaces(self, actions):
        self.actions = actions or self.DEFAULT_ACTIONS
        self.n_actions = len(self.actions)
        self.action_space = Discrete(self.n_actions)
        self.observation_space = Box(low=0, high=self.n_actions, shape=(self.n_max_gates,), dtype=np.int32)

    def _initialize_qaoa_optimizer(self, reps, qaoa_optimizer, optimization_target, num_spanningtrees,
                                   final_decoding):

        self.reps = reps
        self.qaoa_optimizer = qaoa_optimizer
        self.optimization_target = optimization_target
        self.final_decoding = final_decoding

        if self.optimization_target == "logical_energy":
            self.num_spanningtrees = num_spanningtrees if num_spanningtrees is not None else self.num_qubits_logical
        else:
            self.num_spanningtrees = None

        # Initialize qutip qaoa
        self.problem_instance = QutipQaoa(h_dict=self.h_dict, jij=self.local_fields,
                                          program_string=''.join(self.actions),
                                          do_prediagonalization=True,
                                          psi0=None, optimization_target=self.optimization_target,
                                          final_decoding=self.final_decoding,
                                          num_spanningtrees=self.num_spanningtrees
                                          )

        self._action_to_gate = {i: self.actions[i - 1] for i in range(1, self.n_actions + 1)}
        self.program_str_data = {}

    def reset(self):
        self.step_counter = 0
        self.program_str = ""
        self.observation = np.zeros(self.n_max_gates)
        return self.observation

    def step(self, action):

        """
        Executes a single step in the QAOA environment based on the provided action.

        Args:
            action (int): The action index representing a gate to be applied.

        Returns:
            tuple:
                - observation (np.ndarray): Updated observation array representing the current gate sequence.
                - reward (float): The reward calculated based on the QAOA optimization result.
                - done (bool): A flag indicating if the episode has reached its end.
                - info (dict): Additional information including:
                    * "min_energy" (float): The minimum energy obtained from QAOA.
                    * "qaoa_time" (float): The time taken to perform QAOA.
                    * "gate_sequence" (str): The current gate sequence.
        """

        # Map the action to its corresponding gate and update the program string
        selected_gate = self._action_to_gate[action + 1]
        self.program_str += selected_gate
        self.observation[self.step_counter] = action + 1  # actions are 1,2 and 3

        # Initialize variables to track QAOA optimization results
        min_obj_value = None
        qaoa_time = None
        done = False
        reward = 0

        # Check if the maximum gate sequence length is reached, calculate reward, end episode
        if len(self.program_str) >= self.n_max_gates:
            min_obj_value, qaoa_time = self.calculate_min_energy_qaoa()
            reward = -min_obj_value / self.num_qubits  # negative energy density
            done = True

        self.step_counter += 1

        info = {"min_energy": min_obj_value,
                "qaoa_time": qaoa_time,
                "gate_sequence": self.program_str}

        return self.observation, reward, done, info

    def calculate_min_energy_qaoa(self):
        """
            Calculate the minimum energy for the current QAOA program string. The result is cached
            to avoid redundant calculations in future calls.

            Returns:
                tuple:
                    - min_obj_value (float): The minimum QAOA energy obtained from the optimization.
                    - time_taken_total (float): The total time taken for the QAOA optimization.
            """

        # Set the current program string for the QAOA instance
        self.problem_instance.set_programstring(self.program_str)

        # Check if the result for this program string is already cached
        if self.program_str in self.program_str_data:
            cached_data = self.program_str_data[self.program_str]
            return cached_data["energy"], cached_data["time"]

        # Ensure that the specified optimizer is valid
        if self.qaoa_optimizer not in self.SCIPY_OPTIMIZERS:
            raise ValueError(
                f"Optimizer '{self.qaoa_optimizer}' is not available. Valid options are: {self.SCIPY_OPTIMIZERS}")

        # Run the QAOA optimization process with "reps" random initialized angles
        result_container = self.problem_instance.repeated_scipy_optimization(
            optimization_method=self.qaoa_optimizer,
            param_ranges=None,
            reps=self.reps
        )

        # Extract the best QAOA energy and the time QAOA needed in total
        best_result = result_container.get_lowest_objective_value()
        min_obj_value = best_result.final_objective_value
        parameters = best_result.parameters
        time_taken_total = result_container.time_taken_total

        # Cache the result
        self.program_str_data[self.program_str] = {
            "energy": min_obj_value,
            "time": time_taken_total,
            "parameters": parameters
        }

        return min_obj_value, time_taken_total

    def valid_action_mask(self):
        """
            Generate a mask indicating valid actions based on the current state of the program string.

            Following limitations are applied:
            1. The first gate must not be an "X".
            2. Equal unitaries can’t be applied sequentially.
            3. Following two unitaries that commute, a different unitary must be applied.

            Returns:
                np.ndarray: A binary mask (1s and 0s) of length `self.n_actions`, where 1 indicates a valid action.
            """

        # Initialize all actions as valid (1)
        action_mask = np.ones(self.n_actions)

        # 1. first action must not be "X"
        if len(self.program_str) == 0:
            action_mask[self.actions.index("X")] = 0

        # 2. Equal unitaries can’t be applied sequentially.
        if len(self.program_str) != 0:
            last_gate = self.program_str[-1]
            action_mask[self.actions.index(last_gate)] = 0

        # 3. Following two unitaries that commute, a different unitary must be applied.
        if len(self.program_str) >= 2:
            gate_mapping = {"ZC": "Z", "CZ": "C", }
            last_two_gates = self.program_str[-2:]
            if last_two_gates in gate_mapping:
                action_mask[self.actions.index(gate_mapping[last_two_gates])] = 0

        return action_mask

    def render(self):
        """Often expected in a gym environment. Dummy function."""
        pass
