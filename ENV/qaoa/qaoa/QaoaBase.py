"""
Abstract base class for all kinds of QAOA implementations

@author: Kilian
"""

from abc import ABC, abstractmethod
from time import time
import warnings
import numpy as np
from qaoa.programs import SimpleProgram
from typing import List, Callable
from itertools import product


class QaoaSettings:
    def __init__(self, simulation_type, gate_sequence, ):
        pass


class Result:
    # simple class for returning results
    def __init__(self, **kwargs):
        self._set_default_attributes_to_None()
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def _set_default_attributes_to_None(self):
        self.parameters = None
        self.measurements = None
        self.measurement_labels = None
        self.objective_values = None
        self.final_objective_value = None
        self.final_measurements = None
        self.all_parameters = None
        self.time_taken = None
        self.nfeval = None
        self.best_configs = None

    def __repr__(self):
        strr = ''
        if not all(elem is None for elem in self.best_configs):
            strr += f'best configurations with probability {self.best_configs[2]:.3f}:'
            strr += f' physical: {self.best_configs[0]}, logical: {self.best_configs[1]} \n'
        strr += f'final objective value: {self.final_objective_value:.3f}'
        if self.time_taken is not None:
            strr += f'\ntime taken: {self.time_taken:.2f}s'
        if self.nfeval is not None:
            if self.time_taken is not None:
                strr += ', '
            else:
                strr += '\n'
            strr += f'nfeval: {self.nfeval}'
        # strr += '\nfinal measurements: ' + str(self.final_measurements)
        strr += '\nparameters: ' + ', '.join(['%.3f' % par for par in self.parameters])

        return strr

    def save_to_file(self, path):
        import json_tricks
        with open(path, 'w') as f:
            json_tricks.dump(self, f)
        return None

    @classmethod
    def load_from_file(cls, path) -> 'qaoa.QaoaBase.Result':
        import json_tricks
        res = json_tricks.load(path)

        return res


class ResultContainer:
    def __init__(self, results: List[Result] = None):
        self.results = []
        if results is not None:
            for res in results:
                self.results += [res]

    def __getitem__(self, item):
        return self.results[item]

    def __len__(self):
        return len(self.results)

    def add(self, result: Result):
        self.results += [result]

    @property
    def nfeval_total(self):
        return sum([res.nfeval for res in self.results])

    @property
    def time_taken_total(self):
        return sum([res.time_taken for res in self.results])

    def __repr__(self):
        best = self.get_lowest_objective_value()
        strr = ''
        strr += f'lowest objective value: {best.final_objective_value:.3f}'
        if best.time_taken is not None:
            strr += f'\ntotal time taken: {self.time_taken_total:.2f}s'
        if best.nfeval is not None:
            if self.time_taken_total is not None:
                strr += ', '
            else:
                strr += '\n'
            strr += f'nfeval: {self.nfeval_total}'

            return strr

    def get_lowest_objective_value(self):
        min_ob_val = np.inf
        min_ob_val_ind = 'hello'

        for ir, res in enumerate(self.results):
            if res.final_objective_value < min_ob_val:
                min_ob_val = res.final_objective_value
                min_ob_val_ind = ir

        return self.results[min_ob_val_ind]

    def get_lowest_measurement(self, measurement_index=0):
        min_meas_val = np.inf
        min_meas_ind = 'hello again'

        for ir, res in enumerate(self.results):
            if res.final_measurements[measurement_index] < min_meas_val:
                min_meas_val = res.final_measurements[measurement_index]
                min_meas_ind = ir

        return self.results[min_meas_ind]

    def get_highest_measurement(self, measurement_index=0):
        max_meas_val = -np.inf
        max_meas_ind = 'hello again'

        for ir, res in enumerate(self.results):
            if res.final_measurements[measurement_index] > max_meas_val:
                max_meas_val = res.final_measurements[measurement_index]
                max_meas_ind = ir

        return self.results[max_meas_ind]

    def save_to_file(self, path):
        for ir, res in enumerate(self.results):
            res.save_to_file(path + '_' + str(ir))

    @classmethod
    def load_from_file(cls, path):
        # to be implemented
        pass


class QaoaBase(ABC):
    def __init__(self, objective_function, final_energy_function=None):
        self.objective_function = objective_function
        if final_energy_function is not None:
            self.final_energy_function = final_energy_function
        else:
            self.final_energy_function = None
        if self.program is None:
            self.program = SimpleProgram()

    def set_programstring(self, programstring):
        self.program = type(self.program)(programstring)

    @abstractmethod
    def execute_circuit(self):
        pass

    def repeated_mc_opt(self,
                        reps: int = 3,
                        n_maxiterations: int = None,
                        temperature: float = None,
                        par_change_range: float = 0.25,
                        start_params: list = None,
                        rand_seed: int = None,
                        do_convergence_checks: bool = True,
                        return_timelines: bool = False,
                        measurement_functions: List[Callable] = None,
                        measurement_labels: list = None,
                        # parameter_index_to_change_low: int = None,
                        # parameter_index_to_change_high: int = None,
                        # parameter_indizes_to_change: list = None,
                        # fixed_changes_dict: dict = None
                        ):

        res_container = ResultContainer()
        for _ in range(reps):
            res_container.add(self.mc_optimization(
                n_maxiterations=n_maxiterations,
                temperature=temperature,
                par_change_range=par_change_range,
                start_params=start_params,
                rand_seed=rand_seed,
                do_convergence_checks=do_convergence_checks,
                return_timelines=return_timelines,
                measurement_functions=measurement_functions,
                measurement_labels=measurement_labels
            ))

        return res_container

    def multiple_random_init_mc_opt(self,
                                    reps: int = 3,
                                    n_maxiterations: int = None,
                                    temperature: float = None,
                                    par_change_range: float = 0.25,
                                    parameter_ranges: list = None,
                                    rand_seed: int = None,
                                    do_convergence_checks: bool = True,
                                    atol: float = None,
                                    return_timelines: bool = False,
                                    return_all_parameters: bool = False,
                                    measurement_functions: List[Callable] = None,
                                    measurement_labels: list = None,
                                    verbose: bool = False,
                                    # parameter_index_to_change_low: int = None,
                                    # parameter_index_to_change_high: int = None,
                                    # parameter_indizes_to_change: list = None,
                                    # fixed_changes_dict: dict = None
                                    ):

        if parameter_ranges is None:
            parameter_ranges = [(0, 2 * np.pi)] * len(self.program)

        res_container = ResultContainer()
        for _ in range(reps):
            init_parameters = np.random.random(len(self.program))
            for i in range(len(self.program)):
                init_parameters[i] = init_parameters[i] * (parameter_ranges[i][1] - parameter_ranges[i][0]) + parameter_ranges[i][0]

            res_container.add(self.mc_optimization(
                n_maxiterations=n_maxiterations,
                temperature=temperature,
                par_change_range=par_change_range,
                start_params=init_parameters,
                rand_seed=rand_seed,
                do_convergence_checks=do_convergence_checks,
                atol=atol,
                return_timelines=return_timelines,
                return_allparameters=return_all_parameters,
                measurement_functions=measurement_functions,
                measurement_labels=measurement_labels,
                verbose=verbose
            ))

            if verbose:
                print(f'run {_} done, took {res_container[-1].time_taken: .3f}s, '
                      f'objective value: {res_container[-1].final_objective_value: .3f}', flush=True)

        return res_container

    def mc_optimization(self, n_maxiterations: int = None,
                        temperature: float = None,
                        par_change_range: float = 0.25,
                        start_params: list = None,
                        rand_seed: int = None,
                        do_convergence_checks: bool = True,
                        atol: float = None,
                        return_timelines: bool = False,
                        return_allparameters: bool = False,
                        measurement_functions: List[Callable] = None,
                        measurement_labels: list = None,
                        verbose: bool = False,
                        parameter_index_to_change_low: int = None,
                        parameter_index_to_change_high: int = None,
                        parameter_indizes_to_change: list = None,
                        fixed_changes_dict: dict = None) -> Result:
        """
        :param verbose: If true prints current optimization output, only works with convergence checks
        :param measurement_labels: labels for measurments, mainly used for plot_result
        :param fixed_changes_dict: Dictionary of form {(i1, i2,..): [list of fixed steps the parameters at these indices can take], ...}
        :param n_maxiterations: Monte Carlo optimization always aborts at N_maxiterations
        :param temperature: effective temperature of MC optimization
        :param par_change_range: changes parameters in [-par_change_range, +par_change_range]
        :param start_params: initial QAOA angles, if None = [0,0,0,0...]
        :param rand_seed: random seed for the MC optimization
        :param atol: absolute value std of convergence window < atol
        :param do_convergence_checks: do checks if MC optimization is converged,
        with this flag set timelines are saved also, not returned
        :param return_timelines: returns the full timeline of objective values and measurements if set
        :param return_allparameters: returns full timeline of parameters if set
        :param measurement_functions: called on state at every measurment step if return_timelines is on,
        else only once at the end of the run, is what is returned by the implementation of execute_circuit
        :param parameter_index_to_change_low: if set only parameters in half-open interval [low, high) are changed
        :param parameter_index_to_change_high: if set only parameters in half-open interval [low, high) are changed
        :param parameter_indizes_to_change: i set only parameters in this list are changed, ignoring parameter_index_to_change_low/high
        :return: Result of class Result, with parameters, objective value (and optionally the timelines)
        """

        t0 = time()
        result = Result()
        # TODO maybe make this depend on N/problem?
        if temperature is None:
            temperature = 0.02

        if rand_seed is not None:
            np.random.seed(rand_seed)

        if start_params is not None:
            if len(start_params) == len(self.program):
                self.program.linearparameters = start_params.copy()
            else:
                print('start_params has wrong dimension, using previous parameters')
        result.x0 = self.program.linearparameters

        if n_maxiterations is None:
            n_maxiterations = 25 * len(self.program) ** 2

        save_timelines = False
        # convergence stuff
        if verbose and not do_convergence_checks:
            print('verbose=True only works with convergence checks')
        if do_convergence_checks:
            convergence_window = int(30 * len(self.program))
            after_x_steps_check = int(4 * len(self.program))
            converged = False
            save_timelines = True
            if atol is None:
                # warnings.warn('atol set to 0.01, no idea if this makes sense')
                atol = 0.01
            elif atol < 0:
                atol = abs(atol)

        nochangecounter = 0  # do this always, less ifs

        if return_timelines:
            save_timelines = True

        if measurement_functions is None:
            measurement_functions = []

        # if its only a single function put it into list
        if not isinstance(measurement_functions, list):
            measurement_functions = [measurement_functions]

        if return_allparameters:
            all_parameters = np.zeros(shape=(n_maxiterations, len(self.program)))
        if save_timelines:
            measurements = []
            for _ in measurement_functions:
                measurements += [[]]
            measurements += [np.zeros(n_maxiterations)]  # for objective function

        # preparation, 0th step
        if parameter_index_to_change_low is None:
            parameter_index_to_change_low = 0
        if parameter_index_to_change_high is None:
            parameter_index_to_change_high = len(self.program)

        assert parameter_index_to_change_high > parameter_index_to_change_low, "index range cannot be empty"

        if parameter_indizes_to_change is not None:
            assert min(parameter_indizes_to_change) >= 0, "index must be >= 0"
            assert max(parameter_indizes_to_change) < len(self.program), "index must be < len(program)"
            indices = np.random.choice(parameter_indizes_to_change, size=n_maxiterations)
        else:
            indices = np.random.randint(low=parameter_index_to_change_low, high=parameter_index_to_change_high,
                                        size=n_maxiterations)

        change = 2 * par_change_range * (np.random.random(n_maxiterations) - 0.5)

        if fixed_changes_dict is not None:
            for inds, fixed_values in fixed_changes_dict.items():
                for index in inds:
                    mask = np.where(indices == index)
                    change[mask] = np.random.choice(fixed_values, size=(len(mask),))

        state = self.execute_circuit()
        accepted_state = state.copy()
        old_objective_value = self.objective_function(state)

        # saving starting conditions
        if return_allparameters:
            all_parameters[0] = self.program.linearparameters.copy()
        if save_timelines:
            measurements[-1][0] = old_objective_value
            for im, mf in enumerate(measurement_functions):
                measurements[im] += [mf(state)]

        # loop from i=1 to end
        i = 1
        for i in range(1, n_maxiterations):
            # print(i)
            # update parameters
            self.program[indices[i]] += change[i]

            # calculate new state and energy
            state = self.execute_circuit()
            new_objective_value = self.objective_function(state)

            # check if update will be accepted
            accept = False
            if new_objective_value < old_objective_value:
                accept = True
            elif temperature == 0.0:
                accept = False
            else:
                if np.random.random() < np.exp((- new_objective_value + old_objective_value) / temperature):
                    accept = True
            if accept:  # accept new state and energy
                accepted_state = state.copy()
                old_objective_value = new_objective_value
                nochangecounter = 0
            else:  # update not accepted, undo changes to parameters
                # TODO like this it can happen that the last accepted parameters are not the ones with the minimal energy
                # maybe do: whenever a new minimal energy is reached save the corresponding parameters
                nochangecounter += 1
                self.program[indices[i]] -= change[i]

            # measure again after update is done (or rejected)
            if return_allparameters:
                all_parameters[i] = self.program.linearparameters.copy()
            if save_timelines:
                measurements[-1][i] = old_objective_value
                for im, mf in enumerate(measurement_functions):
                    measurements[im] += [mf(state)]

            # convergence checks
            if do_convergence_checks:
                if (i % after_x_steps_check) == 0 and i > 2 * convergence_window:
                    current_deviation = np.std(measurements[-1][i - convergence_window:i])
                    if verbose:
                        print(f'it: {i}, objective value: {old_objective_value: .3f}, current deviation: {current_deviation: .3f}')
                    if current_deviation < atol:
                        converged = True
                        result.exit_state = 'converged: std < atol'

                if nochangecounter >= 2 * convergence_window:
                    result.exit_state = f'converged: no change for 2 * convergence_window = {2 * convergence_window} parameter updates'
                    converged = True

                if converged:
                    break

        result.parameters = self.program.linearparameters
        result.final_objective_value = old_objective_value




        if measurement_labels is not None:
            if len(measurement_labels) != len(measurement_functions):
                print('amount of measurement labels wrong')
            else:
                result.measurement_labels = measurement_labels

        result.final_measurements = [mf(accepted_state) for mf in measurement_functions]

        if return_timelines:
            result.objective_values = measurements[-1][:i + 1]
            result.measurements = [measurement[:i + 1] for measurement in measurements[:-1]]
        if return_allparameters:
            result.all_parameters = all_parameters

        result.time_taken = time() - t0
        result.nfeval = i + 1
        return result

    def basin_hopping(self, x0=None, args=None, minimizer_kwargs=None):
        from scipy.optimize import basinhopping
        t0 = time()

        def minf(params):
            return self.objective_function(self.execute_circuit(params))

        if x0 is None:
            x0 = [0] * len(self.program)
        if args is None:
            args = {'niter': 100, 'T': 0.5, 'stepsize': 0.5}
        if minimizer_kwargs is None:
            minimizer_kwargs = {'method': 'L-BFGS-B',
                                'options':
                                    {'ftol': 1e-3, 'maxfun': 100}
                                }
            # minimizer_kwargs = {'method': 'Nelder-Mead',
            #                     'options':
            #                         {'ftol': 1e-3, 'maxfev': 100}
            #                     }
        bhres = basinhopping(minf, x0, minimizer_kwargs=minimizer_kwargs, **args)
        return Result(parameters=bhres.x, final_objective_value=bhres.fun,
                      fulloutput=bhres, nfeval=bhres.nfev, time_taken=time() - t0)

    def scipy_optimization(self, x0=None, method=None, bounds=None, opts=None):
        best_config_physical = None
        best_config_logical = None
        max_prob = None
        from scipy import optimize
        t0 = time()

        def minf(params):
            return self.objective_function(self.execute_circuit(params))

        if x0 is None:
            x0 = (np.random.random(len(self.program)) - 0.5) * 4
        # if opts is None:
        #     # opts = {'maxiter': 500 * len(self.program)}
        #     opts = {'maxfun': 1000}
        if method is None:
            method = 'Nelder-Mead'
        scipy_res = optimize.minimize(minf, x0, method=method, bounds=bounds)
        if self.final_energy_function is not None:
            final_objective_value, max_prob, best_config_physical, best_config_logical = self.final_energy_function(self.execute_circuit(scipy_res.x))
        else:
            final_objective_value = scipy_res.fun
        return Result(parameters=scipy_res.x, final_objective_value=final_objective_value, best_configs=(best_config_physical, best_config_logical, max_prob),
                      fulloutput=scipy_res, nfeval=scipy_res.nfev, time_taken=time() - t0)

    def repeated_scipy_optimization(self, optimization_method='l-bfgs-b', param_ranges=None, reps=100, seed=None):
        np.random.seed(seed)
        if param_ranges is None:
            param_ranges = [(0, 2 * np.pi)] * len(self.program.program_string)

        rc = ResultContainer()

        for _ in range(reps):
            init_params = [np.random.uniform(*prange) for prange in param_ranges]
            res = self.scipy_optimization(x0=init_params, method=optimization_method, bounds=param_ranges)
            res.x0 = init_params.copy()
            rc.add(res)

        return rc

    def direct_optimization(self, maxf=10000, pifac=1):
        from scipydirect import minimize

        def f(x):
            return self.objective_function(self.execute_circuit(x))

        bounds = [(-pifac * np.pi, pifac * np.pi)] * len(self.program)

        direct_res = minimize(f, bounds, disp=True, algmethod=1, maxf=maxf)

        result = Result(parameters=self.program.linearparameters, fulloutput=direct_res)

        return result

    def grid_search(self, grid_points=10, par_ranges=None, do_local_optimization=True) -> Result:
        """

        :param grid_points: The number of grid points in each dimension (integer). The grid points
            are then automatically chosen equidistantly in the specified range.
        :param par_ranges: The ranges the parameters take. Must be given as an array of 2-tuples.
            The order of the ranges must match the order of the parameter letters in the program string.
        :param do_local_optimization: Specifies whether the local optimum around each grid point is
            searched for. If True, the area around each grid point scanned for a local optimum by using
            the bounded l-bfgs (l-bfgs-b) method.
        :return: An instance of Result.
        """
        if par_ranges is None:
            par_ranges = [(0, 2 * np.pi)] * len(self.program.program_string)

        point_distances = [(par_range[1] - par_range[0]) / grid_points for par_range in par_ranges]
        border_grid_coordinates = [(par_range[0] + point_distances[i] / 2,
                                    par_range[1] - point_distances[i] / 2)
                                   for i, par_range in enumerate(par_ranges)]
        parameter_sets = [list(item)
                          for item in product(*[np.linspace(*outer_coords, grid_points)
                                                for outer_coords in border_grid_coordinates])]
        mine = self.objective_function(self.execute_circuit())
        best_params = self.program.linearparameters
        energies = [None] * len(parameter_sets)

        for i, param_set in enumerate(parameter_sets):
            self.program.linearparameters = param_set
            energies[i] = self.objective_function(self.execute_circuit())
            if do_local_optimization:
                local_bounds = [(param_set[i] - point_distances[i] / 2, param_set[i] + point_distances[i] / 2)
                                for i in range(len(param_set))]
                local_opt_res = self.scipy_optimization(x0=param_set, bounds=local_bounds, method='l-bfgs-b')
                if local_opt_res.fun < mine:
                    mine = local_opt_res.fun
                    best_params = local_opt_res.x

            elif energies[i] < mine:
                mine = energies[i]
                best_params = self.program.linearparameters

        res = Result(parameters=best_params, final_objective_value=mine,
                     objective_values=energies, all_parameters=parameter_sets)

        return res

    def repeated_bfgs_search(self, par_ranges=None, n_repeat=10000, seed=None):
        """
        Chooses n_repeat random points in the parameter space and tries to find the local optimum with
            these random points as initial parameters, using the bfgs method. The global optimum is
            assumed to be the best result from the local optima.
        :param par_ranges: The ranges the parameters take. Must be given as an array of 2-tuples.
            The order of the ranges must match the order of the parameter letters in the program string.
        :param n_repeat: The number of repetitions of the algorithm (integer).
        :param seed: Random seed. Optional, in order to make the search deterministic (i. e. the results
            are reproducible).
        :return: An instance of Result.
        """

        np.random.seed(seed)
        if par_ranges is None:
            par_ranges = [(0, 2 * np.pi)] * len(self.program.program_string)

        rc = ResultContainer()

        for _ in range(n_repeat):
            init_params = [np.random.uniform(*prange) for prange in par_ranges]
            res = self.scipy_optimization(x0=init_params, method='l-bfgs-b', bounds=par_ranges)
            res.x0 = init_params.copy()
            rc.add(res)

        return rc
