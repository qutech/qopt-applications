import numpy as np
from qopt import *
from qopt.optimize import SimulatedAnnealing

from cryoelectronics.cryoel_util import discretize

n_t = 10  # Number of time steps
total_t = 3 * 2 * np.pi  # total time.
delta_t = total_t / n_t  # Time Step
oversampling = 1

lin_freq_rel = 1
amp_max = 1  # Maximum amplitude
# Product of maximum amplitude and total time sufficient for 3 Rotations
# around the Bloch Sphere.
amp_min = 0  # Minimum amplitude
phase_min = 0
phase_max = np.pi  # full 2 pi control not necessary.

maximum_values = amp_max * np.ones((n_t, 2))
maximum_values[:, 1] = phase_max
minimum_values = amp_min * np.ones((n_t, 2))
minimum_values[:, 1] = phase_min

n_noise_traces = 15  # number of noise traces
sigma_noise = .1  # Standard deviation of the noise source

n_digits_discrete_opt = [6, 7, 8, 9, 10, 11, 12]

h_ctrl = [
    .5 * DenseOperator.pauli_x(),
    .5 * DenseOperator.pauli_y()
]

h_drift = [
    0 * DenseOperator.pauli_x()
]

h_noise = [
    .5 * DenseOperator.pauli_z()
]

rotation_angle = np.pi
target = (.5 * DenseOperator.pauli_x()).exp(tau=-1j * rotation_angle)

termination_conditions = {
    "min_gradient_norm": 1e-8,
    "min_cost_gain": 1e-8,
    "max_wall_time": 20 * 60.0,
    "max_cost_func_calls": int(1e6),
    "max_iterations": int(1e6),
    "min_amplitude_change": 1e-9
}


def amp_func_phase_control(x: np.array) -> np.array:
    """
    Creates control pulses in the rotating frame.

    Parameters
    ----------
    x: numpy array of float, shape = (n_t, 2)
        Pulse parameters being the amplitude x[:, 0] and the phase x[:, 1].

    Returns
    -------
    amplitudes: numpy array of float, shape = (n_t, 2)
        Control amplitudes appearing in the rotating frame Hamiltonian.

    """
    amplitudes = np.zeros_like(x)
    assert x.shape == (n_t * oversampling, 2)
    amplitudes[:, 0] = lin_freq_rel * x[:, 0] * np.cos(x[:, 1])
    amplitudes[:, 1] = lin_freq_rel * x[:, 0] * np.sin(x[:, 1])
    return amplitudes


def amp_func_deriv_phase_control(x: np.array) -> np.array:
    """
    Derivatives of the control pulses in the rotating frame.

    Parameters
    ----------
    x: numpy array of float, shape = (n_t, 2)
        Pulse parameters being the amplitude x[:, 0] and the phase x[:, 1].

    Returns
    -------
    derivs: numpy array of float, shape = (n_t, 2, 2)
        Control amplitudes appearing in the rotating frame Hamiltonian.

    """
    assert x.shape == (n_t * oversampling, 2)
    derivs = np.zeros((n_t * oversampling, 2, 2))
    derivs[:, 0, 0] = lin_freq_rel * np.cos(x[:, 1])
    derivs[:, 0, 1] = lin_freq_rel * np.sin(x[:, 1])
    derivs[:, 1, 0] = lin_freq_rel * -1 * x[:, 0] * np.sin(x[:, 1])
    derivs[:, 1, 1] = lin_freq_rel * x[:, 0] * np.cos(x[:, 1])
    return derivs


amplitude_function = CustomAmpFunc(
    value_function=amp_func_phase_control,
    derivative_function=amp_func_deriv_phase_control
)


ntg = NTGQuasiStatic(
    standard_deviation=[sigma_noise, ],
    n_samples_per_trace=n_t,
    n_traces=n_noise_traces,
    always_redraw_samples=False,
    correct_std_for_discrete_sampling=True,
    sampling_mode='uncorrelated_deterministic',
)


solver = SchroedingerSMonteCarlo(
    h_drift=h_drift,
    h_ctrl=h_ctrl,
    tau=delta_t * np.ones(n_t),
    h_noise=h_noise,
    noise_trace_generator=ntg,
    amplitude_function=amplitude_function
)

syst_cost_func = OperationInfidelity(
    solver=solver,
    target=target
)

qs_noise_cost_func = OperationNoiseInfidelity(
    solver=solver,
    target=target,
    neglect_systematic_errors=True
)
simulator = Simulator(
    solvers=[solver, ],
    cost_funcs=[syst_cost_func, qs_noise_cost_func],
    record_performance_statistics=False
)

bounds = np.zeros((2 * n_t, 2), dtype=float)
bounds[:n_t, 0] = amp_min
bounds[:n_t, 1] = amp_max
bounds[n_t:, 0] = phase_min
bounds[n_t:, 1] = phase_max

optimizer = ScalarMinimizingOptimizer(
    system_simulator=simulator,
    bounds=bounds,
    termination_cond=termination_conditions
)


def generate_inital_pulses(n_opt):
    """
    Generates initial pulse values.

    Parameters
    ----------
    n_opt: int
        Number of initial pulses

    Returns
    -------
    initial_pulses: numpy array, shape = (n_t, 2, n_opt)
        The random initial pulses.

    """
    initial_values = np.random.rand(n_t, 2, n_opt)
    initial_values[:, 0, :] = \
        initial_values[:, 0, :] * (amp_max - amp_min) + amp_min
    initial_values[:, 1, :] = \
        initial_values[:, 1, :] * (phase_max - phase_min) + phase_min
    return initial_values


def discretize_pulses(n_digits, pulses):
    """
    Creates discrete initial random pulses.

    Parameters
    ----------
    n_digits: int
        Number of binary digits in the discrete representation.

    pulses: numpy array, shape = (n_t, 2, n_opt)
        Pulses to be discretized. n_t is the number of time steps and n_opt
        the number of pulses.

    Returns
    -------
    integer_pulses: numpy array of int, shape = (n_t, 2, n_opt)
        The integer representation of the discretized pulses.

    discrete_pulses: numpy array of float, shape = (n_t, 2, n_opt)
        The discretized pulses.

    """
    max_values = np.zeros_like(pulses)
    min_values = np.zeros_like(pulses)
    max_values[:, 0, :] = amp_max
    max_values[:, 1, :] = phase_max
    min_values[:, 0, :] = amp_min
    min_values[:, 1, :] = phase_min
    integer_pulses, discrete_pulses = discretize(
        floating_point_number=pulses,
        maximum_values=max_values,
        minimum_values=min_values,
        n_bits=n_digits
    )
    return integer_pulses, discrete_pulses


# I need to implement these function as classes, because I need to pickle
# them in the parallelization.
class AmpFuncPhaseControlDiscrete:
    """
    This function is the discrete analogue to 'amp_func_phase_control'.
    """
    def __init__(self, n_digits):
        self.n_digits = n_digits

    def __call__(self, y):
        # scale the integer representation to the continuous representation
        x = np.zeros_like(y, dtype=float)
        x[:, 0] = y[:, 0] * (amp_max - amp_min) / (
                2 ** self.n_digits - 1) + amp_min
        x[:, 1] = y[:, 1] * (phase_max - phase_min) / (
                2 ** self.n_digits - 1) + phase_min

        amplitudes = np.zeros_like(x)
        assert x.shape == (n_t * oversampling, 2)
        amplitudes[:, 0] = lin_freq_rel * x[:, 0] * np.cos(x[:, 1])
        amplitudes[:, 1] = lin_freq_rel * x[:, 0] * np.sin(x[:, 1])
        return amplitudes


class AmpFuncDerivPhaseDiscrete:
    """
    These derivatives are not required, because gradient-based algorithms
    are not suited for the optimization of discrete functions.
    """
    def __call__(self, x):
        raise NotImplementedError


def create_discrete_optimizer(n_digits: int):
    """ Creates an instance of a simulated annealing optimizer.

    Parameters
    ----------
    n_digits: int
        Number of binary digits of the pulse representation.

    """

    amplitude_function_discrete = CustomAmpFunc(
        value_function=AmpFuncPhaseControlDiscrete(n_digits=n_digits),
        derivative_function=AmpFuncDerivPhaseDiscrete()
    )

    solver_discrete = SchroedingerSMonteCarlo(
        h_drift=h_drift,
        h_ctrl=h_ctrl,
        tau=delta_t * np.ones(n_t),
        h_noise=h_noise,
        noise_trace_generator=ntg,
        amplitude_function=amplitude_function_discrete
    )

    syst_cost_func_discrete = OperationInfidelity(
        solver=solver_discrete,
        target=target
    )

    qs_noise_cost_func_discrete = OperationNoiseInfidelity(
        solver=solver_discrete,
        target=target,
        neglect_systematic_errors=True
    )
    simulator_discrete = Simulator(
        solvers=[solver_discrete, ],
        cost_funcs=[syst_cost_func_discrete, qs_noise_cost_func_discrete],
        record_performance_statistics=False
    )

    bounds_discrete = np.zeros((2, n_t, 2), dtype=float)
    bounds_discrete[0, :, 0] = 0
    bounds_discrete[1, :, 0] = 2 ** n_digits - 1
    bounds_discrete[0, :, 1] = 0
    bounds_discrete[1, :, 1] = 2 ** n_digits - 1

    simanneal_optimizer = SimulatedAnnealing(
        system_simulator=simulator_discrete,
        step_size=1,
        step_ratio=.5,
        bounds=bounds_discrete,
        steps=10000,
        initial_temperature=1e-1,
        final_temperature=5e-6,
        store_optimizer=False
    )

    return simanneal_optimizer
