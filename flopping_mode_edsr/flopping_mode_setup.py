# -*- coding: utf-8 -*-
# =============================================================================
#     qopt-applications
#     Copyright (C) 2020 Julian Teske, Forschungszentrum Juelich
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program. If not, see <http://www.gnu.org/licenses/>.
#
#     Contact email: j.teske@fz-juelich.de
# =============================================================================

r"""
This file contains the core implementation of the flopping-mode EDSR model and
convenience functions for the creation of qopt object instances.

The core implementation contains
- physical constants
- operators
- default parameters
- Hamiltonian functions
- flopping mode calculations: calculate pulse related parameters
- qopt solvers
- custom  cost function
- qopt simulators


Setting the reduced Planck constant to unity, all quantities are measured in
nanoseconds or inverse nanoseconds if not mentioned otherwise. Inverse
seconds are used to quantify angular frequencies and Hertz are used for
regular frequencies.

[1]: accompanied article publication: coming soon

"""


from qopt import *
from qopt.cost_functions import CostFunction
from qopt.noise import sample_1dim_gaussian_distribution
import numpy as np
from scipy.optimize import minimize
import datetime

from typing import Union

from scipy.constants import physical_constants, elementary_charge, giga, \
    micro, milli

# ############################ Physical Constants #############################

hbar = physical_constants["reduced Planck constant"][0]
mu_bohr = physical_constants["Bohr magneton"][0]
g_fac_Si = 2
g_fac_GaAs = -0.44

milli_tesla_GaAs_by_inverse_nano_seconds = \
    g_fac_GaAs * mu_bohr / hbar / giga * milli
milli_tesla_Si_by_inverse_nano_seconds = \
    g_fac_Si * mu_bohr / hbar / giga * milli
micro_eV_by_inverse_nano_seconds = elementary_charge / hbar / giga * micro

nano_eV_squared_per_hertz_by_inverse_nano_seconds = \
    1e-6 * micro_eV_by_inverse_nano_seconds ** 2 / (2 * np.pi * 1e-9)
# 1e-6: converts nano to micro eV (squared)
# micro_eV_by_inverse_nano_seconds ** 2: converts to inverse_nano_secods ** 2
# (2 * np.pi * 1e-9): inverse nano second divided by hertz


# ############################ Operators ######################################

# operators

# paulis
p_0 = DenseOperator.pauli_0()
p_x = DenseOperator.pauli_x()
p_y = DenseOperator.pauli_y()
p_z = DenseOperator.pauli_z()

# projectors
p_l = DenseOperator(np.diag([1, 0]))
p_r = DenseOperator(np.diag([0, 1]))

tau_z = DenseOperator.pauli_0().kron(DenseOperator.pauli_z())
tau_x = DenseOperator.pauli_0().kron(DenseOperator.pauli_x())

sigma_z = DenseOperator.pauli_z().kron(DenseOperator.pauli_0())
sigma_x = DenseOperator.pauli_x().kron(DenseOperator.pauli_0())

# operators including the valley state

tau_x_v = p_x.kron(p_0.kron(p_0))
tau_z_v = p_z.kron(p_0.kron(p_0))
sigma_z_v = p_0.kron(p_0.kron(p_z))
sigma_x_v = p_0.kron(p_0.kron(p_x))
p_l_rho_x = p_l.kron(p_x.kron(p_0))
p_l_rho_y = p_l.kron(p_y.kron(p_0))
p_r_rho_x = p_r.kron(p_x.kron(p_0))
p_r_rho_y = p_r.kron(p_y.kron(p_0))

# spin-orbit model
h_ctrl = [
    .5 * tau_z,
    .5 * sigma_z,
    .5 * tau_x,
    .5 * sigma_x * tau_z,
    .5 * sigma_z * tau_z
]

# spin-orbit-valley model
h_ctrl_valley = [
    .5 * tau_z_v,
    .5 * sigma_z_v,
    .5 * tau_x_v,
    .5 * sigma_x_v * tau_z_v,
    .5 * sigma_z_v * tau_z_v,
    .5 * p_l_rho_x,
    .5 * p_l_rho_y,
    .5 * p_r_rho_x,
    .5 * p_r_rho_y,
]

# operators for a valley-orbit model

tau_x_valley_orbit = p_x.kron(p_0)
tau_z_valley_orbit = p_z.kron(p_0)
p_l_rho_x_valley_orbit = p_l.kron(p_x)
p_l_rho_y_valley_orbit = p_l.kron(p_y)
p_r_rho_x_valley_orbit = p_r.kron(p_x)
p_r_rho_y_valley_orbit = p_r.kron(p_y)

# ############################ Default Parameters #############################


# Default Parameters
default_termination_conditions = {
    "min_gradient_norm": 1e-7,
    "min_cost_gain": 5e-10,
    "max_wall_time": 240 * 60.0,
    "max_cost_func_calls": 1e6,
    "max_iterations": 200,
    "min_amplitude_change": 1e-8
}

default_par_dict = {
    'eps': 0,  # Detuning
    'tunnel_coupling': 100 * micro_eV_by_inverse_nano_seconds,
    'e_zeeman': 20 * milli_tesla_Si_by_inverse_nano_seconds,  # Zeeman energy
    'gmubbx': .4 * milli_tesla_Si_by_inverse_nano_seconds,
    # Magnetic gradient in x direction
    'gmubbz': .1 * milli_tesla_Si_by_inverse_nano_seconds,
    # Magnetic gradient in z direction
    'amp': 320 * micro_eV_by_inverse_nano_seconds,  # Pulse amplitude
    'rabi_rotation_angle': np.pi,
    # Rotation angle of the target gate about x-axis
    'azimuth_angle_target': 0.,
    # Rotation angle of the target gate about z-axis, i.e. the azimuth angle
    'opt_azimuth_angle': None,  # Correction for the rotation about the z-axis
    'opt_azimuth_angle_valley': None,
    'use_opt_azimuth_angle': True,
    # Flag: if true, compute a correction of the azimuth angle
    'initial_spin_state': None,
    'time': datetime.datetime.now(),
    'min_n_time_steps': int(1e4),
    'rabi_freq_calc_meth': 'heuristic',  # see comment below
    'eps_noise_std': 1 * micro_eV_by_inverse_nano_seconds,
    # standard deviation of the quasi-static noise on the detuning
    'eps_res_noise_std': .003 * milli_tesla_Si_by_inverse_nano_seconds,
    # standard deviation of the quasi-static noise on the zeeman energy
    'white_noise_psd':
        .02 ** 2 * nano_eV_squared_per_hertz_by_inverse_nano_seconds,
    # spectral density of the white noise on the detuning
    'n_traces_fast_noise': 100,
    # number of noise traces for the Monte Carlo simulation of white noise
    'n_processes_fast_noise': 50,
    # number of parallel processes for the Monte Carlo simulation of white
    # noise
    'fast_mc_freq_cutoff': 10,  # Cutoff frequency for the white noise (GHz)
    'n_traces': 8,  # Number of traces for the Monte Carlo calculation of
    # quasi-static noise
    'pulse_envelope': True,
    # Flag: if true, use an envelope for the cosine pulse
    'pulse_mode': 'duty_cycle',  # Pulse shape. 'duty_cycle' for rectangular
    # pulse or 'sine' for the cosine pulse
    'leakage_method': 'partial_trace',  # Method to treat leakage states when
    # calculating the state fidelity. Use 'partial_trace' to calculate a
    # partial trace over the leakage degrees of freedom. Use 'cut' to truncate
    # to the computational states.
    'tanh_range': 8,  # steepness of the rectangular pulse
    'phase_shift': 0,  # phase shift is applied to the pulse
    'time_correction_factor': None,  # can be used in the calculation of the
    # Rabi frequency guess
    'termination_conditions': default_termination_conditions,
    'n_time_steps_dc': 500,  # number of time steps for the rectangular pulse
    'bounds': None,  # Boundaries for the optimization parameters
    'initial_control_amplitudes': None,
    'c_duty_cycle': 0.,  # Duty cycle parameter
    'time_scale_par': 1.,  # time scale parameter
    "resonance_frequency": 0,
    'get_init_val_grid': True,  # Flag: if True, evaluate the fidelity on a
    # grid to investigate optimization landscape to find optimal initial values
    # in the optimization
    'init_val_grid_ranges': ((0, 0.3), (0.998, 1.0005)),  # Ranges for the grid
    'value_grid': None,  # Values from the grid calculation
    'n_init_grid_vals': 20,  # Number of grid points along each axis
    'discretization_delta': .02,  # Meta parameter. Defines the conditions
    # for the calculation of the optimal time step.
    'save_intermediary_steps': False,  # Save more data during the optimization
    'verbose': 1  # see below
}
"""
The default parameter dictionary holds a set of standard parameters. 

verbose: 
1: report on pulse optimization and fast noise MC.
2: report on slow qs-noise MC and Liouville.
3: report on state and gate infid.

rabi_freq_calc_meth:
heuristic: formula works well for our desired parameter regime, but does not 
work for detuned dots.
numeric: works well for small amplitudes.

"""


# ############################ Hamiltonian ####################################


def create_hamiltonian(
        tunnel_coupling, eps, gmubbx, gmubbz, e_zeeman, **_
) -> DenseOperator:
    """
    Create the Hamiltonian of a single electron in a double quantum dot.

    Parameters
    ----------
    tunnel_coupling: float
        Tunnel coupling of the double quantum dot.

    eps: float
        Detuning of the double quantum dot. This is the difference in chemical
        potential between the two dots.

    gmubbx: float
        Magnetic gradient across the double quantum dot in x direction.
        Multiplied with the g-factor and Bohr magneton.

    gmubbz: float
        Magnetic gradient across the double quantum dot in z direction.
        Multiplied with the g-factor and Bohr magneton.

    e_zeeman: float
        Average Zeeman energy.

    Returns
    -------
    hamiltonian: DenseOperator
        The Hamiltonian.

    """
    h_drift = (
        .5 * e_zeeman * sigma_z
        + .5 * eps * tau_z
        + tunnel_coupling * tau_x
        + 0.5 * gmubbx * sigma_x * tau_z
        + 0.5 * gmubbz * sigma_z * tau_z
    )
    return h_drift


def create_hamiltonian_valley(
        tunnel_coupling, eps, gmubbx, gmubbz, e_zeeman, real_valley_l,
        imag_valley_l, real_valley_r, imag_valley_r, **_
) -> DenseOperator:
    """
    Create the Hamiltonian of a single electron in a double quantum dot.

    Parameters
    ----------
    tunnel_coupling: float
        Tunnel coupling of the double quantum dot.

    eps: float
        Detuning of the double quantum dot. This is the difference in chemical
        potential between the two dots.

    gmubbx: float
        Magnetic gradient across the double quantum dot in x direction.
        Multiplied with the g-factor and Bohr magneton.

    gmubbz: float
        Magnetic gradient across the double quantum dot in z direction.
        Multiplied with the g-factor and Bohr magneton.

    e_zeeman: float
        Average Zeeman energy.

    real_valley_l: float
        Real part of the valley splitting in the left quantum dot.

    imag_valley_l: float
        Imaginary part of the valley splitting in the left quantum dot.

    real_valley_r: float
        Real part of the valley splitting in the right quantum dot.

    imag_valley_r: float
        Imaginary part of the valley splitting in the right quantum dot.


    Returns
    -------
    hamiltonian: DenseOperator
        The Hamiltonian.

    """
    h_drift = (
        .5 * e_zeeman * sigma_z_v
        + .5 * eps * tau_z_v
        + tunnel_coupling * tau_x_v
        + 0.5 * gmubbx * sigma_x_v * tau_z_v
        + 0.5 * gmubbz * sigma_z_v * tau_z_v
        + .5 * real_valley_l * p_l_rho_x
        + .5 * imag_valley_l * p_l_rho_y
        + .5 * real_valley_r * p_r_rho_x
        + .5 * imag_valley_r * p_r_rho_y
    )
    return h_drift


def create_hamiltonian_valley_orbit(
        tunnel_coupling, eps, real_valley_l, imag_valley_l, real_valley_r, imag_valley_r
) -> DenseOperator:
    """
    Create the Hamiltonian of a single electron in a double quantum dot
    neglecting the spin state.

    Parameters
    ----------
    tunnel_coupling: float
        Tunnel coupling of the double quantum dot.

    eps: float
        Detuning of the double quantum dot. This is the difference in chemical
        potential between the two dots.

    real_valley_l: float
        Real part of the valley splitting in the left quantum dot.

    imag_valley_l: float
        Imaginary part of the valley splitting in the left quantum dot.

    real_valley_r: float
        Real part of the valley splitting in the right quantum dot.

    imag_valley_r: float
        Imaginary part of the valley splitting in the right quantum dot.


    Returns
    -------
    hamiltonian: DenseOperator
        The Hamiltonian.

    """
    h_drift = (
        + .5 * eps * tau_z_valley_orbit
        + tunnel_coupling * tau_x_valley_orbit
        + .5 * real_valley_l * p_l_rho_x_valley_orbit
        + .5 * imag_valley_l * p_l_rho_y_valley_orbit
        + .5 * real_valley_r * p_r_rho_x_valley_orbit
        + .5 * imag_valley_r * p_r_rho_y_valley_orbit
    )
    return h_drift


def exp_val_sigz_dqd(detuning, tunnel_coupling):
    """
    Expectation value of the pauli-z position operator in a double quantum dot.

    The calculation is based on analytical diagonalization of the Hamiltonian
    of a double quantum dot with given detuning and tunnel coupling. The
    expectation value is calculated in the ground state.

    Parameters
    ----------
    detuning: numpy array, shape (n_time_steps)
        The detuning value or multiple detuning values of the double quantum
        dot. The detuning is the difference in chemical potential between the
        two dots.

    tunnel_coupling: float
        The tunnel coupling of the two dots.

    Returns
    -------
    expectation_value. numpy array, shape (n_time_steps)
        The expectation value of the sigma_z operator in the ground state.

    """
    val = (
        detuning * (detuning - np.sqrt(
        detuning ** 2 + 4 * tunnel_coupling ** 2)
            )
        / (
        detuning ** 2 - detuning * np.sqrt(
            detuning ** 2 + 4 * tunnel_coupling ** 2)
        + 4 * tunnel_coupling ** 2
        )
    )
    return val

# ############################ Flopping Mode Calculations #####################


def calculate_resonance_rabi_freq(
        amp, tunnel_coupling, eps, gmubbx, gmubbz, e_zeeman, pulse_mode,
        amp_correction=1, rabi_freq_calc_meth='heuristic', **_):
    """
    Calculates the resonance and Rabi-frequency of the flopping mode quibt.

    For the rectangular pulse, we use as coarse approximation the Zeemann
    energy as resonance frequency and half the perpendicular magnetic field as
    Rabi frequency.

    For a cosine or sine pulse, we transform the hamiltonian in its eigenbasis.
    The energy difference of the lowest two energy levels gives the resonance
    frequency of the transition we want to drive. The derivative of the
    off-diagonal element approximates the Rabi frequency.
    See [1] for details.

    Parameters
    ----------
    amp: float
        Pulse amplitude.

    tunnel_coupling: float
        Tunnel coupling of the double quantum dot.

    eps: float
        Detuning of the double quantum dot. This is the difference in chemical
        potential between the two dots.

    gmubbx: float
        Magnetic gradient across the double quantum dot in x direction.
        Multiplied with the g-factor and Bohr magneton.

    gmubbz: float
        Magnetic gradient across the double quantum dot in z direction.
        Multiplied with the g-factor and Bohr magneton.

    e_zeeman: float
        Average Zeeman energy.

    pulse_mode: string
        Giving the pulse type. Supported are "duty_cycle" for the rectangular
        pulse and "sine" for a sine or cosine pulse.

    amp_correction: float
        Only relevant for pulse_mode="sine".
        This factor allows corrections of the amplitude to account for the
        effective amplitude reduction by the envelope. Defaults to 1.

    rabi_freq_calc_meth: string
        Only relevant for pulse_mode="sine". Determines the method for the
        calculation of the Rabi frequency.
        'numeric' is a general method that bases the value on the off-diagonal
        elements of the driving hamiltonian.
        'heuristic' uses a heuristic formula. This is a good approximation
        for strong driving and zero detuning.
        Defaults to 'heuristic'.

    """

    if pulse_mode == "duty_cycle":
        resonance_frequency_guess = e_zeeman
        rabi_frequency_guess = .5 * gmubbx

    elif pulse_mode == "sine":

        h_drift = create_hamiltonian(
            tunnel_coupling=tunnel_coupling, eps=eps, gmubbx=gmubbx,
            gmubbz=gmubbz, e_zeeman=e_zeeman)

        eig_vals, eig_vecs = h_drift.spectral_decomposition(hermitian=True)
        eigenbasis = DenseOperator(eig_vecs)

        # We assume the orbital splitting to be greater than the spin
        # splitting. We want to drive spin transitions between the two lowest
        # eigenstates.
        resonance_frequency_guess = eig_vals[1] - eig_vals[0]  # in 1/ns

        if rabi_freq_calc_meth == 'numeric':
            # To calculate the Rabi-frequency we approximate the derivative of
            # the off-diagonal element connecting the lowest two energy states
            # by a finite difference.
            h_plus = create_hamiltonian(
                tunnel_coupling=tunnel_coupling,
                eps=eps + amp * amp_correction,
                gmubbx=gmubbx, gmubbz=gmubbz, e_zeeman=e_zeeman)

            h_minus = create_hamiltonian(
                tunnel_coupling=tunnel_coupling,
                eps=eps - amp * amp_correction,
                gmubbx=gmubbx, gmubbz=gmubbz, e_zeeman=e_zeeman)

            h_diff = h_plus - h_minus
            h_diff_transformed = eigenbasis.dag() * h_diff * eigenbasis

            rabi_frequency_guess = np.abs(h_diff_transformed[0, 1])

        elif rabi_freq_calc_meth == 'heuristic':
            rabi_frequency_guess = gmubbx / 4 * (
                    amp / 2 / tunnel_coupling) ** .5
        else:
            raise ValueError('Unknown Rabi-Frequency calculation method.')
    else:
        raise ValueError('Unknown pulse mode!')
    return resonance_frequency_guess, rabi_frequency_guess


def rabi_freq_correction_duty_cycle_pulse(
        resonance_frequency_guess, amp, tunnel_coupling, eps, gmubbx, gmubbz,
        e_zeeman, pulse_mode, total_time_guess, phase_shift, n_time_steps,
        tanh_range, n_time_steps_dc, **_):
    """
    Calculate a correction to the Rabi frequency of the rectangular pulse.

    For a coarse estimation of a single period of the rectangular pulse, this
    function estimates the corresponding polar rotation of the electron spin.
    See [1] for details.

    Parameters
    ----------
    resonance_frequency_guess: float
        Approximation of the resonance frequency.

    amp: float
        Pulse amplitude.

    tunnel_coupling: float
        Tunnel coupling of the double quantum dot.

    eps: float
        Detuning of the double quantum dot. This is the difference in chemical
        potential between the two dots.

    gmubbx: float
        Magnetic gradient across the double quantum dot in x direction.
        Multiplied with the g-factor and Bohr magneton.

    gmubbz: float
        Magnetic gradient across the double quantum dot in z direction.
        Multiplied with the g-factor and Bohr magneton.

    e_zeeman: float
        Average Zeeman energy.

    pulse_mode: string
        Giving the pulse type. Supported are "duty_cycle" for the rectangular
        pulse and "sine" for a sine or cosine pulse.

    total_time_guess: float
        Estimation of the total pulse time before we correct the Rabi
        frequency.

    phase_shift: float
        Angle for a phase shift of the applied pulse.

    n_time_steps: int
        Number of time steps in the time discretization.

    tanh_range: float
        Range parameter of the hyperbolic tangent. Determines the smoothness
        of the rectangular pulse.

    n_time_steps_dc: int
        Number of time steps in every period of the rectangular pulse.

    Returns
    -------
    correction: float
        Correction parameter for the estimated Rabi frequency.

    """

    resonance_time_guess = 2 * np.pi / resonance_frequency_guess
    time = (total_time_guess / n_time_steps) * np.ones(n_time_steps)
    acc_time = np.cumsum(time)

    value_func = AmplitudeValueFunc(
        n_time_steps=n_time_steps, acc_time=acc_time, eps=eps,
        total_time_guess=total_time_guess, amp=amp, e_zeeman=e_zeeman,
        gmubbz=gmubbz, gmubbx=gmubbx, tunnel_coupling=tunnel_coupling,
        pulse_envelope=None, pulse_mode=pulse_mode,
        resonance_time_guess=resonance_time_guess, tanh_range=tanh_range,
        phase_shift=phase_shift
    )

    pulse_segment = value_func(
        np.asarray([[0, 1]]))[0:n_time_steps_dc//2, 0]
    exp_val_sigz = exp_val_sigz_dqd(detuning=pulse_segment,
                                    tunnel_coupling=tunnel_coupling)

    correction = np.sum(
        exp_val_sigz * np.cos(e_zeeman * acc_time[0:n_time_steps_dc // 2]))

    # omit this step to calculate the correction factor
    # off_diag_element *= par_dict['gmubbx'] / 2

    # we multiply with gmubbx / 2 in our analytical formula and we divide by
    # the time step 2 * np.pi / e_zeeman / 2 / 50 of the discrete integration
    # then we multiply with 2 * e_zeeman / 2 / np.pi to calculate the average

    correction /= n_time_steps_dc // 2

    # We are correcting the predicted frequency of gmubbx / 2 so we need to
    # reverse the factor of 2 previously assumed.
    correction *= 2
    return correction


def physical_minimal_time_steps(pulse_mode, amp, e_zeeman, tunnel_coupling,
                                total_time_guess, tanh_range,
                                discretization_delta, **_):
    """
    Estimates the minimal time steps to resolve the time dynamics.

    Calculation described in [1].

    Parameters
    ----------
    pulse_mode: string
        Giving the pulse type. Supported are "duty_cycle" for the rectangular
        pulse and "sine" for a sine or cosine pulse.

    amp: float
        Pulse amplitude.

    e_zeeman: float
        Average Zeeman energy.

    tunnel_coupling: float
        Tunnel coupling of the double quantum dot.

    total_time_guess: float
        Estimation of the total pulse time.

    tanh_range: float
        Range parameter of the hyperbolic tangent. Determines the smoothness
        of the rectangular pulse.

    discretization_delta: float
        Threshold value for the accuracy.

    Returns
    -------
    n_time_steps: int
        Minimal number of time steps required for the pulse.

    """
    if pulse_mode == 'sine':
        pulse_time_derivative_bound = amp * e_zeeman
        omega = 2 * tunnel_coupling
    elif pulse_mode == 'duty_cycle':
        pulse_time_derivative_bound = amp * tanh_range / np.pi * e_zeeman
        # omega = ((2 * tunnel_coupling) ** 2 + amp ** 2) ** .5
        omega = 2 * tunnel_coupling
        # Modification Idea: Substitute the orbital splitting for the tunnel
        # coupling for a worst case estimation.
        # we use the orbital energy splitting omega instead of 2 *
        # tunnel_coupling,
        # because simulations indicate that epsilon also has an effect.
    else:
        raise ValueError('Unknown pulse mode.')

    n_time_steps_cond_1 = (
        total_time_guess ** 2 * pulse_time_derivative_bound
        / discretization_delta
    ) ** .5

    n_time_steps_cond_2 = (
        total_time_guess ** 3 * 2 * omega
        * pulse_time_derivative_bound
        / discretization_delta
    ) ** (1 / 3)
    n_time_steps = max(n_time_steps_cond_1, n_time_steps_cond_2)
    return int(np.ceil(n_time_steps))


def time_and_steps_guess(amp, tunnel_coupling, e_zeeman,
                         resonance_frequency_guess, rabi_frequency_guess,
                         rabi_rotation_angle, min_n_time_steps,
                         pulse_mode, time_correction_factor, tanh_range,
                         n_time_steps_dc, discretization_delta, **_):
    """
    Calculate the total time and the time steps of the chosen pulse.

    Parameters
    ----------
    amp: float
        Pulse amplitude.

    tunnel_coupling: float
        Tunnel coupling of the double quantum dot.

    e_zeeman: float
        Average Zeeman energy.

    resonance_frequency_guess: float
        Estimation for the resonance frequency.

    rabi_frequency_guess: float
        Estimation for the Rabi frequency.

    rabi_rotation_angle: float
        Polar angle of the optimal gate.

    min_n_time_steps: int
        Minimal number of time steps for the cosine pulse.

    pulse_mode: string
        Giving the pulse type. Supported are "duty_cycle" for the rectangular
        pulse and "sine" for a sine or cosine pulse.

    time_correction_factor: float
        Can be used to correct the total time guess for systematic deviations.

    tanh_range: float
        Range parameter of the hyperbolic tangent. Determines the smoothness
        of the rectangular pulse.

    n_time_steps_dc: int
        Number of time steps in every period of the rectangular pulse.

    discretization_delta: float
        Threshold value for the accuracy.


    Returns
    -------
    n_time_steps: int
        Number of time steps in the pulse.

    total_time_guess: float
        Estimation of the total pulse time.

    """
    total_time_guess = rabi_rotation_angle / rabi_frequency_guess  # * 2
    if pulse_mode == 'sine':
        n_time_steps = physical_minimal_time_steps(
            pulse_mode=pulse_mode,
            amp=amp,
            tunnel_coupling=tunnel_coupling,
            tanh_range=tanh_range,
            total_time_guess=total_time_guess,
            e_zeeman=e_zeeman,
            discretization_delta=discretization_delta
        )
        n_time_steps = max(
            n_time_steps,
            min_n_time_steps
        )

    elif pulse_mode == 'duty_cycle':
        t_resonance = 2 * np.pi / resonance_frequency_guess
        if time_correction_factor is not None:
            total_time_guess *= time_correction_factor
        n_resonance_periods = int(np.ceil(total_time_guess / t_resonance))
        total_time_guess = n_resonance_periods * t_resonance
        # so we need this resolution to resolve the applied pulse
        n_time_steps = int(n_time_steps_dc * n_resonance_periods)
    else:
        raise ValueError('Unknown pulse mode.')

    return n_time_steps, total_time_guess


class AmplitudeValueFunc:
    """ Implements the pulse shapes and the time stretching for the amplitude
    function.

    In addition to the actual driving pulse, also the other terms in the
    Hamiltonian are included because the time stretching parameter is included
    as optimization parameter and occurs in the calculation of these amplitudes
    in the Hamiltonian.

    Implemented as class because a nested function cannot be pickled, which is
    required for parallel execution.

    Parameters
    ----------
    n_time_steps: int
        Number of time steps in the pulse.

    acc_time: numpy array of float
        Accumulated gate times.

    total_time_guess: float
        Estimation of the total time.

    amp: float
        Pulse amplitude.

    tunnel_coupling: float
        Tunnel coupling of the double quantum dot.

    eps: float
        Detuning of the double quantum dot. This is the difference in chemical
        potential between the two dots.

    gmubbx: float
        Magnetic gradient across the double quantum dot in x direction.
        Multiplied with the g-factor and Bohr magneton.

    gmubbz: float
        Magnetic gradient across the double quantum dot in z direction.
        Multiplied with the g-factor and Bohr magneton.

    e_zeeman: float
        Average Zeeman energy.

    pulse_mode: string
        Giving the pulse type. Supported are "duty_cycle" for the rectangular
        pulse and "sine" for a sine or cosine pulse.

    total_time_guess: float
        Estimation of the total pulse time before we correct the Rabi
        frequency.

    phase_shift: float
        Angle for a phase shift of the applied pulse.

    n_time_steps: int
        Number of time steps in the time discretization.

    tanh_range: float
        Range parameter of the hyperbolic tangent. Determines the smoothness
        of the rectangular pulse.

    pulse_envelope: bool
        Set to True, if a pulse envelope is used.

    resonance_time_guess: float
        Estimation of the time period of the resonant pulse.

    """
    def __init__(
            self, n_time_steps, acc_time, total_time_guess, eps, amp, e_zeeman,
            tunnel_coupling, gmubbx, gmubbz, pulse_envelope, pulse_mode,
            resonance_time_guess, tanh_range, phase_shift, **_
    ):
        self.n_time_steps = n_time_steps
        self.acc_time = acc_time
        self.total_time_guess = total_time_guess
        self.eps = eps
        self.amp = amp
        self.e_zeeman = e_zeeman
        self.tunnel_coupling = tunnel_coupling
        self.gmubbx = gmubbx
        self.gmubbz = gmubbz
        self.pulse_envelope = pulse_envelope
        self.pulse_mode = pulse_mode
        self.resonance_time_guess = resonance_time_guess
        self.tanh_range = tanh_range
        self.phase_shift = phase_shift
        if self.pulse_mode == 'duty_cycle':
            # save auxiliary quantities for the pulse calculation.
            self.t_tanh = self.resonance_time_guess / 2
            tanh_limit = np.tanh(self.tanh_range)
            self.corrected_amp = self.amp / tanh_limit

            self.t_res = self.resonance_time_guess
            # We apply the phase shift:
            t = self.acc_time + self.phase_shift / (2 * np.pi) \
                * self.resonance_time_guess
            # and reset the time after each period
            self.t = t - self.t_res * (t // self.t_res)

    def __call__(self, opt_pars):
        """
        Creates the pulse shape and computes the other amplitudes in the
        Hamiltonian.

        Parameters
        ----------
            opt_pars: numpy array of float,
            shape (n_time_steps, 2) = num_t, num_par
            The two values are repeated along the time axis for convenience.
            For the cosine pulse, the optimization parameters are the resonance
            frequency and the time stretching factor. While the rectangular
            pulse receives the duty cycle parameter and the time stretching
            factor.

        Returns
        -------
            amplitudes: numpy array of shape (n_time_steps, 5)
            The amplitudes are along the second axis:
            - detuning
            - zeeman splitting
            - tunnel_coupling
            - gmubbx: Magnetic gradient along x direction
            - gmubbz: Magnetic gradient along z direction


        """
        amplitudes = np.empty(shape=[self.n_time_steps, 5], dtype=float)

        if self.pulse_mode == 'sine':

            # Compute the envelope function:
            if self.pulse_envelope:
                envelope = 2 * np.sin(
                    np.pi * self.acc_time / self.total_time_guess
                ) ** 2
            else:
                envelope = np.ones_like(self.acc_time)

            # Compute the resonant signal:
            amplitudes[:, 0] = (
                self.eps + self.amp * np.cos(
                    opt_pars[0, 0] * opt_pars[0, 1] * self.acc_time
                ) * envelope
            )

        elif self.pulse_mode == 'duty_cycle':
            pulse = np.empty(shape=[self.n_time_steps], dtype=float)

            c_duty_cycle = opt_pars[0, 0]

            # Rising half of the pulse
            temp_index = self.t < (self.t_res / 2)
            temp_tanh_argument = (self.tanh_range / self.t_tanh) \
                * (self.t[temp_index] - .5 * self.t_tanh * (1 + c_duty_cycle))
            pulse[temp_index] = (
                    self.corrected_amp * np.tanh(temp_tanh_argument)
            )

            # Falling part of the pulse
            temp_index = self.t >= (self.t_res / 2)
            temp_tanh_argument = (self.tanh_range / self.t_tanh) \
                * (self.t[temp_index] - .5 * self.t_tanh * (3 - c_duty_cycle))
            pulse[temp_index] = (
                    -1 * self.corrected_amp * np.tanh(temp_tanh_argument)
            )

            amplitudes[:, 0] = pulse + self.eps
        else:
            raise ValueError('Unknown pulse mode.')

        amplitudes[:, 1] = self.e_zeeman
        amplitudes[:, 2] = 2 * self.tunnel_coupling
        amplitudes[:, 3] = self.gmubbx
        amplitudes[:, 4] = self.gmubbz
        amplitudes = opt_pars[0, 1] * amplitudes

        if self.pulse_mode == 'duty_cycle':
            asymmetry_correction_factor = (
                1 - opt_pars[0, 0] * self.gmubbz / 2 / self.e_zeeman
            )
            amplitudes = asymmetry_correction_factor * amplitudes

        return amplitudes


class AmplitudeValueFuncValley:
    """ Implement the pulse shape when including the valley degree of freedom
     for the amplitude function.

    In addition to the actual driving pulse, also the other terms in the
    Hamiltonian are included because the time stretching parameter is included
    as optimization parameter and occurs in the calculation of these amplitudes
    in the Hamiltonian.

    Implemented as class because a nested function cannot be pickled, which is
    required for parallel execution.

    Parameters
    ----------
    n_time_steps: int
        Number of time steps in the pulse.

    acc_time: numpy array of float
        Accumulated gate times.

    total_time_guess: float
        Estimation of the total time.

    amp: float
        Pulse amplitude.

    tunnel_coupling: float
        Tunnel coupling of the double quantum dot.

    eps: float
        Detuning of the double quantum dot. This is the difference in chemical
        potential between the two dots.

    gmubbx: float
        Magnetic gradient across the double quantum dot in x direction.
        Multiplied with the g-factor and Bohr magneton.

    gmubbz: float
        Magnetic gradient across the double quantum dot in z direction.
        Multiplied with the g-factor and Bohr magneton.

    e_zeeman: float
        Average Zeeman energy.

    pulse_mode: string
        Giving the pulse type. Supported are "duty_cycle" for the rectangular
        pulse and "sine" for a sine or cosine pulse.

    total_time_guess: float
        Estimation of the total pulse time before we correct the Rabi
        frequency.

    phase_shift: float
        Angle for a phase shift of the applied pulse.

    n_time_steps: int
        Number of time steps in the time discretization.

    tanh_range: float
        Range parameter of the hyperbolic tangent. Determines the smoothness
        of the rectangular pulse.

    pulse_envelope: bool
        Set to True, if a pulse envelope is used.

    resonance_time_guess: float
        Estimation of the time period of the resonant pulse.

    real_valley_l: float
        Real part of the valley splitting in the left quantum dot.

    imag_valley_l: float
        Imaginary part of the valley splitting in the left quantum dot.

    real_valley_r: float
        Real part of the valley splitting in the right quantum dot.

    imag_valley_r: float
        Imaginary part of the valley splitting in the right quantum dot.

    """
    def __init__(
            self, n_time_steps, acc_time, total_time_guess, eps, amp, e_zeeman,
            tunnel_coupling, gmubbx, gmubbz, pulse_envelope, pulse_mode,
            resonance_time_guess, tanh_range, phase_shift, real_valley_l,
            imag_valley_l, real_valley_r, imag_valley_r
    ):
        self.n_time_steps = n_time_steps
        self.acc_time = acc_time
        self.total_time_guess = total_time_guess
        self.eps = eps
        self.amp = amp
        self.e_zeeman = e_zeeman
        self.tunnel_coupling = tunnel_coupling
        self.gmubbx = gmubbx
        self.gmubbz = gmubbz
        self.pulse_envelope = pulse_envelope
        self.pulse_mode = pulse_mode
        self.resonance_time_guess = resonance_time_guess
        self.tanh_range = tanh_range
        self.phase_shift = phase_shift

        self.real_valley_l = real_valley_l
        self.imag_valley_l = imag_valley_l
        self.real_valley_r = real_valley_r
        self.imag_valley_r = imag_valley_r

        if self.pulse_mode == 'duty_cycle':
            # save auxiliary quantities for the pulse calculation.
            self.t_tanh = self.resonance_time_guess / 2
            tanh_limit = np.tanh(self.tanh_range)
            self.corrected_amp = self.amp / tanh_limit

            self.t_res = self.resonance_time_guess
            # We apply the phase shift:
            t = self.acc_time + self.phase_shift / (2 * np.pi) \
                * self.resonance_time_guess
            # and reset the time after each period
            self.t = t - self.t_res * (t // self.t_res)

    def __call__(self, opt_pars):
        """
        Creates the pulse shape and computes the other amplitudes in the
        Hamiltonian.

        Parameters
        ----------
            opt_pars: numpy array of float,
            shape (n_time_steps, 2) = num_t, num_par
            The two values are repeated along the time axis for convenience.
            For the cosine pulse, the optimization parameters are the resonance
            frequency and the time stretching factor. While the rectangular
            pulse receives the duty cycle parameter and the time stretching
            factor.

        Returns
        -------
            amplitudes: numpy array of shape (n_time_steps, 9)
            The amplitudes are along the second axis:
            - detuning
            - zeeman splitting
            - tunnel_coupling
            - gmubbx: Magnetic gradient along x direction
            - gmubbz: Magnetic gradient along z direction
            - real_valley_l: Real part of the valley parameter in the left dot
            - imag_valley_l: Imaginary part of the valley parameter in the left
            dot
            - real_valley_r: Real part of the valley parameter in the right dot
            - imag_valley_r: Imaginary part of the valley parameter in the
            right dot

        """
        amplitudes = np.empty(shape=[self.n_time_steps, 9], dtype=float)

        if self.pulse_mode == 'sine':

            # Compute the envelope function:
            if self.pulse_envelope:
                envelope = 2 * np.sin(
                    np.pi * self.acc_time / self.total_time_guess
                ) ** 2
            else:
                envelope = np.ones_like(self.acc_time)

            # Compute the resonant signal:
            amplitudes[:, 0] = (
                self.eps + self.amp * np.cos(
                    opt_pars[0, 0] * opt_pars[0, 1] * self.acc_time
                ) * envelope
            )

        elif self.pulse_mode == 'duty_cycle':
            pulse = np.empty(shape=[self.n_time_steps], dtype=float)

            c_duty_cycle = opt_pars[0, 0]

            # Rising half of the pulse
            temp_index = self.t < (self.t_res / 2)
            temp_tanh_argument = (self.tanh_range / self.t_tanh) \
                * (self.t[temp_index] - .5 * self.t_tanh * (1 + c_duty_cycle))
            pulse[temp_index] = (
                    self.corrected_amp * np.tanh(temp_tanh_argument)
            )

            # Falling part of the pulse
            temp_index = self.t >= (self.t_res / 2)
            temp_tanh_argument = (self.tanh_range / self.t_tanh) \
                * (self.t[temp_index] - .5 * self.t_tanh * (3 - c_duty_cycle))
            pulse[temp_index] = (
                    -1 * self.corrected_amp * np.tanh(temp_tanh_argument)
            )

            amplitudes[:, 0] = pulse + self.eps
        else:
            raise ValueError('Unknown pulse mode.')

        amplitudes[:, 1] = self.e_zeeman
        amplitudes[:, 2] = 2 * self.tunnel_coupling
        amplitudes[:, 3] = self.gmubbx
        amplitudes[:, 4] = self.gmubbz
        amplitudes[:, 5] = self.real_valley_l
        amplitudes[:, 6] = self.imag_valley_l
        amplitudes[:, 7] = self.real_valley_r
        amplitudes[:, 8] = self.imag_valley_r

        amplitudes = opt_pars[0, 1] * amplitudes

        if self.pulse_mode == 'duty_cycle':
            asymmetry_correction_factor = (
                1 - opt_pars[0, 0] * self.gmubbz / 2 / self.e_zeeman
            )
            amplitudes = asymmetry_correction_factor * amplitudes

        return amplitudes


class AmplitudeDerivativeFunc:
    """
    Derivative of the pulse shapes for the amplitude function.

    Implemented as class because a nested function cannot be pickled, which is
    required for parallel execution.

    For a description of the input parameters see 'AmplitudeValueFunc'.

    """
    def __init__(
            self, n_time_steps, acc_time, total_time_guess, eps, amp, e_zeeman,
            tunnel_coupling, gmubbx, gmubbz, pulse_envelope, pulse_mode,
            resonance_time_guess, tanh_range, phase_shift
    ):
        self.n_time_steps = n_time_steps
        self.acc_time = acc_time
        self.total_time_guess = total_time_guess
        self.eps = eps
        self.amp = amp
        self.e_zeeman = e_zeeman
        self.tunnel_coupling = tunnel_coupling
        self.gmubbx = gmubbx
        self.gmubbz = gmubbz
        self.pulse_envelope = pulse_envelope
        self.pulse_mode = pulse_mode
        self.resonance_time_guess = resonance_time_guess
        self.tanh_range = tanh_range
        self.phase_shift = phase_shift
        if self.pulse_mode == 'duty_cycle':
            # save auxiliary quantities for the pulse calculation.
            self.t_tanh = self.resonance_time_guess / 2
            tanh_limit = np.tanh(self.tanh_range)
            self.corrected_amp = self.amp / tanh_limit

            self.t_res = self.resonance_time_guess
            # We apply the phase shift:
            t = self.acc_time \
                + self.phase_shift / (2 * np.pi) * self.resonance_time_guess
            # and reset the time after each period
            self.t = t - self.t_res * (t // self.t_res)

    def __call__(self, opt_pars):
        """
        Calculate the derivatives of the amplitudes in the Hamiltonian.

        Parameters
        ----------
            opt_pars: numpy array of float,
            shape (n_time_steps, 2) = num_t, num_par
            The two values are repeated along the time axis for convenience.
            For the cosine pulse, the optimization parameters are the resonance
            frequency and the time stretching factor. While the rectangular
            pulse receives the duty cycle parameter and the time stretching
            factor.

        Returns
        -------
            derivatives: numpy array of shape (n_time_steps, 2, 5)
            The derivatives are along the third axis:
            - detuning
            - zeeman splitting
            - tunnel_coupling
            - gmubbx: Magnetic gradient along x direction
            - gmubbz: Magnetic gradient along z direction
            calculated with respect to the entry on the second axis, being
            the received parameters.

        """

        derivatives = np.zeros(shape=[self.n_time_steps, 2, 5], dtype=float)
        if self.pulse_mode == 'sine':
            # omega = opt_pars[0, 0]
            # s_T = opt_pars[0, 1]

            if self.pulse_envelope:
                envelope = 2 * np.sin(
                    np.pi * self.acc_time / self.total_time_guess
                ) ** 2
            else:
                envelope = np.ones_like(self.acc_time)

            derivatives[:, 0, 0] = (
                    -1 * opt_pars[0, 1] * opt_pars[0, 1]
                    * self.amp * self.acc_time * np.sin(
                        opt_pars[0, 0] * opt_pars[0, 1] * self.acc_time)
                    * envelope
            )

            derivatives[:, 1, 0] = (
                self.eps
                + self.amp * np.cos(
                    opt_pars[0, 0] * opt_pars[0, 1] * self.acc_time) * envelope
                - opt_pars[0, 1] * opt_pars[0, 0] * self.acc_time * self.amp
                * np.sin(opt_pars[0, 0] * opt_pars[0, 1] * self.acc_time)
                * envelope
            )
            derivatives[:, 1, 1] = self.e_zeeman
            derivatives[:, 1, 2] = 2 * self.tunnel_coupling
            derivatives[:, 1, 3] = self.gmubbx
            derivatives[:, 1, 4] = self.gmubbz

        elif self.pulse_mode == 'duty_cycle':
            # c_duty_cycle = opt_pars[0, 0]
            # s_T = opt_pars[0, 1]

            deriv_s_t = np.empty(shape=[self.n_time_steps], dtype=float)

            c_duty_cycle = opt_pars[0, 0]

            asymmetry_correction_factor = (
                1 - c_duty_cycle * self.gmubbz / 2 / self.e_zeeman
            )
            deriv_asymmetry_correction_factor = (
                    - self.gmubbz / 2 / self.e_zeeman
            )

            temp_index_1 = self.t < (self.t_res / 2)
            tanh_argument_1 = (self.tanh_range / self.t_tanh) \
                * (self.t[temp_index_1]
                   - .5 * self.t_tanh * (1 + c_duty_cycle))
            deriv_s_t[temp_index_1] = (
                    asymmetry_correction_factor * self.corrected_amp *
                    np.tanh(tanh_argument_1)
            )

            temp_index_2 = self.t >= (self.t_res / 2)
            tanh_argument_2 = (self.tanh_range / self.t_tanh) \
                * (self.t[temp_index_2]
                   - .5 * self.t_tanh * (3 - c_duty_cycle))
            deriv_s_t[temp_index_2] = (
                    asymmetry_correction_factor *
                    -1 * self.corrected_amp * np.tanh(tanh_argument_2)
            )

            # Include the derivative due to the constant offset.
            # Note that it is time independent.
            deriv_s_t += self.eps * asymmetry_correction_factor

            derivatives[:, 1, 0] = deriv_s_t

            deriv_c_dc = np.empty(shape=[self.n_time_steps], dtype=float)

            deriv_tanh_argument = .5 * self.tanh_range
            # derivative of the tanh argument by the duty cycle parameter

            # The factor of self.t_tanh / 2 in the denominator accounts for the
            # rescaling of the duty cycle parameter.
            deriv_c_dc[temp_index_1] = (
               asymmetry_correction_factor *
               -2 * self.corrected_amp * opt_pars[0, 1] * deriv_tanh_argument
            ) / (np.cosh(2 * tanh_argument_1) + 1)

            deriv_c_dc[temp_index_1] += (
                    deriv_asymmetry_correction_factor
                    * self.corrected_amp * opt_pars[0, 1]
                    * np.tanh(tanh_argument_1)
            )

            deriv_c_dc[temp_index_2] = (
               asymmetry_correction_factor *
               -2 * self.corrected_amp * opt_pars[0, 1] * deriv_tanh_argument
            ) / (np.cosh(2 * tanh_argument_2) + 1)

            deriv_c_dc[temp_index_2] += (
                    -deriv_asymmetry_correction_factor
                    * self.corrected_amp * opt_pars[0, 1]
                    * np.tanh(tanh_argument_2)
            )

            # Include the derivative due to the constant offset.
            # Note that it is time independent.
            deriv_c_dc += deriv_asymmetry_correction_factor \
                * self.eps * opt_pars[0, 1]

            derivatives[:, 0, 0] = deriv_c_dc

            # other derivatives by the time stretching parameter
            derivatives[:, 1, 1] = self.e_zeeman * asymmetry_correction_factor
            derivatives[:, 1, 2] = 2 * self.tunnel_coupling * \
                asymmetry_correction_factor
            derivatives[:, 1, 3] = self.gmubbx * asymmetry_correction_factor
            derivatives[:, 1, 4] = self.gmubbz * asymmetry_correction_factor

            # derivatives by the duty cycle parameter due to the asymmetry
            # correction
            derivatives[:, 0, 1] = (
                self.e_zeeman * deriv_asymmetry_correction_factor
                * opt_pars[0, 1]
            )
            derivatives[:, 0, 2] = (
                2 * self.tunnel_coupling * deriv_asymmetry_correction_factor
                * opt_pars[0, 1]
            )
            derivatives[:, 0, 3] = (
                self.gmubbx * deriv_asymmetry_correction_factor
                * opt_pars[0, 1]
            )
            derivatives[:, 0, 4] = (
                self.gmubbz * deriv_asymmetry_correction_factor
                * opt_pars[0, 1]
            )

        else:
            raise ValueError('Unknown pulse mode.')

        return derivatives


class AmplitudeDerivativeFuncValley:
    """
    Derivative of the pulse shapes including the valley degree of freedom for
    the amplitude function.

    Implemented as class because a nested function cannot be pickled, which is
    required for parallel execution.

    For a description of the input parameters see 'AmplitudeValueFuncValley'.

    """
    def __init__(
            self, n_time_steps, acc_time, total_time_guess, eps, amp, e_zeeman,
            tunnel_coupling, gmubbx, gmubbz, pulse_envelope, pulse_mode,
            resonance_time_guess, tanh_range, phase_shift, real_valley_l,
            imag_valley_l, real_valley_r, imag_valley_r
    ):
        self.n_time_steps = n_time_steps
        self.acc_time = acc_time
        self.total_time_guess = total_time_guess
        self.eps = eps
        self.amp = amp
        self.e_zeeman = e_zeeman
        self.tunnel_coupling = tunnel_coupling
        self.gmubbx = gmubbx
        self.gmubbz = gmubbz
        self.pulse_envelope = pulse_envelope
        self.pulse_mode = pulse_mode
        self.resonance_time_guess = resonance_time_guess
        self.tanh_range = tanh_range
        self.phase_shift = phase_shift

        self.real_valley_l = real_valley_l
        self.imag_valley_l = imag_valley_l
        self.real_valley_r = real_valley_r
        self.imag_valley_r = imag_valley_r

        if self.pulse_mode == 'duty_cycle':
            # save auxiliary quantities for the pulse calculation.
            self.t_tanh = self.resonance_time_guess / 2
            tanh_limit = np.tanh(self.tanh_range)
            self.corrected_amp = self.amp / tanh_limit

            self.t_res = self.resonance_time_guess
            # We apply the phase shift:
            t = self.acc_time \
                + self.phase_shift / (2 * np.pi) * self.resonance_time_guess
            # and reset the time after each period
            self.t = t - self.t_res * (t // self.t_res)

    def __call__(self, opt_pars):
        """
        Calculate the derivatives of the amplitudes in the Hamiltonian.

        Parameters
        ----------
            opt_pars: numpy array of float,
            shape (n_time_steps, 2) = num_t, num_par
            The two values are repeated along the time axis for convenience.
            For the cosine pulse, the optimization parameters are the resonance
            frequency and the time stretching factor. While the rectangular
            pulse receives the duty cycle parameter and the time stretching
            factor.

        Returns
        -------
            amplitudes: numpy array of shape (n_time_steps, 9)
            The derivatives are along the third axis:
            - detuning
            - zeeman splitting
            - tunnel_coupling
            - gmubbx: Magnetic gradient along x direction
            - gmubbz: Magnetic gradient along z direction
            - real_valley_l: Real part of the valley parameter in the left dot
            - imag_valley_l: Imaginary part of the valley parameter in the left
            dot
            - real_valley_r: Real part of the valley parameter in the right dot
            - imag_valley_r: Imaginary part of the valley parameter in the
            right dot
            calculated with respect to the entry on the second axis, being
            the received parameters.

        """

        derivatives = np.zeros(shape=[self.n_time_steps, 2, 9], dtype=float)
        if self.pulse_mode == 'sine':
            # omega = opt_pars[0, 0]
            # s_T = opt_pars[0, 1]

            if self.pulse_envelope:
                envelope = 2 * np.sin(
                    np.pi * self.acc_time / self.total_time_guess
                ) ** 2
            else:
                envelope = np.ones_like(self.acc_time)

            derivatives[:, 0, 0] = (
                    -1 * opt_pars[0, 1] * opt_pars[0, 1]
                    * self.amp * self.acc_time * np.sin(
                        opt_pars[0, 0] * opt_pars[0, 1] * self.acc_time)
                    * envelope
            )

            derivatives[:, 1, 0] = (
                self.eps
                + self.amp * np.cos(
                    opt_pars[0, 0] * opt_pars[0, 1] * self.acc_time) * envelope
                - opt_pars[0, 1] * opt_pars[0, 0] * self.acc_time * self.amp
                * np.sin(opt_pars[0, 0] * opt_pars[0, 1] * self.acc_time)
                * envelope
            )
            derivatives[:, 1, 1] = self.e_zeeman
            derivatives[:, 1, 2] = 2 * self.tunnel_coupling
            derivatives[:, 1, 3] = self.gmubbx
            derivatives[:, 1, 4] = self.gmubbz
            derivatives[:, 1, 5] = self.real_valley_l
            derivatives[:, 1, 6] = self.imag_valley_l
            derivatives[:, 1, 7] = self.real_valley_r
            derivatives[:, 1, 8] = self.imag_valley_r

        elif self.pulse_mode == 'duty_cycle':
            # c_duty_cycle = opt_pars[0, 0]
            # s_T = opt_pars[0, 1]

            deriv_s_t = np.empty(shape=[self.n_time_steps], dtype=float)

            c_duty_cycle = opt_pars[0, 0]

            asymmetry_correction_factor = (
                1 - c_duty_cycle * self.gmubbz / 2 / self.e_zeeman
            )
            deriv_asymmetry_correction_factor = (
                    - self.gmubbz / 2 / self.e_zeeman
            )

            temp_index_1 = self.t < (self.t_res / 2)
            tanh_argument_1 = (self.tanh_range / self.t_tanh) \
                * (self.t[temp_index_1]
                   - .5 * self.t_tanh * (1 + c_duty_cycle))
            deriv_s_t[temp_index_1] = (
                    asymmetry_correction_factor * self.corrected_amp *
                    np.tanh(tanh_argument_1)
            )

            temp_index_2 = self.t >= (self.t_res / 2)
            tanh_argument_2 = (self.tanh_range / self.t_tanh) \
                * (self.t[temp_index_2]
                   - .5 * self.t_tanh * (3 - c_duty_cycle))
            deriv_s_t[temp_index_2] = (
                    asymmetry_correction_factor *
                    -1 * self.corrected_amp * np.tanh(tanh_argument_2)
            )

            # Include the derivative due to the constant offset.
            # Note that it is time independent.
            deriv_s_t += self.eps * asymmetry_correction_factor

            derivatives[:, 1, 0] = deriv_s_t

            deriv_c_dc = np.empty(shape=[self.n_time_steps], dtype=float)

            deriv_tanh_argument = .5 * self.tanh_range
            # derivative of the tanh argument by the duty cycle parameter

            # The factor of self.t_tanh / 2 in the denominator accounts for the
            # rescaling of the duty cycle parameter.
            deriv_c_dc[temp_index_1] = (
               asymmetry_correction_factor *
               -2 * self.corrected_amp * opt_pars[0, 1] * deriv_tanh_argument
            ) / (np.cosh(2 * tanh_argument_1) + 1)

            deriv_c_dc[temp_index_1] += (
                    deriv_asymmetry_correction_factor
                    * self.corrected_amp * opt_pars[0, 1]
                    * np.tanh(tanh_argument_1)
            )

            deriv_c_dc[temp_index_2] = (
               asymmetry_correction_factor *
               -2 * self.corrected_amp * opt_pars[0, 1] * deriv_tanh_argument
            ) / (np.cosh(2 * tanh_argument_2) + 1)

            deriv_c_dc[temp_index_2] += (
                    -deriv_asymmetry_correction_factor
                    * self.corrected_amp * opt_pars[0, 1]
                    * np.tanh(tanh_argument_2)
            )

            # Include the derivative due to the constant offset.
            # Note that it is time independent.
            deriv_c_dc += deriv_asymmetry_correction_factor \
                * self.eps * opt_pars[0, 1]

            derivatives[:, 0, 0] = deriv_c_dc

            # other derivatives by the time stretching parameter
            derivatives[:, 1, 1] = self.e_zeeman * asymmetry_correction_factor
            derivatives[:, 1, 2] = 2 * self.tunnel_coupling * \
                asymmetry_correction_factor
            derivatives[:, 1, 3] = self.gmubbx * asymmetry_correction_factor
            derivatives[:, 1, 4] = self.gmubbz * asymmetry_correction_factor
            derivatives[:, 1, 5] = self.real_valley_l \
                * asymmetry_correction_factor
            derivatives[:, 1, 6] = self.imag_valley_l \
                * asymmetry_correction_factor
            derivatives[:, 1, 7] = self.real_valley_r \
                * asymmetry_correction_factor
            derivatives[:, 1, 8] = self.imag_valley_r \
                * asymmetry_correction_factor

            # derivatives by the duty cycle parameter due to the asymmetry
            # correction
            derivatives[:, 0, 1] = (
                self.e_zeeman * deriv_asymmetry_correction_factor
                * opt_pars[0, 1]
            )
            derivatives[:, 0, 2] = (
                2 * self.tunnel_coupling * deriv_asymmetry_correction_factor
                * opt_pars[0, 1]
            )
            derivatives[:, 0, 3] = (
                self.gmubbx * deriv_asymmetry_correction_factor
                * opt_pars[0, 1]
            )
            derivatives[:, 0, 4] = (
                self.gmubbz * deriv_asymmetry_correction_factor
                * opt_pars[0, 1]
            )
            derivatives[:, 0, 5] = (
                self.real_valley_l * deriv_asymmetry_correction_factor
                * opt_pars[0, 1]
            )
            derivatives[:, 0, 6] = (
                self.imag_valley_l * deriv_asymmetry_correction_factor
                * opt_pars[0, 1]
            )
            derivatives[:, 0, 7] = (
                self.real_valley_r * deriv_asymmetry_correction_factor
                * opt_pars[0, 1]
            )
            derivatives[:, 0, 8] = (
                self.imag_valley_r * deriv_asymmetry_correction_factor
                * opt_pars[0, 1]
            )

        else:
            raise ValueError('Unknown pulse mode.')

        return derivatives


def calculate_opt_azimuth_angle(total_propagator, rabi_rotation_angle,
                                theta_0=.0 * np.pi, **_):
    """
    Determine the azimuth angle rotation of an operator.

    I calculate the azimuth angle by comparing the given operator to a
    reference operator. As boundary condition, I require the reference operator
    to realize a rotation of a given polar angle. I then minimize the
    entanglement fidelity between the given operator and the reference operator
    by changing the azimuth angle.

    Parameters
    ----------
    total_propagator: DenseOperator
        Operator as unitary.

    rabi_rotation_angle: float
        The polar angle of the reference operator.

    theta_0: float
        Initial value of the azimuth angle in the optimization. Defaults to 0.

    Returns
    -------
    azimuth_angle: float
        The azimuth angle of the operator.

    """
    def azimuth_angle_infid(phi):
        return 1 - entanglement_fidelity(
            target=target_gate(phi[0], rabi_rotation_angle),
            propagator=total_propagator,
            computational_states=[0, 1],
            map_to_closest_unitary=True
        )
    result = minimize(azimuth_angle_infid, x0=np.asarray(
        theta_0), bounds=[[-np.pi, np.pi]])

    return result['x'][0]


def calculate_opt_azimuth_angle_valley(
        total_propagator_valley, rabi_rotation_angle, tunnel_coupling, eps,
        gmubbx, gmubbz, e_zeeman, real_valley_l, imag_valley_l, real_valley_r,
        imag_valley_r, amp, pulse_mode, theta_0=.0 * np.pi, **_):
    """
    Determine the azimuth angle rotation of an operator including the valley
    state.

    I calculate the azimuth angle by comparing the given operator to a
    reference operator. As boundary condition, I require the reference operator
    to realize a rotation of a given polar angle. I then minimize the
    entanglement fidelity between the given operator and the reference operator
    by changing the azimuth angle.

    Parameters
    ----------
    total_propagator: DenseOperator
        Operator as unitary.

    rabi_rotation_angle: float
        The polar angle of the reference operator.

    theta_0: float
        Initial value of the azimuth angle in the optimization. Defaults to 0.

    Returns
    -------
    azimuth_angle: float
        The azimuth angle of the operator.

    """

    computational_states = find_computational_states(
        tunnel_coupling=tunnel_coupling,
        eps=eps,
        gmubbx=gmubbx,
        gmubbz=gmubbz,
        e_zeeman=e_zeeman,
        real_valley_l=real_valley_l,
        imag_valley_l=imag_valley_l,
        real_valley_r=real_valley_r,
        imag_valley_r=imag_valley_r,
        amp=amp,
        pulse_mode=pulse_mode
    )

    def azimuth_angle_infid(phi):
        return 1 - entanglement_fidelity(
            target=target_gate(phi[0], rabi_rotation_angle),
            propagator=total_propagator_valley,
            computational_states=computational_states,
            map_to_closest_unitary=True
        )
    result = minimize(azimuth_angle_infid, x0=np.asarray(
        theta_0), bounds=[[-np.pi, np.pi]])

    return result['x'][0]


def target_gate(phi, rabi_rotation_angle):
    """
    Calculate the target gate as unitary operator given the rotation angles on
    the Bloch sphere.

    Parameters
    ----------
    phi: float
        Azimuth angle.

    rabi_rotation_angle: float
        Polar angle.

    Returns
    -------
    unitary_gate: DenseOperator
        Target gate as unitary operator.

    """
    target = (
        .5 * np.sin(phi) * DenseOperator.pauli_y()
        + .5 * np.cos(phi) * DenseOperator.pauli_x()
    ).exp(1j * rabi_rotation_angle)
    return target


def eigenbasis(tunnel_coupling, eps, gmubbx, gmubbz, e_zeeman, pulse_mode,
               amp, **_):
    """
    Calculates a transformation into the eigenbasis of the electron in
    resting position.

    Parameters
    ----------
    amp: float
        Pulse amplitude.

    tunnel_coupling: float
        Tunnel coupling of the double quantum dot.

    eps: float
        Detuning of the double quantum dot. This is the difference in chemical
        potential between the two dots.

    gmubbx: float
        Magnetic gradient across the double quantum dot in x direction.
        Multiplied with the g-factor and Bohr magneton.

    gmubbz: float
        Magnetic gradient across the double quantum dot in z direction.
        Multiplied with the g-factor and Bohr magneton.

    e_zeeman: float
        Average Zeeman energy.

    pulse_mode: string
        Giving the pulse type. Supported are "duty_cycle" for the rectangular
        pulse and "sine" for a sine or cosine pulse.

    Returns
    -------
    transformation: DenseOperator
        Basis transformation to the eigenbasis at the beginning and end of
        the pulse.

    """
    if pulse_mode == 'sine':
        h_drift = create_hamiltonian(
            tunnel_coupling=tunnel_coupling, eps=eps, gmubbx=gmubbx,
            gmubbz=gmubbz, e_zeeman=e_zeeman)
    elif pulse_mode == 'duty_cycle':
        h_drift = create_hamiltonian(
            tunnel_coupling=tunnel_coupling, eps=eps-amp, gmubbx=gmubbx,
            gmubbz=gmubbz, e_zeeman=e_zeeman)
    else:
        raise ValueError('Unknown pulse mode.')

    eig_vals, eig_vecs = h_drift.spectral_decomposition(hermitian=True)
    return DenseOperator(eig_vecs)


# ############################ Create Instances of qopt Solvers ###############


def create_coherent_solvers(
        n_time_steps, total_time_guess, eps, tunnel_coupling, gmubbx, gmubbz,
        e_zeeman, amp, pulse_envelope,
        pulse_mode, resonance_frequency_guess, tanh_range, phase_shift, **_):
    """
    Build the instances of Solver for the coherent simulations.

    Helper function assembling the instances of the qopt Solver. There is one
    for with explicit initial state set and one without.

    Returns
    -------
    solver_state: SchroedingerSolver
    This solver propagates the initial state forward in time and can be used to
    calculate a state fidelity.

    solver_unitary: SchroedingerSolver
    This solver calculates the general total propagator and can be used to
    calculate gate fidelities.

    """

    resonance_time_guess = 2 * np.pi / resonance_frequency_guess
    time = (total_time_guess / n_time_steps) * np.ones(n_time_steps)
    acc_time = np.cumsum(time)

    value_func = AmplitudeValueFunc(
        n_time_steps=n_time_steps, acc_time=acc_time, eps=eps,
        total_time_guess=total_time_guess, amp=amp, e_zeeman=e_zeeman,
        gmubbz=gmubbz, gmubbx=gmubbx, tunnel_coupling=tunnel_coupling,
        pulse_envelope=pulse_envelope, pulse_mode=pulse_mode,
        resonance_time_guess=resonance_time_guess, tanh_range=tanh_range,
        phase_shift=phase_shift
    )

    derivative_func = AmplitudeDerivativeFunc(
        n_time_steps=n_time_steps, acc_time=acc_time, eps=eps,
        total_time_guess=total_time_guess, amp=amp, e_zeeman=e_zeeman,
        gmubbz=gmubbz, gmubbx=gmubbx, tunnel_coupling=tunnel_coupling,
        pulse_envelope=pulse_envelope, pulse_mode=pulse_mode,
        resonance_time_guess=resonance_time_guess, tanh_range=tanh_range,
        phase_shift=phase_shift
    )

    transfer_func = OversamplingTF(num_ctrls=2, oversampling=n_time_steps)

    amp_func = CustomAmpFunc(
        value_function=value_func,
        derivative_function=derivative_func
    )

    if pulse_mode == 'sine':
        h_drift = create_hamiltonian(
            tunnel_coupling=tunnel_coupling, eps=eps, gmubbx=gmubbx,
            gmubbz=gmubbz, e_zeeman=e_zeeman)
    elif pulse_mode == 'duty_cycle':
        h_drift = create_hamiltonian(
            tunnel_coupling=tunnel_coupling, eps=eps-amp, gmubbx=gmubbx,
            gmubbz=gmubbz, e_zeeman=e_zeeman)
    else:
        raise ValueError('Unknown pulse mode.')

    eig_vals, eig_vecs = h_drift.spectral_decomposition(hermitian=True)
    eigenbasis = DenseOperator(eig_vecs)

    h_ctrl_transformed = [eigenbasis.dag() * m * eigenbasis for m in h_ctrl]

    solver_state = SchroedingerSolver(
        h_ctrl=h_ctrl_transformed,
        h_drift=[0 * tau_z],
        tau=total_time_guess * np.ones(1),
        amplitude_function=amp_func,
        initial_state=DenseOperator(np.asarray([[1], [0], [0], [0]])),
        transfer_function=transfer_func
    )

    solver_unitary = SchroedingerSolver(
        h_ctrl=h_ctrl_transformed,
        h_drift=[0 * tau_z],
        tau=total_time_guess * np.ones(1),
        amplitude_function=amp_func,
        transfer_function=transfer_func
    )

    return solver_state, solver_unitary


def create_coherent_solvers_valley(
        n_time_steps, total_time_guess, eps, tunnel_coupling, gmubbx, gmubbz,
        e_zeeman, amp, pulse_envelope, pulse_mode, resonance_frequency_guess,
        tanh_range, phase_shift, real_valley_l, imag_valley_l, real_valley_r,
        imag_valley_r, **_):
    """
    Build the instances of Solver for the coherent simulations.

    Helper function assembling the instances of the qopt Solver. There is one
    for with explicit initial state set and one without.

    Returns
    -------
    solver_state: SchroedingerSolver
    This solver propagates the initial state forward in time and can be used to
    calculate a state fidelity.

    solver_unitary: SchroedingerSolver
    This solver calculates the general total propagator and can be used to
    calculate gate fidelities.

    """

    resonance_time_guess = 2 * np.pi / resonance_frequency_guess
    time = (total_time_guess / n_time_steps) * np.ones(n_time_steps)
    acc_time = np.cumsum(time)

    value_func = AmplitudeValueFuncValley(
        n_time_steps=n_time_steps, acc_time=acc_time, eps=eps,
        total_time_guess=total_time_guess, amp=amp, e_zeeman=e_zeeman,
        gmubbz=gmubbz, gmubbx=gmubbx, tunnel_coupling=tunnel_coupling,
        pulse_envelope=pulse_envelope, pulse_mode=pulse_mode,
        resonance_time_guess=resonance_time_guess, tanh_range=tanh_range,
        phase_shift=phase_shift, real_valley_l=real_valley_l,
        imag_valley_l=imag_valley_l, real_valley_r=real_valley_r,
        imag_valley_r=imag_valley_r
    )

    derivative_func = AmplitudeDerivativeFuncValley(
        n_time_steps=n_time_steps, acc_time=acc_time, eps=eps,
        total_time_guess=total_time_guess, amp=amp, e_zeeman=e_zeeman,
        gmubbz=gmubbz, gmubbx=gmubbx, tunnel_coupling=tunnel_coupling,
        pulse_envelope=pulse_envelope, pulse_mode=pulse_mode,
        resonance_time_guess=resonance_time_guess, tanh_range=tanh_range,
        phase_shift=phase_shift, real_valley_l=real_valley_l,
        imag_valley_l=imag_valley_l, real_valley_r=real_valley_r,
        imag_valley_r=imag_valley_r
    )

    transfer_func = OversamplingTF(num_ctrls=2, oversampling=n_time_steps)

    amp_func = CustomAmpFunc(
        value_function=value_func,
        derivative_function=derivative_func
    )

    if pulse_mode == 'sine':
        h_drift = create_hamiltonian_valley(
            tunnel_coupling=tunnel_coupling, eps=eps, gmubbx=gmubbx,
            gmubbz=gmubbz, e_zeeman=e_zeeman, real_valley_l=real_valley_l,
            imag_valley_l=imag_valley_l, real_valley_r=real_valley_r,
            imag_valley_r=imag_valley_r)
    elif pulse_mode == 'duty_cycle':
        h_drift = create_hamiltonian_valley(
            tunnel_coupling=tunnel_coupling, eps=eps-amp, gmubbx=gmubbx,
            gmubbz=gmubbz, e_zeeman=e_zeeman, real_valley_l=real_valley_l,
            imag_valley_l=imag_valley_l, real_valley_r=real_valley_r,
            imag_valley_r=imag_valley_r)
    else:
        raise ValueError('Unknown pulse mode.')

    eig_vals, eig_vecs = h_drift.spectral_decomposition(hermitian=True)
    eigenbasis = DenseOperator(eig_vecs)

    h_ctrl_transformed = [eigenbasis.dag() * m * eigenbasis
                          for m in h_ctrl_valley]

    solver_state = SchroedingerSolver(
        h_ctrl=h_ctrl_transformed,
        h_drift=[0 * tau_z_v],
        tau=total_time_guess * np.ones(1),
        amplitude_function=amp_func,
        initial_state=DenseOperator(np.asarray(
            [[1], [0], [0], [0], [0], [0], [0], [0]])),
        transfer_function=transfer_func
    )

    solver_unitary = SchroedingerSolver(
        h_ctrl=h_ctrl_transformed,
        h_drift=[0 * tau_z_v],
        tau=total_time_guess * np.ones(1),
        amplitude_function=amp_func,
        transfer_function=transfer_func
    )

    return solver_state, solver_unitary


def create_mc_solvers(
        n_time_steps, total_time_guess, eps, tunnel_coupling, gmubbx, gmubbz,
        e_zeeman, amp, pulse_envelope, pulse_mode, resonance_frequency_guess,
        tanh_range, phase_shift, eps_noise_std, eps_res_noise_std, n_traces,
        **_):
    """
    Build the instances of SchroedingerSMonteCarlo for the Monte Carlo
    simulation of quasistatic noise.

    Helper function assembling the instances of the qopt
    SchroedingerSMonteCarlo. This function create one solver for electric noise
    on the detuning and one for magnetic noise on the total magnetic field.

    Returns
    -------
    solver_chrg_noise: SchroedingerSMonteCarlo
    This solver simulates quasistatic electric noise on the detuning.

    solver_res_noise: SchroedingerSMonteCarlo
    This solver simulates magnetic noise on the total magnetic field.

    """

    resonance_time_guess = 2 * np.pi / resonance_frequency_guess
    time = (total_time_guess / n_time_steps) * np.ones(n_time_steps)
    acc_time = np.cumsum(time)

    value_func = AmplitudeValueFunc(
        n_time_steps=n_time_steps, acc_time=acc_time, eps=eps,
        total_time_guess=total_time_guess, amp=amp, e_zeeman=e_zeeman,
        gmubbz=gmubbz, gmubbx=gmubbx, tunnel_coupling=tunnel_coupling,
        pulse_envelope=pulse_envelope, pulse_mode=pulse_mode,
        resonance_time_guess=resonance_time_guess, tanh_range=tanh_range,
        phase_shift=phase_shift
    )

    derivative_func = AmplitudeDerivativeFunc(
        n_time_steps=n_time_steps, acc_time=acc_time, eps=eps,
        total_time_guess=total_time_guess, amp=amp, e_zeeman=e_zeeman,
        gmubbz=gmubbz, gmubbx=gmubbx, tunnel_coupling=tunnel_coupling,
        pulse_envelope=pulse_envelope, pulse_mode=pulse_mode,
        resonance_time_guess=resonance_time_guess, tanh_range=tanh_range,
        phase_shift=phase_shift
    )

    transfer_func = OversamplingTF(num_ctrls=2, oversampling=n_time_steps)

    amp_func = CustomAmpFunc(
        value_function=value_func,
        derivative_function=derivative_func
    )

    if pulse_mode == 'sine':
        h_drift = create_hamiltonian(
            tunnel_coupling=tunnel_coupling, eps=eps, gmubbx=gmubbx,
            gmubbz=gmubbz, e_zeeman=e_zeeman)
    elif pulse_mode == 'duty_cycle':
        h_drift = create_hamiltonian(
            tunnel_coupling=tunnel_coupling, eps=eps-amp, gmubbx=gmubbx,
            gmubbz=gmubbz, e_zeeman=e_zeeman)
    else:
        raise ValueError('Unknown pulse mode.')

    eig_vals, eig_vecs = h_drift.spectral_decomposition(hermitian=True)
    eigenbasis = DenseOperator(eig_vecs)

    h_ctrl_transformed = [eigenbasis.dag() * m * eigenbasis for m in h_ctrl]

    noise_gen_chrg_noise = NTGQuasiStatic(
        standard_deviation=[eps_noise_std],
        n_samples_per_trace=n_time_steps,
        n_traces=n_traces,
        always_redraw_samples=False,
        sampling_mode='uncorrelated_deterministic'
    )

    noise_gen_res_noise = NTGQuasiStatic(
        standard_deviation=[eps_res_noise_std],
        n_samples_per_trace=n_time_steps,
        n_traces=n_traces,
        always_redraw_samples=False,
        sampling_mode='uncorrelated_deterministic'
    )

    solver_chrg_noise = SchroedingerSMonteCarlo(
        h_ctrl=h_ctrl_transformed,
        h_drift=[0 * tau_z],
        h_noise=[eigenbasis.dag() * .5 * tau_z * eigenbasis],
        noise_trace_generator=noise_gen_chrg_noise,
        tau=total_time_guess * np.ones(1),
        amplitude_function=amp_func,
        transfer_function=transfer_func
    )

    solver_res_noise = SchroedingerSMonteCarlo(
        h_ctrl=h_ctrl_transformed,
        h_drift=[0 * tau_z],
        h_noise=[eigenbasis.dag() * .5 * sigma_z * eigenbasis],
        noise_trace_generator=noise_gen_res_noise,
        tau=total_time_guess * np.ones(1),
        amplitude_function=amp_func,
        transfer_function=transfer_func
    )
    return solver_chrg_noise, solver_res_noise


def create_mc_solvers_valley(
        n_time_steps, total_time_guess, eps, tunnel_coupling, gmubbx, gmubbz,
        e_zeeman, amp, pulse_envelope, pulse_mode, resonance_frequency_guess,
        tanh_range, phase_shift, eps_noise_std, eps_res_noise_std, n_traces,
        real_valley_l, imag_valley_l, real_valley_r, imag_valley_r,
        **_):
    """
    Build the instances of SchroedingerSMonteCarlo for the Monte Carlo
    simulation of quasistatic noise.

    Helper function assembling the instances of the qopt
    SchroedingerSMonteCarlo. This function create one solver for electric noise
    on the detuning and one for magnetic noise on the total magnetic field.

    Returns
    -------
    solver_chrg_noise: SchroedingerSMonteCarlo
    This solver simulates quasistatic electric noise on the detuning.

    solver_res_noise: SchroedingerSMonteCarlo
    This solver simulates magnetic noise on the total magnetic field.

    """

    resonance_time_guess = 2 * np.pi / resonance_frequency_guess
    time = (total_time_guess / n_time_steps) * np.ones(n_time_steps)
    acc_time = np.cumsum(time)

    value_func = AmplitudeValueFuncValley(
        n_time_steps=n_time_steps, acc_time=acc_time, eps=eps,
        total_time_guess=total_time_guess, amp=amp, e_zeeman=e_zeeman,
        gmubbz=gmubbz, gmubbx=gmubbx, tunnel_coupling=tunnel_coupling,
        pulse_envelope=pulse_envelope, pulse_mode=pulse_mode,
        resonance_time_guess=resonance_time_guess, tanh_range=tanh_range,
        phase_shift=phase_shift, real_valley_l=real_valley_l,
        imag_valley_l=imag_valley_l, real_valley_r=real_valley_r,
        imag_valley_r=imag_valley_r
    )

    derivative_func = AmplitudeDerivativeFuncValley(
        n_time_steps=n_time_steps, acc_time=acc_time, eps=eps,
        total_time_guess=total_time_guess, amp=amp, e_zeeman=e_zeeman,
        gmubbz=gmubbz, gmubbx=gmubbx, tunnel_coupling=tunnel_coupling,
        pulse_envelope=pulse_envelope, pulse_mode=pulse_mode,
        resonance_time_guess=resonance_time_guess, tanh_range=tanh_range,
        phase_shift=phase_shift, real_valley_l=real_valley_l,
        imag_valley_l=imag_valley_l, real_valley_r=real_valley_r,
        imag_valley_r=imag_valley_r
    )

    transfer_func = OversamplingTF(num_ctrls=2, oversampling=n_time_steps)

    amp_func = CustomAmpFunc(
        value_function=value_func,
        derivative_function=derivative_func
    )

    if pulse_mode == 'sine':
        h_drift = create_hamiltonian_valley(
            tunnel_coupling=tunnel_coupling, eps=eps, gmubbx=gmubbx,
            gmubbz=gmubbz, e_zeeman=e_zeeman, real_valley_l=real_valley_l,
            imag_valley_l=imag_valley_l, real_valley_r=real_valley_r,
            imag_valley_r=imag_valley_r)
    elif pulse_mode == 'duty_cycle':
        h_drift = create_hamiltonian_valley(
            tunnel_coupling=tunnel_coupling, eps=eps-amp, gmubbx=gmubbx,
            gmubbz=gmubbz, e_zeeman=e_zeeman, real_valley_l=real_valley_l,
            imag_valley_l=imag_valley_l, real_valley_r=real_valley_r,
            imag_valley_r=imag_valley_r)
    else:
        raise ValueError('Unknown pulse mode.')

    eig_vals, eig_vecs = h_drift.spectral_decomposition(hermitian=True)
    eigenbasis = DenseOperator(eig_vecs)

    h_ctrl_transformed = [eigenbasis.dag() * m * eigenbasis
                          for m in h_ctrl_valley]

    noise_gen_chrg_noise = NTGQuasiStatic(
        standard_deviation=[eps_noise_std],
        n_samples_per_trace=n_time_steps,
        n_traces=n_traces,
        always_redraw_samples=False,
        sampling_mode='uncorrelated_deterministic'
    )

    noise_gen_res_noise = NTGQuasiStatic(
        standard_deviation=[eps_res_noise_std],
        n_samples_per_trace=n_time_steps,
        n_traces=n_traces,
        always_redraw_samples=False,
        sampling_mode='uncorrelated_deterministic'
    )

    solver_chrg_noise = SchroedingerSMonteCarlo(
        h_ctrl=h_ctrl_transformed,
        h_drift=[0 * tau_z_v],
        h_noise=[eigenbasis.dag() * .5 * tau_z_v * eigenbasis],
        noise_trace_generator=noise_gen_chrg_noise,
        tau=total_time_guess * np.ones(1),
        amplitude_function=amp_func,
        transfer_function=transfer_func
    )

    solver_res_noise = SchroedingerSMonteCarlo(
        h_ctrl=h_ctrl_transformed,
        h_drift=[0 * tau_z_v],
        h_noise=[eigenbasis.dag() * .5 * sigma_z_v * eigenbasis],
        noise_trace_generator=noise_gen_res_noise,
        tau=total_time_guess * np.ones(1),
        amplitude_function=amp_func,
        transfer_function=transfer_func
    )
    return solver_chrg_noise, solver_res_noise


def create_mc_fast_noise_solver(
        n_time_steps, total_time_guess, eps, tunnel_coupling, gmubbx, gmubbz,
        e_zeeman, amp, pulse_envelope, pulse_mode, resonance_frequency_guess,
        tanh_range, phase_shift, n_traces_fast_noise, white_noise_psd,
        n_processes_fast_noise, fast_mc_freq_cutoff, **_):
    """
    Build the instances of SchroedingerSMonteCarlo for the Monte Carlo
    simulation of fast electric noise.

    Helper function assembling the instances of the qopt
    SchroedingerSMonteCarlo.

    Returns
    -------
    solver_fast_noise: SchroedingerSMonteCarlo
    This solver simulates fast electric noise on the detuning.

    """

    resonance_time_guess = 2 * np.pi / resonance_frequency_guess
    time = (total_time_guess / n_time_steps) * np.ones(n_time_steps)
    acc_time = np.cumsum(time)

    value_func = AmplitudeValueFunc(
        n_time_steps=n_time_steps, acc_time=acc_time, eps=eps,
        total_time_guess=total_time_guess, amp=amp, e_zeeman=e_zeeman,
        gmubbz=gmubbz, gmubbx=gmubbx, tunnel_coupling=tunnel_coupling,
        pulse_envelope=pulse_envelope, pulse_mode=pulse_mode,
        resonance_time_guess=resonance_time_guess, tanh_range=tanh_range,
        phase_shift=phase_shift
    )

    derivative_func = AmplitudeDerivativeFunc(
        n_time_steps=n_time_steps, acc_time=acc_time, eps=eps,
        total_time_guess=total_time_guess, amp=amp, e_zeeman=e_zeeman,
        gmubbz=gmubbz, gmubbx=gmubbx, tunnel_coupling=tunnel_coupling,
        pulse_envelope=pulse_envelope, pulse_mode=pulse_mode,
        resonance_time_guess=resonance_time_guess, tanh_range=tanh_range,
        phase_shift=phase_shift
    )

    transfer_func = OversamplingTF(num_ctrls=2, oversampling=n_time_steps)

    amp_func = CustomAmpFunc(
        value_function=value_func,
        derivative_function=derivative_func
    )

    if pulse_mode == 'sine':
        h_drift = create_hamiltonian(
            tunnel_coupling=tunnel_coupling, eps=eps, gmubbx=gmubbx,
            gmubbz=gmubbz, e_zeeman=e_zeeman)
    elif pulse_mode == 'duty_cycle':
        h_drift = create_hamiltonian(
            tunnel_coupling=tunnel_coupling, eps=eps-amp, gmubbx=gmubbx,
            gmubbz=gmubbz, e_zeeman=e_zeeman)
    else:
        raise ValueError('Unknown pulse mode.')

    eig_vals, eig_vecs = h_drift.spectral_decomposition(hermitian=True)
    eigenbasis = DenseOperator(eig_vecs)

    h_ctrl_transformed = [eigenbasis.dag() * m * eigenbasis for m in h_ctrl]

    def noise_spectral_density(f):
        delta = np.zeros_like(f)
        delta[f <= fast_mc_freq_cutoff] = white_noise_psd
        return delta

    noise_gen_fast_noise = NTGColoredNoise(
        noise_spectral_density=noise_spectral_density,
        dt=total_time_guess / n_time_steps,
        n_samples_per_trace=n_time_steps,
        n_traces=n_traces_fast_noise,
        always_redraw_samples=True
    )

    solver_fast_noise = SchroedingerSMonteCarlo(
        h_ctrl=h_ctrl_transformed,
        h_drift=[0 * tau_z],
        h_noise=[eigenbasis.dag() * .5 * tau_z * eigenbasis],
        noise_trace_generator=noise_gen_fast_noise,
        processes=n_processes_fast_noise,
        tau=total_time_guess * np.ones(1),
        amplitude_function=amp_func,
        transfer_function=transfer_func
    )

    return solver_fast_noise


def create_mc_fast_noise_solver_valley(
        n_time_steps, total_time_guess, eps, tunnel_coupling, gmubbx, gmubbz,
        e_zeeman, amp, pulse_envelope, pulse_mode, resonance_frequency_guess,
        tanh_range, phase_shift, n_traces_fast_noise, white_noise_psd,
        n_processes_fast_noise, fast_mc_freq_cutoff, real_valley_l,
        imag_valley_l, real_valley_r, imag_valley_r, **_):
    """
    Build the instances of SchroedingerSMonteCarlo for the Monte Carlo
    simulation of fast electric noise.

    Helper function assembling the instances of the qopt
    SchroedingerSMonteCarlo.

    Returns
    -------
    solver_fast_noise: SchroedingerSMonteCarlo
    This solver simulates fast electric noise on the detuning.

    """

    resonance_time_guess = 2 * np.pi / resonance_frequency_guess
    time = (total_time_guess / n_time_steps) * np.ones(n_time_steps)
    acc_time = np.cumsum(time)

    value_func = AmplitudeValueFuncValley(
        n_time_steps=n_time_steps, acc_time=acc_time, eps=eps,
        total_time_guess=total_time_guess, amp=amp, e_zeeman=e_zeeman,
        gmubbz=gmubbz, gmubbx=gmubbx, tunnel_coupling=tunnel_coupling,
        pulse_envelope=pulse_envelope, pulse_mode=pulse_mode,
        resonance_time_guess=resonance_time_guess, tanh_range=tanh_range,
        phase_shift=phase_shift, real_valley_l=real_valley_l,
        imag_valley_l=imag_valley_l, real_valley_r=real_valley_r,
        imag_valley_r=imag_valley_r
    )

    derivative_func = AmplitudeDerivativeFuncValley(
        n_time_steps=n_time_steps, acc_time=acc_time, eps=eps,
        total_time_guess=total_time_guess, amp=amp, e_zeeman=e_zeeman,
        gmubbz=gmubbz, gmubbx=gmubbx, tunnel_coupling=tunnel_coupling,
        pulse_envelope=pulse_envelope, pulse_mode=pulse_mode,
        resonance_time_guess=resonance_time_guess, tanh_range=tanh_range,
        phase_shift=phase_shift, real_valley_l=real_valley_l,
        imag_valley_l=imag_valley_l, real_valley_r=real_valley_r,
        imag_valley_r=imag_valley_r
    )

    transfer_func = OversamplingTF(num_ctrls=2, oversampling=n_time_steps)

    amp_func = CustomAmpFunc(
        value_function=value_func,
        derivative_function=derivative_func
    )

    if pulse_mode == 'sine':
        h_drift = create_hamiltonian_valley(
            tunnel_coupling=tunnel_coupling, eps=eps, gmubbx=gmubbx,
            gmubbz=gmubbz, e_zeeman=e_zeeman, real_valley_l=real_valley_l,
            imag_valley_l=imag_valley_l, real_valley_r=real_valley_r,
            imag_valley_r=imag_valley_r)
    elif pulse_mode == 'duty_cycle':
        h_drift = create_hamiltonian_valley(
            tunnel_coupling=tunnel_coupling, eps=eps-amp, gmubbx=gmubbx,
            gmubbz=gmubbz, e_zeeman=e_zeeman, real_valley_l=real_valley_l,
            imag_valley_l=imag_valley_l, real_valley_r=real_valley_r,
            imag_valley_r=imag_valley_r)
    else:
        raise ValueError('Unknown pulse mode.')

    eig_vals, eig_vecs = h_drift.spectral_decomposition(hermitian=True)
    eigenbasis = DenseOperator(eig_vecs)

    h_ctrl_transformed = [eigenbasis.dag() * m * eigenbasis
                          for m in h_ctrl_valley]

    def noise_spectral_density(f):
        delta = np.zeros_like(f)
        delta[f <= fast_mc_freq_cutoff] = white_noise_psd
        return delta

    noise_gen_fast_noise = NTGColoredNoise(
        noise_spectral_density=noise_spectral_density,
        dt=total_time_guess / n_time_steps,
        n_samples_per_trace=n_time_steps,
        n_traces=n_traces_fast_noise,
        always_redraw_samples=True
    )

    solver_fast_noise = SchroedingerSMonteCarlo(
        h_ctrl=h_ctrl_transformed,
        h_drift=[0 * tau_z_v],
        h_noise=[eigenbasis.dag() * .5 * tau_z_v * eigenbasis],
        noise_trace_generator=noise_gen_fast_noise,
        processes=n_processes_fast_noise,
        tau=total_time_guess * np.ones(1),
        amplitude_function=amp_func,
        transfer_function=transfer_func
    )

    return solver_fast_noise

# ############################ Create Custom Cost Function ####################


def estimate_default_bounds(pulse_mode, resonance_frequency_guess, **_):
    """
    Estimate the default values for the bounds of the pulse optimization.

    Parameters
    ----------
    pulse_mode: string
        Giving the pulse type. Supported are "duty_cycle" for the rectangular
        pulse and "sine" for a sine or cosine pulse.

    resonance_frequency_guess: float
        Estimation for the resonance frequency.

    Returns
    -------
    bounds: list of list of float
        Default baunds for the pulse optimization.

    """
    if pulse_mode == 'sine':
        # first parameter is the resonance frequency.
        # second parameter is the time stretching parameter.
        bounds = [
            [.1 * resonance_frequency_guess,
             10 * resonance_frequency_guess],
            [.02, 20]
        ]
    elif pulse_mode == 'duty_cycle':
        # The range of very high duty cycle parameters is probably not useful
        # first parameter is the duty cycle parameter.
        # the second parameter is the time stretching parameter.
        bounds = [
            [0., .30],
            [.98, 1.02]
        ]
    else:
        raise ValueError('Unknown pulse mode.')
    return bounds


def estimate_default_initial_cond(pulse_mode, resonance_frequency_guess, **_):
    """
    Estimate the default values for the bounds of the pulse optimization.

    Parameters
    ----------
    pulse_mode: string
        Giving the pulse type. Supported are "duty_cycle" for the rectangular
        pulse and "sine" for a sine or cosine pulse.

    resonance_frequency_guess: float
        Estimation for the resonance frequency.

    Returns
    -------
    bounds: list of list of float
        Default baunds for the pulse optimization.

    """
    if pulse_mode == 'sine':
        initial_control_amplitudes = np.asarray(
            [[resonance_frequency_guess, 1]])
    elif pulse_mode == 'duty_cycle':
        # We do not set the initial duty_cycle tuning to 0 to avoid a zero
        # derivative.
        initial_control_amplitudes = np.asarray([[0.04, .9998]])
    else:
        raise ValueError('Unknown pulse mode.')
    return initial_control_amplitudes


def projection_fidelity(propagator, target):
    """ Calculates the process fidelity on the spin degree of freedom.

    Parameters
    ----------
    propagator: DenseOperator
        Full propagator on the full system.

    target: DenseOperator
        Target operation on the spin state.

    Returns
    -------
    fidelity: float
        The spin fidelity.

    """
    dim = propagator.shape[0]
    projector = np.zeros((dim, dim), dtype=complex)
    projector[0, 0] = 1
    projector[1, 1] = 1

    reduced_propagator = propagator * projector

    if dim == 4:
        reduced_propagator = reduced_propagator.ptrace(
            dims=[2, 2],
            remove=[0]
        )
    elif dim == 8:
        reduced_propagator = reduced_propagator.ptrace(
            dims=[2, 2, 2],
            remove=[0, 1]
        )
    return entanglement_fidelity(
        target=target,
        propagator=reduced_propagator
    )


class ProjectionInfidelity(CostFunction):
    """
    Custom cost function class for the process infidelity.

    Equivalent to the 'OperationInfidelity' in qopt with an automatic
    distinction between the spin-orbit and the sping-orbit-valley model.

    """
    def __init__(self,
                 solver,
                 target,
                 label=None):
        if label is None:
            label = ['Projection Infid']
        super().__init__(solver=solver, label=label)
        self.target = target

    def costs(self) -> Union[float, np.ndarray]:
        """ See base class. """
        return 1 - projection_fidelity(
            propagator=self.solver.forward_propagators[-1],
            target=self.target
        )

    def grad(self):
        return NotImplementedError


class ProjectionInfidelityMC(CostFunction):
    """
    Custom cost function class for the process infidelity in Monte Carlo
    simulations.

    Equivalent to the 'OperationNoiseInfidelity' in qopt with an automatic
    distinction between the spin-orbit and the sping-orbit-valley model.

    """
    def __init__(self,
                 solver,
                 target,
                 label=None):
        if label is None:
            label = ['Projection Infid MC']
        super().__init__(solver=solver, label=label)
        self.target = target

    def costs(self) -> Union[float, np.ndarray]:
        """ See base class. """
        n_traces = self.solver.noise_trace_generator.n_traces
        infids = np.empty((n_traces, ), dtype=float)
        for i in range(n_traces):
            infids[i] = projection_fidelity(
                propagator=self.solver.forward_propagators_noise[i][-1],
                target=self.target
            )
        return 1 - np.mean(infids)

    def grad(self):
        return NotImplementedError


# ############################ Create instances of qopt simulators ############


def create_coherent_state_simulator(
        n_time_steps, total_time_guess, eps, tunnel_coupling, gmubbx, gmubbz,
        e_zeeman, amp, rabi_rotation_angle, white_noise_psd, pulse_envelope,
        pulse_mode, resonance_frequency_guess, tanh_range, phase_shift,
        leakage_method, **_):
    """
    This simulator is required for the optimization with only two degrees of
    freedom.

    Returns
    -------
    coherent_state_simulator: Simulator
        Calculates the state fidelity and leakage.

    coherent_state_simulator_ex_leakage: Simulator
        Simulator for the pulse optimization calculating only the state
        fidelity.

    """
    solver_state, _ = create_coherent_solvers(
        n_time_steps=n_time_steps, total_time_guess=total_time_guess, eps=eps,
        tunnel_coupling=tunnel_coupling, gmubbx=gmubbx, gmubbz=gmubbz,
        e_zeeman=e_zeeman, amp=amp, white_noise_psd=white_noise_psd,
        pulse_envelope=pulse_envelope, pulse_mode=pulse_mode,
        resonance_frequency_guess=resonance_frequency_guess,
        tanh_range=tanh_range, phase_shift=phase_shift
    )

    up_state = DenseOperator(np.asarray([[1], [0]]))
    # This state is the initial state in our eigenbasis after tracing out
    # (or cutting out) the charge degree of freedom.
    target = (.5 * DenseOperator.pauli_x()).exp(1j * rabi_rotation_angle)
    # in case of our pi rotation the unitary is the pauli x matrix itself.
    target_state = target * up_state

    if leakage_method == 'cut':
        state_infid_ex_leakage = StateInfidelity(
            solver=solver_state,
            target=target_state,
            computational_states=[0, 1],
            rescale_propagated_state=True
        )
    elif leakage_method == 'partial_trace':
        state_infid_ex_leakage = StateInfidelitySubspace(
            solver=solver_state,
            target=target_state,
            dims=[2, 2],
            remove=[0]
        )
    # We remove the first subspace in the eigenbasis. I assume that the tunnel
    # coupling is much larger than the Zeeman energy 2tc >> E_z and the
    # amplitude is much larger than. Therefore,
    # in the order of the eigenbasis the first two states are distinct by spin.
    else:
        raise ValueError('Unknown leakage method!')

    full_target = DenseOperator(np.vstack(
        (target_state.data, np.zeros((2, 1)))
    ))
    state_infid = StateInfidelity(
        target=full_target,
        solver=solver_state,
    )

    coherent_state_simulator = Simulator(
        solvers=[solver_state, ],
        cost_funcs=[state_infid, state_infid_ex_leakage]
    )
    coherent_state_simulator_ex_leakage = Simulator(
        solvers=[solver_state, ],
        cost_funcs=[state_infid_ex_leakage, ]
    )
    return coherent_state_simulator, coherent_state_simulator_ex_leakage


def find_computational_states(
        eps, tunnel_coupling, gmubbx, gmubbz, e_zeeman, amp, pulse_mode,
        real_valley_l, imag_valley_l, real_valley_r, imag_valley_r, **_):
    """
    Calculates the computational states.

    This function determines the two state with lowest energy which have a
    different spin direction.

    Parameters
    ----------
    amp: float
        Pulse amplitude.

    tunnel_coupling: float
        Tunnel coupling of the double quantum dot.

    eps: float
        Detuning of the double quantum dot. This is the difference in chemical
        potential between the two dots.

    gmubbx: float
        Magnetic gradient across the double quantum dot in x direction.
        Multiplied with the g-factor and Bohr magneton.

    gmubbz: float
        Magnetic gradient across the double quantum dot in z direction.
        Multiplied with the g-factor and Bohr magneton.

    e_zeeman: float
        Average Zeeman energy.

    pulse_mode: string
        Giving the pulse type. Supported are "duty_cycle" for the rectangular
        pulse and "sine" for a sine or cosine pulse.

    real_valley_l: float
        Real part of the valley splitting in the left quantum dot.

    imag_valley_l: float
        Imaginary part of the valley splitting in the left quantum dot.

    real_valley_r: float
        Real part of the valley splitting in the right quantum dot.

    imag_valley_r: float
        Imaginary part of the valley splitting in the right quantum dot.

    Returns
    -------
    computational states: list of int, length: 2
        The indices of the computational states. For example [0, 1] means that
        the computational states are the first two states in the eigenbasis
        ordered by ascending eigenvalue.

    """

    if pulse_mode == 'sine':
        eps_end = eps
    elif pulse_mode == 'duty_cycle':
        eps_end = eps - amp
    else:
        raise ValueError('Unknown pulse mode!')

    h = create_hamiltonian_valley(
        tunnel_coupling=tunnel_coupling,
        eps=eps_end,
        gmubbx=gmubbx,
        gmubbz=gmubbz,
        e_zeeman=e_zeeman,
        real_valley_l=real_valley_l,
        imag_valley_l=imag_valley_l,
        real_valley_r=real_valley_r,
        imag_valley_r=imag_valley_r
    )

    eig_vals, eig_vecs = h.spectral_decomposition(hermitian=True)

    # take the eigenvalues
    ground = DenseOperator(np.expand_dims(eig_vecs[:, 0], axis=1))
    first_exited = DenseOperator(np.expand_dims(eig_vecs[:, 1], axis=1))
    second_exited = DenseOperator(np.expand_dims(eig_vecs[:, 2], axis=1))

    # trace out the orbit and valley
    ground = ground.ptrace(dims=[2, 2, 2], remove=[0, 1])
    first_exited = first_exited.ptrace(dims=[2, 2, 2], remove=[0, 1])
    second_exited = second_exited.ptrace(dims=[2, 2, 2], remove=[0, 1])

    # calculate fidelity to see which has a different spin.
    fid_state_first = np.abs((ground * first_exited.dag()).tr()) ** 2
    fid_state_second = np.abs((ground * second_exited.dag()).tr()) ** 2

    # we want a spin like transition:
    if fid_state_first < fid_state_second:
        computational_states = [0, 1]
    else:
        computational_states = [0, 2]

    return computational_states


def create_coherent_state_simulator_valley(
        n_time_steps, total_time_guess, eps, tunnel_coupling, gmubbx, gmubbz,
        e_zeeman, amp, rabi_rotation_angle, white_noise_psd, pulse_envelope,
        pulse_mode, resonance_frequency_guess, tanh_range, phase_shift,
        leakage_method, real_valley_l, imag_valley_l, real_valley_r,
        imag_valley_r, **_):
    """
    This simulator is required for the optimization with only two degrees of
    freedom.

    Returns
    -------
    coherent_state_simulator: Simulator
        Calculates the state fidelity and leakage.

    coherent_state_simulator_ex_leakage: Simulator
        Simulator for the pulse optimization calculating only the state
        fidelity.

    """
    solver_state, _ = create_coherent_solvers_valley(
        n_time_steps=n_time_steps, total_time_guess=total_time_guess, eps=eps,
        tunnel_coupling=tunnel_coupling, gmubbx=gmubbx, gmubbz=gmubbz,
        e_zeeman=e_zeeman, amp=amp, white_noise_psd=white_noise_psd,
        pulse_envelope=pulse_envelope, pulse_mode=pulse_mode,
        resonance_frequency_guess=resonance_frequency_guess,
        tanh_range=tanh_range, phase_shift=phase_shift,
        real_valley_l=real_valley_l, imag_valley_l=imag_valley_l,
        real_valley_r=real_valley_r, imag_valley_r=imag_valley_r
    )

    up_state = DenseOperator(np.asarray([[1], [0]]))
    # This state is the initial state in our eigenbasis after tracing out
    # (or cutting out) the charge degree of freedom.
    target = (.5 * DenseOperator.pauli_x()).exp(1j * rabi_rotation_angle)
    # in case of our pi rotation the unitary is the pauli x matrix itself.
    target_state = target * up_state

    if leakage_method == 'cut':
        # for the cosine pulse, we need to use this more flexible method.

        # estimate the first excitation state that has a different spin:
        computational_states = find_computational_states(
            tunnel_coupling=tunnel_coupling,
            eps=eps,
            gmubbx=gmubbx,
            gmubbz=gmubbz,
            e_zeeman=e_zeeman,
            real_valley_l=real_valley_l,
            imag_valley_l=imag_valley_l,
            real_valley_r=real_valley_r,
            imag_valley_r=imag_valley_r,
            amp=amp,
            pulse_mode=pulse_mode
        )

        state_infid_ex_leakage = StateInfidelity(
            solver=solver_state,
            target=target_state,
            computational_states=computational_states,
            rescale_propagated_state=True
        )
    elif leakage_method == 'partial_trace':
        state_infid_ex_leakage = StateInfidelitySubspace(
            solver=solver_state,
            target=target_state,
            dims=[2, 2, 2],
            remove=[0, 1]  # keep in mind that these are states in the
            # eigenbasis, which is ordered 'orbital x valley x spin' as long as
            # we assure this order of energy levels.
            # This should always work, we just need that the spin splitting is
            # the smallest and hope that valley-orbit mixing does not create
            # states that mess up the order.
            # Well the valley-orbit mixing does mess it up in the case of the
            # cosine pulse. (central start point.)
        )
    # We remove the first subspace in the eigenbasis. I assume that the tunnel
    # coupling is much larger than the Zeeman energy 2tc >> E_z and the
    # amplitude is much larger than. Therefore,
    # in the order of the eigenbasis the first two states are distinct by spin.
    else:
        raise ValueError('Unknown leakage method!')

    full_target = DenseOperator(np.vstack(
        (target_state.data, np.zeros((6, 1)))
    ))
    state_infid = StateInfidelity(
        target=full_target,
        solver=solver_state,
    )

    coherent_state_simulator = Simulator(
        solvers=[solver_state, ],
        cost_funcs=[state_infid, state_infid_ex_leakage]
    )
    coherent_state_simulator_ex_leakage = Simulator(
        solvers=[solver_state, ],
        cost_funcs=[state_infid_ex_leakage, ]
    )
    return coherent_state_simulator, coherent_state_simulator_ex_leakage


def create_coherent_gate_simulator(
        n_time_steps, total_time_guess, eps, tunnel_coupling, gmubbx, gmubbz,
        e_zeeman, amp, rabi_rotation_angle, white_noise_psd, pulse_envelope,
        pulse_mode, resonance_frequency_guess, tanh_range, phase_shift,
        azimuth_angle_target, use_opt_azimuth_angle, opt_azimuth_angle,
        **_):
    """
    This simulator is required to calculate the gate fidelity and leakage in
    the coherent simulation.

    Returns
    -------
    simulator_gate_leakage: Simulator
        Simulator for the calculation of leakage.

    simulator_gate_infid: Simulator
        Simulator for the calculation of entanglement fidelity.

    """
    _, solver_unitary = create_coherent_solvers(
        n_time_steps=n_time_steps, total_time_guess=total_time_guess, eps=eps,
        tunnel_coupling=tunnel_coupling, gmubbx=gmubbx, gmubbz=gmubbz,
        e_zeeman=e_zeeman, amp=amp, white_noise_psd=white_noise_psd,
        pulse_envelope=pulse_envelope, pulse_mode=pulse_mode,
        resonance_frequency_guess=resonance_frequency_guess,
        tanh_range=tanh_range, phase_shift=phase_shift
    )

    if use_opt_azimuth_angle:
        azimuth_angle = opt_azimuth_angle
    else:
        azimuth_angle = azimuth_angle_target

    target = target_gate(phi=azimuth_angle,
                         rabi_rotation_angle=rabi_rotation_angle)

    projection_infid = ProjectionInfidelity(
        solver=solver_unitary,
        target=target
    )
    leakage_infid = LeakageLiouville(
        solver=solver_unitary,
        computational_states=[0, 1],
        input_unitary=True,
        monte_carlo=False
    )

    simulator_gate_infid = Simulator(
        solvers=[solver_unitary, ],
        cost_funcs=[projection_infid, ]
    )
    simulator_gate_leakage = Simulator(
        solvers=[solver_unitary, ],
        cost_funcs=[leakage_infid, ]
    )

    return simulator_gate_leakage, simulator_gate_infid


def create_coherent_gate_simulator_valley(
        n_time_steps, total_time_guess, eps, tunnel_coupling, gmubbx, gmubbz,
        e_zeeman, amp, rabi_rotation_angle, white_noise_psd, pulse_envelope,
        pulse_mode, resonance_frequency_guess, tanh_range, phase_shift,
        azimuth_angle_target, use_opt_azimuth_angle,
        real_valley_l, imag_valley_l, real_valley_r, imag_valley_r,
        opt_azimuth_angle_valley, **_):
    """
    This simulator is required to calculate the gate fidelity and leakage in
    the coherent simulation.

    Returns
    -------
    simulator_gate_leakage: Simulator
        Simulator for the calculation of leakage.

    simulator_gate_infid: Simulator
        Simulator for the calculation of entanglement fidelity.

    """
    _, solver_unitary = create_coherent_solvers_valley(
        n_time_steps=n_time_steps, total_time_guess=total_time_guess, eps=eps,
        tunnel_coupling=tunnel_coupling, gmubbx=gmubbx, gmubbz=gmubbz,
        e_zeeman=e_zeeman, amp=amp, white_noise_psd=white_noise_psd,
        pulse_envelope=pulse_envelope, pulse_mode=pulse_mode,
        resonance_frequency_guess=resonance_frequency_guess,
        tanh_range=tanh_range, phase_shift=phase_shift,
        real_valley_l=real_valley_l, imag_valley_l=imag_valley_l,
        real_valley_r=real_valley_r, imag_valley_r=imag_valley_r
    )

    if use_opt_azimuth_angle:
        azimuth_angle = opt_azimuth_angle_valley
    else:
        azimuth_angle = azimuth_angle_target

    target = target_gate(phi=azimuth_angle,
                         rabi_rotation_angle=rabi_rotation_angle)

    computational_states = find_computational_states(
        tunnel_coupling=tunnel_coupling,
        eps=eps,
        gmubbx=gmubbx,
        gmubbz=gmubbz,
        e_zeeman=e_zeeman,
        real_valley_l=real_valley_l,
        imag_valley_l=imag_valley_l,
        real_valley_r=real_valley_r,
        imag_valley_r=imag_valley_r,
        amp=amp,
        pulse_mode=pulse_mode
    )

    gate_infid = OperationInfidelity(
        solver=solver_unitary,
        target=target,
        computational_states=computational_states
    )

    leakage_infid = LeakageLiouville(
        solver=solver_unitary,
        computational_states=computational_states,
        input_unitary=True,
        monte_carlo=False
    )

    simulator_gate_infid = Simulator(
        solvers=[solver_unitary, ],
        cost_funcs=[gate_infid, ]
    )
    simulator_gate_leakage = Simulator(
        solvers=[solver_unitary, ],
        cost_funcs=[leakage_infid, ]
    )

    return simulator_gate_leakage, simulator_gate_infid


def create_mc_gate_simulator(
        n_time_steps, total_time_guess, eps, tunnel_coupling, gmubbx, gmubbz,
        e_zeeman, amp, rabi_rotation_angle, white_noise_psd, pulse_envelope,
        pulse_mode, resonance_frequency_guess, tanh_range, phase_shift,
        eps_noise_std, eps_res_noise_std, azimuth_angle_target,
        use_opt_azimuth_angle, opt_azimuth_angle, n_traces, **_):
    """
    Assembles simulators for the calculation of fidelity and leakage in the
    presence of quasistatic electric and magnetic noise.

    Returns
    -------
    simulator_chrg_noise_infid: Simulator
        Simulator for the calculation of the entanglement infidelity caused by
        quasistatic electric noise.

    simulator_chrg_noise_leakage: Simulator
        Simulator for the calculation of leakage caused by
        quasistatic electric noise.

    simulator_res_noise_infid: Simulator
        Simulator for the calculation of the entanglement infidelity caused by
        quasistatic magnetic noise.

    simulator_res_noise_leakage: Simulator
        Simulator for the calculation of leakage caused by
        quasistatic magnetic noise.

    """

    solver_chrg_noise, solver_res_noise = create_mc_solvers(
        n_time_steps=n_time_steps, total_time_guess=total_time_guess, eps=eps,
        tunnel_coupling=tunnel_coupling, gmubbx=gmubbx, gmubbz=gmubbz,
        e_zeeman=e_zeeman, amp=amp, white_noise_psd=white_noise_psd,
        pulse_envelope=pulse_envelope, pulse_mode=pulse_mode,
        resonance_frequency_guess=resonance_frequency_guess,
        tanh_range=tanh_range, phase_shift=phase_shift,
        eps_noise_std=eps_noise_std, eps_res_noise_std=eps_res_noise_std,
        n_traces=n_traces
    )


    if use_opt_azimuth_angle:
        azimuth_angle = opt_azimuth_angle
    else:
        azimuth_angle = azimuth_angle_target

    target = target_gate(phi=azimuth_angle,
                         rabi_rotation_angle=rabi_rotation_angle)

    chrg_noise_infid = ProjectionInfidelityMC(
        solver=solver_chrg_noise,
        target=target
    )
    leakage_chrg_noise_infid = LeakageLiouville(
        solver=solver_chrg_noise,
        computational_states=[0, 1],
        input_unitary=True,
        monte_carlo=True
    )
    simulator_chrg_noise_infid = Simulator(
        solvers=[solver_chrg_noise, ],
        cost_funcs=[chrg_noise_infid, ]
    )
    simulator_chrg_noise_leakage = Simulator(
        solvers=[solver_chrg_noise, ],
        cost_funcs=[leakage_chrg_noise_infid, ]
    )

    res_noise_infid = ProjectionInfidelityMC(
        solver=solver_res_noise,
        target=target
    )
    leakage_res_noise_infid = LeakageLiouville(
        solver=solver_res_noise,
        computational_states=[0, 1],
        input_unitary=True,
        monte_carlo=True
    )
    simulator_res_noise_infid = Simulator(
        solvers=[solver_res_noise, ],
        cost_funcs=[res_noise_infid, ]
    )
    simulator_res_noise_leakage = Simulator(
        solvers=[solver_res_noise, ],
        cost_funcs=[leakage_res_noise_infid, ]
    )

    return simulator_chrg_noise_infid, simulator_chrg_noise_leakage, \
        simulator_res_noise_infid, simulator_res_noise_leakage


def create_mc_gate_simulator_valley(
        n_time_steps, total_time_guess, eps, tunnel_coupling, gmubbx, gmubbz,
        e_zeeman, amp, rabi_rotation_angle, white_noise_psd, pulse_envelope,
        pulse_mode, resonance_frequency_guess, tanh_range, phase_shift,
        eps_noise_std, eps_res_noise_std, azimuth_angle_target,
        use_opt_azimuth_angle, n_traces,
        real_valley_l, imag_valley_l, real_valley_r, imag_valley_r,
        opt_azimuth_angle_valley, **_):
    """
    Assembles simulators for the calculation of fidelity and leakage in the
    presence of quasistatic electric and magnetic noise.

    Returns
    -------
    simulator_chrg_noise_infid: Simulator
        Simulator for the calculation of the entanglement infidelity caused by
        quasistatic electric noise.

    simulator_chrg_noise_leakage: Simulator
        Simulator for the calculation of leakage caused by
        quasistatic electric noise.

    simulator_res_noise_infid: Simulator
        Simulator for the calculation of the entanglement infidelity caused by
        quasistatic magnetic noise.

    simulator_res_noise_leakage: Simulator
        Simulator for the calculation of leakage caused by
        quasistatic magnetic noise.

    """

    solver_chrg_noise, solver_res_noise = create_mc_solvers_valley(
        n_time_steps=n_time_steps, total_time_guess=total_time_guess, eps=eps,
        tunnel_coupling=tunnel_coupling, gmubbx=gmubbx, gmubbz=gmubbz,
        e_zeeman=e_zeeman, amp=amp, white_noise_psd=white_noise_psd,
        pulse_envelope=pulse_envelope, pulse_mode=pulse_mode,
        resonance_frequency_guess=resonance_frequency_guess,
        tanh_range=tanh_range, phase_shift=phase_shift,
        eps_noise_std=eps_noise_std, eps_res_noise_std=eps_res_noise_std,
        n_traces=n_traces,
        real_valley_l=real_valley_l, imag_valley_l=imag_valley_l,
        real_valley_r=real_valley_r, imag_valley_r=imag_valley_r
    )

    if use_opt_azimuth_angle:
        azimuth_angle = opt_azimuth_angle_valley
    else:
        azimuth_angle = azimuth_angle_target

    target = target_gate(phi=azimuth_angle,
                         rabi_rotation_angle=rabi_rotation_angle)

    computational_states = find_computational_states(
        tunnel_coupling=tunnel_coupling,
        eps=eps,
        gmubbx=gmubbx,
        gmubbz=gmubbz,
        e_zeeman=e_zeeman,
        real_valley_l=real_valley_l,
        imag_valley_l=imag_valley_l,
        real_valley_r=real_valley_r,
        imag_valley_r=imag_valley_r,
        amp=amp,
        pulse_mode=pulse_mode
    )

    chrg_noise_infid = OperationNoiseInfidelity(
        solver=solver_chrg_noise,
        target=target,
        computational_states=computational_states
    )
    leakage_chrg_noise_infid = LeakageLiouville(
        solver=solver_chrg_noise,
        computational_states=computational_states,
        input_unitary=True,
        monte_carlo=True
    )
    simulator_chrg_noise_infid = Simulator(
        solvers=[solver_chrg_noise, ],
        cost_funcs=[chrg_noise_infid, ]
    )
    simulator_chrg_noise_leakage = Simulator(
        solvers=[solver_chrg_noise, ],
        cost_funcs=[leakage_chrg_noise_infid, ]
    )

    res_noise_infid = OperationNoiseInfidelity(
        solver=solver_res_noise,
        target=target,
        computational_states=computational_states
    )
    leakage_res_noise_infid = LeakageLiouville(
        solver=solver_res_noise,
        computational_states=computational_states,
        input_unitary=True,
        monte_carlo=True
    )
    simulator_res_noise_infid = Simulator(
        solvers=[solver_res_noise, ],
        cost_funcs=[res_noise_infid, ]
    )
    simulator_res_noise_leakage = Simulator(
        solvers=[solver_res_noise, ],
        cost_funcs=[leakage_res_noise_infid, ]
    )

    return simulator_chrg_noise_infid, simulator_chrg_noise_leakage, \
        simulator_res_noise_infid, simulator_res_noise_leakage


def create_mc_fast_noise_gate_simulator(
        n_time_steps, total_time_guess, eps, tunnel_coupling, gmubbx, gmubbz,
        e_zeeman, amp, rabi_rotation_angle, white_noise_psd, pulse_envelope,
        pulse_mode, resonance_frequency_guess, tanh_range, phase_shift,
        eps_noise_std, eps_res_noise_std, azimuth_angle_target,
        use_opt_azimuth_angle, opt_azimuth_angle, n_traces,
        n_traces_fast_noise, n_processes_fast_noise, fast_mc_freq_cutoff, **_):
    """
    Assembles simulators for the calculation of fidelity and leakage in the
    presence of fast electric noise with a Monte Carlo simulation.

    Returns
    -------
    simulator_fast_noise_entanglement: Simulator
        Simulator for the calculation of the entanglement infidelity caused by
        fast electric noise using a Monate Carlo simulation.

    simulator_fast_noise_leakage: Simulator
        Simulator for the calculation of leakage caused by
        fast electric noise using a Monate Carlo simulation.


    """

    solver_fast_noise = create_mc_fast_noise_solver(
        n_time_steps=n_time_steps, total_time_guess=total_time_guess, eps=eps,
        tunnel_coupling=tunnel_coupling, gmubbx=gmubbx, gmubbz=gmubbz,
        e_zeeman=e_zeeman, amp=amp, white_noise_psd=white_noise_psd,
        pulse_envelope=pulse_envelope, pulse_mode=pulse_mode,
        resonance_frequency_guess=resonance_frequency_guess,
        tanh_range=tanh_range, phase_shift=phase_shift,
        eps_noise_std=eps_noise_std, eps_res_noise_std=eps_res_noise_std,
        n_traces=n_traces, n_traces_fast_noise=n_traces_fast_noise,
        n_processes_fast_noise=n_processes_fast_noise,
        fast_mc_freq_cutoff=fast_mc_freq_cutoff
    )

    if use_opt_azimuth_angle:
        azimuth_angle = opt_azimuth_angle
    else:
        azimuth_angle = azimuth_angle_target

    target = target_gate(phi=azimuth_angle,
                         rabi_rotation_angle=rabi_rotation_angle)

    # fast_noise_infid = LiouvilleMonteCarloEntanglementInfidelity(
    fast_noise_infid = ProjectionInfidelity(
        solver=solver_fast_noise,
        target=target
    )
    leakage_fast_noise_infid = LeakageLiouville(
        solver=solver_fast_noise,
        computational_states=[0, 1],
        input_unitary=True,
        monte_carlo=True
    )

    simulator_fast_noise_leakage = Simulator(
        solvers=[solver_fast_noise, ],
        cost_funcs=[leakage_fast_noise_infid, ]
    )

    simulator_fast_noise_entanglement = Simulator(
        solvers=[solver_fast_noise, ],
        cost_funcs=[fast_noise_infid, ]
    )

    return simulator_fast_noise_entanglement, simulator_fast_noise_leakage


def create_mc_fast_noise_gate_simulator_valley(
        n_time_steps, total_time_guess, eps, tunnel_coupling, gmubbx, gmubbz,
        e_zeeman, amp, rabi_rotation_angle, white_noise_psd, pulse_envelope,
        pulse_mode, resonance_frequency_guess, tanh_range, phase_shift,
        eps_noise_std, eps_res_noise_std, azimuth_angle_target,
        use_opt_azimuth_angle, n_traces,
        n_traces_fast_noise, n_processes_fast_noise, fast_mc_freq_cutoff,
        real_valley_l, imag_valley_l, real_valley_r, imag_valley_r,
        opt_azimuth_angle_valley, **_):
    """
    Assembles simulators for the calculation of fidelity and leakage in the
    presence of fast electric noise with a Monte Carlo simulation.

    Returns
    -------
    simulator_fast_noise_entanglement: Simulator
        Simulator for the calculation of the entanglement infidelity caused by
        fast electric noise using a Monate Carlo simulation.

    simulator_fast_noise_leakage: Simulator
        Simulator for the calculation of leakage caused by
        fast electric noise using a Monate Carlo simulation.


    """

    solver_fast_noise = create_mc_fast_noise_solver_valley(
        n_time_steps=n_time_steps, total_time_guess=total_time_guess, eps=eps,
        tunnel_coupling=tunnel_coupling, gmubbx=gmubbx, gmubbz=gmubbz,
        e_zeeman=e_zeeman, amp=amp, white_noise_psd=white_noise_psd,
        pulse_envelope=pulse_envelope, pulse_mode=pulse_mode,
        resonance_frequency_guess=resonance_frequency_guess,
        tanh_range=tanh_range, phase_shift=phase_shift,
        eps_noise_std=eps_noise_std, eps_res_noise_std=eps_res_noise_std,
        n_traces=n_traces, n_traces_fast_noise=n_traces_fast_noise,
        n_processes_fast_noise=n_processes_fast_noise,
        fast_mc_freq_cutoff=fast_mc_freq_cutoff,
        real_valley_l=real_valley_l, imag_valley_l=imag_valley_l,
        real_valley_r=real_valley_r, imag_valley_r=imag_valley_r
    )

    if use_opt_azimuth_angle:
        azimuth_angle = opt_azimuth_angle_valley
    else:
        azimuth_angle = azimuth_angle_target

    target = target_gate(phi=azimuth_angle,
                         rabi_rotation_angle=rabi_rotation_angle)

    computational_states = find_computational_states(
        tunnel_coupling=tunnel_coupling,
        eps=eps,
        gmubbx=gmubbx,
        gmubbz=gmubbz,
        e_zeeman=e_zeeman,
        real_valley_l=real_valley_l,
        imag_valley_l=imag_valley_l,
        real_valley_r=real_valley_r,
        imag_valley_r=imag_valley_r,
        amp=amp,
        pulse_mode=pulse_mode
    )

    fast_noise_infid = OperationNoiseInfidelity(
        solver=solver_fast_noise,
        target=target,
        computational_states=computational_states
    )
    leakage_fast_noise_infid = LeakageLiouville(
        solver=solver_fast_noise,
        computational_states=computational_states,
        input_unitary=True,
        monte_carlo=True
    )

    simulator_fast_noise_leakage = Simulator(
        solvers=[solver_fast_noise, ],
        cost_funcs=[leakage_fast_noise_infid, ]
    )

    simulator_fast_noise_entanglement = Simulator(
        solvers=[solver_fast_noise, ],
        cost_funcs=[fast_noise_infid, ]
    )

    return simulator_fast_noise_entanglement, simulator_fast_noise_leakage
