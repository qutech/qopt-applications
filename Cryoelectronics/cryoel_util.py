import numpy as np
from typing import Union


def discretize(floating_point_number: Union[np.ndarray, float],
               n_bits: Union[np.ndarray, int],
               maximum_values: Union[np.ndarray, float],
               minimum_values: Union[np.ndarray, float] = 0,
               ) -> (Union[np.ndarray, int], Union[np.ndarray, float]):
    """
    Discretizes a single pulse value or an array of pulse values.

    This function approximates floating numbers between minimum_values and
    maximum_values on a discrete scale with n_bits bits.

    Parameters
    ----------
    floating_point_number: float
        Number or array of numbers that will be discretized. Values between 0
        and maximum_values are assumed.

    n_bits: int
        Number of digits.

    minimum_values: float
        We assume all numbers to be greater or equal to minimum_values.

    maximum_values: float
        We assume all numbers to be smaller or equal to minimum_values.

    Returns
    -------
    digital_number: int or np.array
        Positive integer describing the discrete value.

    out_rounded_value: float or np.array
        Value which can be expressed by a binary number with length n_digits.

    """

    # First, we need to check dimensions:
    if type(n_bits) != int:
        assert n_bits.shape == floating_point_number.shape

    if type(minimum_values) != float:
        assert minimum_values.shape == floating_point_number.shape

    if type(maximum_values) != float:
        assert maximum_values.shape == floating_point_number.shape

    # Then we need to check value bounds
    if (floating_point_number > maximum_values).any():
        raise ValueError('The rounding numbers exceed the maximum!')

    if (floating_point_number < minimum_values).any():
        raise ValueError('The rounding numbers are below the minimum!')

    discrete_steps = (maximum_values - minimum_values) / (2 ** n_bits - 1)

    # calculate values differences
    relative_numbers = (floating_point_number - minimum_values)

    # express as a multiples of the discrete steps
    digital_number = np.round(
        relative_numbers / discrete_steps
    )

    # bring the rounded values to the former scale
    out_rounded_value = digital_number * discrete_steps + minimum_values

    return digital_number, out_rounded_value


