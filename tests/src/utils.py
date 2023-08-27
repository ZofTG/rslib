"""testing utils module"""


#! IMPORTS

from numpy import random, ndarray, std

__all__ = ["add_noise"]


#! FUNCTIONS


def add_noise(
    arr: ndarray,
    amp: float | int = 0.2,
):
    """
    add_noise with the given amplitude to arr

    Parameters
    ----------
    arr : np.ndarray
        the input array

    amp : float | int, optional
        the noise amplitude, by default 0.2

    Returns
    -------
    arn : np.ndarray
        the input array with added noise
    """
    return arr + random.randn(len(arr)) * amp * std(arr)
