"""rslib.regression testing module"""

#! IMPORTS


from os.path import dirname, join
import sys

sys.path += [join(dirname(dirname(dirname(__file__))), "src")]

from rslib.regression import *
import numpy as np


__all__ = ["test_regression"]


#! FUNCTION


def add_noise(
    arr: np.ndarray,
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
    return arr + np.random.randn(len(arr)) * amp * np.std(arr)


def test_regression():
    """test the regression module"""
    x = np.linspace(1, 101, 101)

    # linear regression
    print("\nTESTING LINEAR REGRESSION")
    b_in = [2, 0.5]
    y = add_noise(x * b_in[1] + b_in[0], 0.1)
    model = LinearRegression().fit(x, y)
    betas = model.coefs.values.flatten().tolist()
    z = model.predict(x).flatten()
    rmse = np.mean((y - z) ** 2) ** 0.5
    print(f"Input betas: {b_in}\nOutput betas: {betas}\nRMSE: {rmse:0.3f}\n")

    # polynomial regression
    print("\nTESTING POLYNOMIAL REGRESSION")
    b_in = [2, 0.5, 0.1]
    y = add_noise(b_in[0] + x * b_in[1] + x**2 * b_in[2], 0.1)
    model = PolynomialRegression(degree=2).fit(x, y)
    betas = model.coefs.values.flatten().tolist()
    z = model.predict(x).flatten()
    rmse = np.mean((y - z) ** 2) ** 0.5
    print(f"Input betas: {b_in}\nOutput betas: {betas}\nRMSE: {rmse:0.3f}\n")

    # power regression
    print("\nTESTING POWER REGRESSION")
    b_in = [2, -0.5]
    y = abs(add_noise(b_in[0] * x ** b_in[1], 0.1))
    model = PowerRegression().fit(x, y)
    betas = model.coefs.values.flatten().tolist()
    z = model.predict(x).flatten()
    rmse = np.mean((y - z) ** 2) ** 0.5
    print(f"Input betas: {b_in}\nOutput betas: {betas}\nRMSE: {rmse:0.3f}\n")

    # hyperbolic regression
    print("\nTESTING HYPERBOLIC REGRESSION")
    b_in = [2, -0.5]
    y = abs(add_noise(b_in[0] + b_in[1] / x, 0.1))
    model = HyperbolicRegression().fit(x, y)
    betas = model.coefs.values.flatten().tolist()
    z = model.predict(x).flatten()
    rmse = np.mean((y - z) ** 2) ** 0.5
    print(f"Input betas: {b_in}\nOutput betas: {betas}\nRMSE: {rmse:0.3f}\n")

    # exponential regression
    print("\nTESTING EXPONENTIAL REGRESSION")
    b_in = [2, -0.5]
    y = add_noise(b_in[0] + b_in[1] * np.e**x, 0.1)
    model = ExponentialRegression().fit(x, y)
    betas = model.coefs.values.flatten().tolist()
    z = model.predict(x).flatten()
    rmse = np.mean((y - z) ** 2) ** 0.5
    print(f"Input betas: {b_in}\nOutput betas: {betas}\nRMSE: {rmse:0.3f}\n")


if __name__ == "__main__":
    test_regression()
