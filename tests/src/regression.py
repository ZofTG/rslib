"""rslib.regression testing module"""

#! IMPORTS


import sys
from os.path import dirname, join

import numpy as np

from .utils import add_noise

sys.path += [join(dirname(dirname(dirname(__file__))), "src", "rslib")]

from regression import *

__all__ = ["test_regression"]


#! FUNCTION


def test_regression():
    """test the regression module"""
    x = np.linspace(1, 101, 101)

    # linear regression
    print("\nTESTING LINEAR REGRESSION")
    b_in = [2, 0.5]
    y = add_noise(x * b_in[1] + b_in[0], 0.1)
    model = LinearRegression().fit(x, y)
    betas = model.betas.values.flatten().tolist()
    z = model.predict(x).flatten()
    rmse = np.mean((y - z) ** 2) ** 0.5
    print(f"Input betas: {b_in}\nOutput betas: {betas}\nRMSE: {rmse:0.3f}\n")

    # Log regression
    print("\nTESTING LOG REGRESSION")
    b_in = [2, 0.5, 0.1]
    y = add_noise(b_in[0] + np.log(x) * b_in[1] + np.log(x) ** 2 * b_in[2], 0.1)
    model = LogRegression(degree=2).fit(x, y)
    betas = model.betas.values.flatten().tolist()
    z = model.predict(x).flatten()
    rmse = np.mean((y - z) ** 2) ** 0.5
    print(f"Input betas: {b_in}\nOutput betas: {betas}\nRMSE: {rmse:0.3f}\n")

    # polynomial regression
    print("\nTESTING POLYNOMIAL REGRESSION")
    b_in = [2, 0.5, 0.1]
    y = add_noise(b_in[0] + x * b_in[1] + x**2 * b_in[2], 0.1)
    model = PolynomialRegression(degree=2).fit(x, y)
    betas = model.betas.values.flatten().tolist()
    z = model.predict(x).flatten()
    rmse = np.mean((y - z) ** 2) ** 0.5
    print(f"Input betas: {b_in}\nOutput betas: {betas}\nRMSE: {rmse:0.3f}\n")

    # power regression
    print("\nTESTING POWER REGRESSION")
    b_in = [2, -0.5]
    y = abs(add_noise(b_in[0] * x ** b_in[1], 0.1))
    model = PowerRegression().fit(x, y)
    betas = model.betas.values.flatten().tolist()
    z = model.predict(x).flatten()
    rmse = np.mean((y - z) ** 2) ** 0.5
    print(f"Input betas: {b_in}\nOutput betas: {betas}\nRMSE: {rmse:0.3f}\n")

    # hyperbolic regression
    print("\nTESTING HYPERBOLIC REGRESSION")
    b_in = [2, -0.5]
    y = abs(add_noise(b_in[0] + b_in[1] / x, 0.1))
    model = HyperbolicRegression().fit(x, y)
    betas = model.betas.values.flatten().tolist()
    z = model.predict(x).flatten()
    rmse = np.mean((y - z) ** 2) ** 0.5
    print(f"Input betas: {b_in}\nOutput betas: {betas}\nRMSE: {rmse:0.3f}\n")

    # exponential regression
    print("\nTESTING EXPONENTIAL REGRESSION")
    b_in = [2, -0.5]
    y = add_noise(b_in[0] + b_in[1] * np.e**x, 0.1)
    model = ExponentialRegression().fit(x, y)
    betas = model.betas.values.flatten().tolist()
    z = model.predict(x).flatten()
    rmse = np.mean((y - z) ** 2) ** 0.5
    print(f"Input betas: {b_in}\nOutput betas: {betas}\nRMSE: {rmse:0.3f}\n")


if __name__ == "__main__":
    test_regression()
