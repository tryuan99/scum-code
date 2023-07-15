"""The exponential regression class performs a exponential regression on the given data.

y = a * exp(-x/tau) + b
"""

from typing import Any

import numpy as np
import scipy.optimize
from absl import logging


class ExponentialRegression:
    """Performs an exponential regression."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.a, self.tau, self.b = self._perform_exponential_regression(x, y)

    @property
    def time_constant(self) -> float:
        """Returns the time constant."""
        return self.tau

    @property
    def offset(self) -> float:
        """Returns the offset."""
        return self.b

    def evaluate(self, x: Any) -> Any:
        """Evaluates the exponential regression at the given x-values.

        Args:
            x: x-values.

        Returns:
            The y-values corresponding to the x-values.
        """
        return self.a * np.exp(-1 / self.tau * x) + self.b

    @staticmethod
    def _perform_exponential_regression(
            x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        """Performs an exponential regression.

        Args:
            x: x-values of the data.
            y: y-values of the data.

        Returns:
            (a, tau, b), where y = a * exp(-x/tau) + b are the coefficients of
            the exponential.
        """
        A = np.vstack([x, np.ones(len(x))]).T
        result = np.squeeze(np.linalg.lstsq(A, np.log(y), rcond=None)[0])
        tau_guess, a_guess, b_guess = -1 / result[0], np.exp(result[1]), 0

        # Use an optimizer to find the optimal parameters of the exponential.
        # TODO(titan): Debug why this does not quite work for increasing exponentials.
        def cost(params: np.ndarray):
            """Calculates how well the exponential fits the given data.

            Args:
                params: Three-dimensional vector consisting of (a, tau, b).

            Returns:
                The squared cost between the given exponential and the given data.
            """
            a, tau, b = params
            return np.linalg.norm(a * np.exp(-1 / tau * x) + b - y)

        optimization_results = scipy.optimize.minimize(
            cost,
            np.array([a_guess, tau_guess, b_guess]),
            method="Nelder-Mead",
            options={"maxiter": 10000},
        )
        if not optimization_results.success:
            logging.warning("Optimization failed with message: %s",
                            optimization_results.message)
        return optimization_results.x
