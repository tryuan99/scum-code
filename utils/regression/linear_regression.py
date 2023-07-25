"""The linear regression class performs a linear regression on the given data.

y = mx + b
"""

from typing import Any

import numpy as np


class LinearRegression:
    """Performs a linear regression."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.m, self.b, self.residuals = self._perform_linear_regression(x, y)

    @property
    def slope(self) -> float:
        """Returns the slope."""
        return self.m

    @property
    def y_intercept(self) -> float:
        """Returns the y-intercept."""
        return self.b

    def evaluate(self, x: Any) -> Any:
        """Evaluates the linear regression at the given x-values.

        Args:
            x: x-values.

        Returns:
            The y-values corresponding to the x-values.
        """
        return self.m * x + self.b

    @staticmethod
    def _perform_linear_regression(x: np.ndarray,
                                   y: np.ndarray) -> tuple[float, float, float]:
        """Performs a linear regression.

        Args:
            x: x-values of the data.
            y: y-values of the data.

        Returns:
            (m, b, residuals), where m is the slope and b is the y-intercept.
        """
        A = np.vstack([x, np.ones(len(x))]).T
        result, residuals = np.linalg.lstsq(A, y, rcond=None)[:2]
        m, b = np.squeeze(result)
        return m, b, residuals[0] if len(residuals) > 0 else 0


class WeightedLinearRegression(LinearRegression):
    """Performs a linear regression with weighted least squares."""

    def __init__(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray):
        self.m, self.b, self.residuals = self._perform_weighted_linear_regression(
            x, y, weights)

    @staticmethod
    def _perform_weighted_linear_regression(
            x: np.ndarray, y: np.ndarray,
            weights: np.ndarray) -> tuple[float, float, float]:
        """Performs a linear regression with weighted least squares.

        Args:
            x: x-values of the data.
            y: y-values of the data.
            weights: Weights for each sample.

        Returns:
            (m, b, residuals), where m is the slope and b is the y-intercept.
        """
        A = np.vstack([x, np.ones(len(x))]).T
        W = np.sqrt(np.diag(weights))
        A_weighted = np.multiply(A, weights[:, np.newaxis])
        y_weighted = np.multiply(weights, y)
        result, residuals = np.linalg.lstsq(A_weighted, y_weighted,
                                            rcond=None)[:2]
        m, b = np.squeeze(result)
        return m, b, residuals[0] if len(residuals) > 0 else 0
