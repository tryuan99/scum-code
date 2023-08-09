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
        """Slope of the linear regression."""
        return self.m

    @property
    def y_intercept(self) -> float:
        """y-intercept of the linear regression."""
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
        A = (x**np.arange(2)[:, np.newaxis]).T
        result, residuals = np.linalg.lstsq(A, y, rcond=None)[:2]
        b, m = np.squeeze(result)
        return m, b, residuals[0] if len(residuals) > 0 else 0


class WeightedLinearRegression(LinearRegression):
    """Performs a linear regression with weighted least squares."""

    def __init__(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray):
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.weights = np.copy(weights)
        assert np.all(self.weights >= 0)
        self.m, self.b, self.residuals = self._perform_weighted_linear_regression(
            self.x, self.y, self.weights)

    @property
    def x_mean(self):
        """Weighted mean of the x-values."""
        return np.sum(self.weights * self.x) / np.sum(self.weights)

    @property
    def y_mean(self):
        """Weighted mean of the y-values."""
        return np.sum(self.weights * self.y) / np.sum(self.weights)

    @property
    def slope_variance(self) -> float:
        """Variance of the slope of the linear regression.

        This function assumes that the weights are equal to 1/sample variance.
        """
        return 1 / np.sum(self.weights * (self.x - self.x_mean)**2)

    @property
    def y_intercept_variance(self) -> float:
        """Variance of the y-intercept of the linear regression.

        This function assumes that the weights are equal to 1/sample variance.
        """
        return 1 / np.sum(self.weights) + self.slope_variance * self.x_mean**2

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
        A = (x**np.arange(2)[:, np.newaxis]).T
        W = np.sqrt(weights)
        A_weighted = A * W[:, np.newaxis]
        y_weighted = y * W
        result, residuals = np.linalg.lstsq(A_weighted, y_weighted,
                                            rcond=None)[:2]
        b, m = np.squeeze(result)
        return m, b, residuals[0] if len(residuals) > 0 else 0
