from typing import Any

import numpy as np


class PolynomialRegression:
    """Performs a polynomial regression."""

    def __init__(self, x: np.ndarray, y: np.ndarray, degree: int):
        self.coeffs, self.residuals = self._perform_polynomial_regression(
            x, y, degree)

    @property
    def coefficients(self) -> float:
        """Coefficients of the polynomial regression.

        The coefficients are sorted in increasing powers.
        """
        return self.coeffs

    def evaluate(self, x: Any) -> Any:
        """Evaluates the polynomial regression at the given x-values.

        Args:
            x: x-values.

        Returns:
            The y-values corresponding to the x-values.
        """
        powers = np.arange(len(self.coefficients))
        powers_of_x = x**powers[:, np.newaxis]
        return np.dot(self.coeffs, powers_of_x)

    @staticmethod
    def _perform_polynomial_regression(x: np.ndarray, y: np.ndarray,
                                       degree: int) -> tuple[np.ndarray, float]:
        """Performs a polynomial regression.

        Args:
            x: x-values of the data.
            y: y-values of the data.

        Returns:
            (coefficients, residuals), where coefficients are the polynomial
            coefficients.
        """
        powers = np.arange(degree + 1)
        A = (x**powers[:, np.newaxis]).T
        result, residuals = np.linalg.lstsq(A, y, rcond=None)[:2]
        coefficients = np.squeeze(result)
        return coefficients, residuals[0] if len(residuals) > 0 else 0
