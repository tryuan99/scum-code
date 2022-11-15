"""The parabolic regression class performs a parabolic interpolation on the given data.

y = ax^2 + bx + c
"""

import numpy as np
from typing import Any, Tuple


class ParabolicRegression:
    """Performs a parabolic interpolation."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.a, self.b, self.c = self._perform_parabolic_regression(x, y)

    def peak(self) -> Tuple[float, float]:
        """Returns the x and y-values of the vertex of the parabola."""
        peak_x = -self.b / (2 * self.a)
        return peak_x, self.evaluate(peak_x)

    def evaluate(self, x: Any) -> Any:
        """Evaluates the parabola at the given x-values.

        Args:
            x: x-values.

        Returns:
            The y-values corresponding to the x-values.
        """
        return self.a * x ** 2 + self.b * x + self.c

    @staticmethod
    def _perform_parabolic_regression(
        x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, float, float]:
        """Performs a parabolic regression.

        Args:
            x: x-values of the data.
            y: y-values of the data.

        Returns:
            (a, b, c), where ax^2 + bx + c are the coefficients of the parabola.
        """
        A = np.vstack([x ** 2, x, np.ones(len(x))]).T
        result = np.linalg.lstsq(A, y, rcond=None)[0]
        return np.squeeze(result)
