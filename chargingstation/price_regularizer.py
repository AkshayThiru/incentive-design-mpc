import warnings

import cvxpy as cv
import numpy as np

from chargingstation.settings import PRINT_LEVEL


class PriceRegularizer:
    """
    Solves the LP:
    min  c.T @ x,
    s.t. A @ x == b,
         x >= 0.
    When c = phi(w), A = D phi(w).T, and b = D phi(w).T @ lmbd,
    where w = w*(lmbd), the LP minimizes total price without
    affecting the incentive controllability property.
    """

    def __init__(self, N: int, r: int) -> None:
        """
        Inputs:
            N:  Horizon length.
            r:  Unit price (incentive) vector length.
        """
        assert (N >= 0) and (r >= 0)
        self.N = N
        self.r = r

        self._set_cvx_variables()
        self._set_cvx_parameters()
        self._update_cvx_parameters(
            np.zeros((self.N, self.r)), np.zeros((self.N,)), np.zeros((self.r,))
        )

        self.cons = []
        self._set_cvx_linear_constraint()

        self.cost = 0
        self._set_cvx_cost()

        self.prob = cv.Problem(cv.Minimize(self.cost), self.cons)
        if PRINT_LEVEL >= 3:
            print("Price regularization problem is DCP:", self.prob.is_dcp())
        self.prob.solve(warm_start=False)

    def _set_cvx_variables(self) -> None:
        self.x = cv.Variable(self.r, nonneg=True)

    def _set_cvx_parameters(self) -> None:
        self.A = cv.Parameter((self.N, self.r))
        self.b = cv.Parameter(self.N)
        self.c = cv.Parameter(self.r)

    def _update_cvx_parameters(
        self, A: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> None:
        self.A.value = A
        self.b.value = b
        self.c.value = c

    def _set_cvx_linear_constraint(self) -> None:
        self.cons += [self.A @ self.x == self.b]

    def _set_cvx_cost(self) -> None:
        self.cost += self.c @ self.x

    def solve_price_regularization(
        self, A: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> np.ndarray:
        """
        A x = b should be feasible.
        Inputs:
            A:  Constraint matrix.
            b:  Constraint vector.
            c:  Cost vector.
        Outputs:
            x_opt:  Optimal regularized vector.
        """
        self._update_cvx_parameters(A, b, c)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.prob.solve(warm_start=False)
        x_opt = self.x.value
        return x_opt
