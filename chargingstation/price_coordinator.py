import cvxpy as cv
import numpy as np

from chargingstation.lompc import LoMPC, LoMPCConstants
from chargingstation.price_regularizer import PriceRegularizer
from chargingstation.settings import (MAX_PRICE_COORD_ITERATIONS,
                                      PRICE_COORD_EPS_REG, PRICE_COORD_EPS_TOL,
                                      PRICE_COORD_SOLVER, PRINT_SOLVER_INFO)


class PriceCoordinator:
    def __init__(self, N: int, consts: LoMPCConstants, price_type: str) -> None:
        """
        Inputs:
            N:          Horizon length.
            consts:     LoMPC constants.
            price_type: "linear" or "linear-convex".
        """
        assert (price_type == "linear") or (price_type == "linear-convex")
        self.lompc = LoMPC(N, consts)
        self._set_constants(N, consts, price_type)
        self.price_reg = PriceRegularizer(self.N, self.r)

        # Setup CVXPY problem.
        self._set_cvx_variables()
        self._set_cvx_parameters()
        self._update_cvx_parameters(np.eye(self.r), np.zeros((self.r,)))

        self.cost = 0
        self._set_cvx_cost()

        self.prob = cv.Problem(cv.Minimize(self.cost))
        if PRINT_SOLVER_INFO:
            print("Price gradient-descent problem is DCP:", self.prob.is_dcp())
        self.prob.solve(solver=PRICE_COORD_SOLVER, warm_start=True)

    def _set_constants(self, N: int, consts: LoMPCConstants, price_type: str) -> None:
        self.nrobots = None
        self.N = N
        if price_type == "linear":
            self.r = 2 * self.N
        else:
            self.r = 3 * self.N
        self.consts = consts
        self.price_type = price_type
        # Initialize charge levels.
        self.s0 = None
        self.s0_rng = None
        self.gamma_center = None
        # Initialize prices.
        self.prev_prices = np.zeros((self.r,))
        # LoMPC input matrix, y = A w.
        self.A = self.lompc.get_input_mat()
        # Gradient descent regularization weight.
        self.eps_reg = PRICE_COORD_EPS_REG
        # Gradient descent tolerance.
        self.eps_tol = PRICE_COORD_EPS_TOL
        # Strong convexity modulus.
        self.m = self.lompc.get_sc_modulus()

    def set_charge_levels(self, s0: np.ndarray) -> None:
        """
        Inputs:
            s0: (nrobots,) ndarray: Robot charge levels.
        """
        assert all(s0 >= 0) and all(s0 <= self.consts.s_max)
        assert len(s0.shape) == 1
        self.nrobots = len(s0)
        self.s0 = s0
        self.s0_rng = (np.max(self.s0) - np.min(self.s0)) / 2
        self.gamma_center = self.consts.s_max - (np.max(self.s0) + np.min(self.s0)) / 2

    def compute_optimal_prices(
        self, w_ref: np.ndarray, lmbd_r: float
    ) -> tuple[np.ndarray, dict]:
        """
        Inputs:
            w_ref:  Reference input vector from the BiMPC.
            lmbd_r: Robustness price parameter.
        Outputs:
            lmbd:           Optimal price vector.
            solver_stats:   Additional solver info.

        solver_stats is a dict with keys
            iter:               Number of solver iterations.
            price_before_reg:   Price before regularization.
            price_after_reg:    Price after regularization.
        """
        # Convergence tolerance.
        tol, _ = self._get_robustness_bounds(lmbd_r)
        tol += self.eps_tol
        # w-inner product metric.
        A_bar, A_bar_inv = self._get_w_inner_product_metric(lmbd_r)

        # Initialize price iterate from previous prices.
        lmbd_k = np.zeros((3 * self.N))
        lmbd_k[: self.r] = self.prev_prices
        w_k, _ = self.lompc.solve_lompc(lmbd_k, lmbd_r, self.gamma_center)
        # Gradient descent till convergence:
        for iter in range(MAX_PRICE_COORD_ITERATIONS):
            err = self._get_w_avg_err(lmbd_k, lmbd_r, w_ref, A_bar)
            if PRINT_SOLVER_INFO:
                print(
                    f"Iteration: {iter:4d} | Error: {err:13.8f} | Tolerance: {tol:13.8f}",
                    end="\r",
                )
            if err <= tol:
                if PRINT_SOLVER_INFO:
                    print("")
                break
            lmbd_k = self._price_gradient_descent_step(A_bar_inv, w_ref, w_k, lmbd_k)
            w_k, _ = self.lompc.solve_lompc(lmbd_k, lmbd_r, self.gamma_center)
        # Regularize prices.
        price_pre = self.lompc.phi(w_k) @ lmbd_k
        lmbd_k = self._regularize_prices(w_k, lmbd_k)
        price_new = self.lompc.phi(w_k) @ lmbd_k
        if PRINT_SOLVER_INFO:
            w_k_, _ = self.lompc.solve_lompc(lmbd_k, lmbd_r, self.gamma_center)
            print(f"Regularization: Total price: {price_pre:9.3f} -> {price_new:9.3f}")
            print(f"                w-error: {np.linalg.norm(w_k - w_k_):13.8f}")
            err = self._get_w_avg_err(lmbd_k, lmbd_r, w_ref, A_bar)
            print(f"w-error: {err:13.8f} | Tolerance: {tol:13.8f}")
        # Update previous prices.
        self.prev_prices = lmbd_k[: self.r]
        solver_stats = {
            "iter": iter,
            "price_before_reg": price_pre,
            "price_after_reg": price_new,
        }
        return lmbd_k, solver_stats

    def _get_robustness_bounds(self, lmbd_r: float) -> tuple[float, float]:
        kappa = lmbd_r / self.consts.delta
        w_err_bound = np.sqrt(self.N) * self.s0_rng
        w0_err_bound = np.sqrt(self.N) / np.sqrt(self.N + kappa) * self.s0_rng
        return w_err_bound, w0_err_bound

    def _get_w_inner_product_metric(
        self, lmbd_r: float
    ) -> tuple[np.ndarray, np.ndarray]:
        kappa = lmbd_r / self.consts.delta
        A_bar = self.A.T @ self.A + kappa * np.eye(self.N)
        A_bar_inv = np.linalg.inv(A_bar)
        return A_bar, A_bar_inv

    def _get_w_avg_err(
        self, lmbd: np.ndarray, lmbd_r: float, w_ref: np.ndarray, A_bar: np.ndarray
    ) -> float:
        w_avg = np.zeros((self.N,))
        gamma = self.consts.s_max - self.s0
        for i in range(self.nrobots):
            w_i, _ = self.lompc.solve_lompc(lmbd, lmbd_r, gamma[i])
            w_avg += w_i
        w_avg /= self.nrobots
        w_avg_err = np.sqrt((w_avg - w_ref) @ A_bar @ (w_avg - w_ref))
        return w_avg_err

    def _price_gradient_descent_step(
        self, A_bar_inv: np.ndarray, w_ref: np.ndarray, w: np.ndarray, lmbd: np.ndarray
    ) -> np.ndarray:
        """
        Inputs:
            A_bar_inv:  Inverse of the w-inner product metric.
            w_ref:      Reference w vector from the BiMPC.
            w:          Current w iterate that is optimal for lmbd, i.e., w = w*(lmbd).
            lmbd:       Current price iterate.
        Outputs:
            lmbd_next:  Next price iterate.
        """
        phi_ref = self.lompc.phi(w_ref)[: self.r]
        phi = self.lompc.phi(w)[: self.r]
        Dphi = self.lompc.Dphi(w)[: self.r, :]
        P_qp = 1 / (2 * self.m) * Dphi @ A_bar_inv @ Dphi.T + self.eps_reg * np.eye(
            self.r
        )
        q_qp = -2 * P_qp @ lmbd - (phi - phi_ref)

        self._update_cvx_parameters(P_qp, q_qp)
        self.prob.solve(solver=PRICE_COORD_SOLVER, warm_start=True)
        lmbd_next = self.lmbd.value
        return lmbd_next

    def _regularize_prices(self, w: np.ndarray, lmbd: np.ndarray) -> np.ndarray:
        """
        w should be optimal for lmbd, i.e., w = w*(lmbd).
        """
        phi = self.lompc.phi(w)[: self.r]
        Dphi = self.lompc.Dphi(w)[: self.r, :]
        lmbd_reg = self.price_reg.solve_price_regularization(Dphi.T, Dphi.T @ lmbd, phi)
        return lmbd_reg

    def _set_cvx_variables(self) -> None:
        self.lmbd = cv.Variable(self.r, nonneg=True)

    def _set_cvx_parameters(self) -> None:
        self.P_chol_qp = cv.Parameter((self.r, self.r))
        self.q_qp = cv.Parameter(self.r)

    def _update_cvx_parameters(self, P_qp: np.ndarray, q_qp: np.ndarray) -> None:
        P_chol_qp = np.linalg.cholesky(P_qp).T
        self.P_chol_qp.value = P_chol_qp
        self.q_qp.value = q_qp

    def _set_cvx_cost(self) -> None:
        self.cost += cv.sum_squares(self.P_chol_qp @ self.lmbd) + self.q_qp @ self.lmbd
