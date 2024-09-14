import warnings

import cvxpy as cv
import numpy as np

from chargingstation.lompc import LoMPC, LoMPCConstants
from chargingstation.price_regularizer import PriceRegularizer
from chargingstation.settings import (MAX_PRICE_SOLVER_ITERATIONS,
                                      PRICE_SOLVER_EPS_REG,
                                      PRICE_SOLVER_EPS_TOL,
                                      PRICE_SOLVER_SOLVER,
                                      PRICE_SOLVER_TOL_TYPE, PRINT_LEVEL)


class PriceSolver:
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
        if PRINT_LEVEL >= 3:
            print("Price gradient-descent problem is DCP:", self.prob.is_dcp())
        self.prob.solve(solver=PRICE_SOLVER_SOLVER, warm_start=True)

    def _set_constants(self, N: int, consts: LoMPCConstants, price_type: str) -> None:
        self.nEVs = None
        self.N = N
        if price_type == "linear":
            self.r = 2 * self.N
        else:
            self.r = 3 * self.N
        self.consts = consts
        self.price_type = price_type
        # Initialize charge levels.
        self.y0 = None
        self.y0_rng = None
        self.gamma_sc = None
        # Initialize prices.
        self.prev_prices = np.zeros((self.r,))
        # LoMPC input matrix, y = A w + y_0 1.
        self.A = self.lompc.get_input_mat()
        # Gradient descent regularization weight.
        self.eps_reg = PRICE_SOLVER_EPS_REG
        # Gradient descent tolerance.
        self.eps_tol = PRICE_SOLVER_EPS_TOL
        # Strong convexity modulus.
        self.m = self.lompc.get_sc_modulus()

    def set_charge_levels(self, y0: np.ndarray) -> None:
        """
        Inputs:
            y0: (nEVs,) ndarray: EV normalized SoC array.
        """
        assert all(y0 >= 0) and all(y0 <= self.consts.y_max)
        assert len(y0.shape) == 1
        self.nEVs = len(y0)
        self.y0 = y0
        self.y0_rng = (np.max(self.y0) - np.min(self.y0)) / 2  # = \bar{\Gamma}
        self.gamma_sc = self.consts.y_max - (np.max(self.y0) + np.min(self.y0)) / 2
        self.gamma_sm = self.consts.y_max - np.mean(self.y0)

    def compute_optimal_prices(
        self, w_ref: np.ndarray, lmbd_r: float
    ) -> tuple[np.ndarray, dict]:
        """
        Inputs:
            w_ref:  Reference w vector (team-optimal solution) from the BiMPC.
            lmbd_r: Robustness price parameter.
        Outputs:
            lmbd:           Optimal unit price (incentive) vector.
            solver_stats:   Additional solver info.

        solver_stats is a dict with keys
            iter:                           Number of solver iterations.
            price_before_reg:               Price before regularization.
            price_after_reg:                Price after regularization.
            dual_cost_decrease_actual:      Decrease in -\tilde{g}^*.
            dual_cost_decrease_predicted:   Decrease in -\tilde{g}^*(., \lambda^k).
        """
        # Convergence tolerance.
        tol, w0_err_bound = self.get_robustness_bounds(lmbd_r)
        # w-inner product metric.
        A_bar, A_bar_inv = self._get_w_inner_product_metric(lmbd_r)

        # Initialize price iterate from previous prices.
        lmbd_k, lmbd_k_new = np.zeros((3 * self.N)), np.zeros((3 * self.N))
        lmbd_k[: self.r] = self.prev_prices
        phi_w_ref = self.lompc.phi(w_ref)
        w_k, dual_cost = self.lompc.solve_lompc(lmbd_k, lmbd_r, self.gamma_sc)
        dual_cost_decrease_ac = []
        dual_cost_decrease_pred = []
        # Gradient descent till convergence:
        for iter in range(MAX_PRICE_SOLVER_ITERATIONS):
            w_err_max, _, w_avg_err = self._get_w_err(lmbd_k, lmbd_r, w_ref, A_bar)
            if PRINT_LEVEL >= 2:
                print(
                    f"Iteration     : {iter:4d} || Error (max): {w_err_max:13.8f} | Tolerance: {tol:13.8f} "
                    f"|| Error (avg): {w_avg_err:13.8f} | Tolerance: {tol:13.8f}",
                    end="\r",
                )
                if iter % 10 == 0:
                    print("")
            if PRICE_SOLVER_TOL_TYPE == "max":
                w_err = w_err_max
            else:
                w_err = w_avg_err
            if w_err <= tol:
                if (PRINT_LEVEL >= 2) and not (iter % 10 == 0):
                    print("")
                break
            lmbd_k_new[: self.r], dual_cost_derease = self._price_gradient_descent_step(
                A_bar_inv, w_ref, w_k, lmbd_k[: self.r]
            )
            w_k, dual_cost_new = self.lompc.solve_lompc(
                lmbd_k_new, lmbd_r, self.gamma_sc
            )
            dual_cost_decrease_ac.append(
                dual_cost_new - dual_cost + (lmbd_k - lmbd_k_new) @ phi_w_ref
            )
            dual_cost_decrease_pred.append(dual_cost_derease)
            dual_cost = dual_cost_new
            lmbd_k = lmbd_k_new
        # Regularize prices.
        price_pre = self.lompc.phi(w_k) @ lmbd_k
        lmbd_k[: self.r] = self._regularize_prices(w_k, lmbd_k[: self.r])
        price_new = self.lompc.phi(w_k) @ lmbd_k
        if PRINT_LEVEL >= 1:
            w_k_, _ = self.lompc.solve_lompc(lmbd_k, lmbd_r, self.gamma_sc)
            w_err_max, w0_err, w_avg_err = self._get_w_err(lmbd_k, lmbd_r, w_ref, A_bar)
            if PRINT_LEVEL >= 2:
                print(f"Regularization: Price  : {price_pre:9.3f} -> {price_new:9.3f}")
                print(f"                w-error: {np.linalg.norm(w_k - w_k_):13.8f}")
                print(
                    f"w-error (max) : {w_err_max:13.8f} | Tolerance     : {tol:13.8f}"
                )
                print(
                    f"w-error (avg) : {w_avg_err:13.8f} | Tolerance     : {tol:13.8f}"
                )
            print(
                f"w0-error      : {w0_err:13.8f} | w0 error bound: {w0_err_bound:13.8f}"
            )
        # Update previous prices.
        self.prev_prices = lmbd_k[: self.r]
        solver_stats = {
            "iter": iter,
            "price_before_reg": price_pre,
            "price_after_reg": price_new,
            "dual_cost_decrease_actual": np.array(dual_cost_decrease_ac),
            "dual_cost_decrease_predicted": np.array(dual_cost_decrease_pred),
        }
        return lmbd_k, solver_stats

    def get_gamma_sc(self) -> float:
        return self.gamma_sc

    def get_gamma_sm(self) -> float:
        return self.gamma_sm

    def get_robustness_bounds(self, lmbd_r: float) -> tuple[float, float]:
        kappa = lmbd_r / self.consts.delta + 1e-5
        w_err_bound = np.sqrt(self.N) * self.y0_rng + self.eps_tol
        w0_err_bound = w_err_bound * np.min((1, 1 / np.sqrt(kappa)))
        return w_err_bound, w0_err_bound

    def _get_w_inner_product_metric(
        self, lmbd_r: float
    ) -> tuple[np.ndarray, np.ndarray]:
        kappa = lmbd_r / self.consts.delta
        A_bar = self.A.T @ self.A + kappa * np.eye(self.N)
        A_bar_inv = np.linalg.inv(A_bar)
        return A_bar, A_bar_inv

    def _get_w_err(
        self, lmbd: np.ndarray, lmbd_r: float, w_ref: np.ndarray, A_bar: np.ndarray
    ) -> tuple[float, float, float]:
        w_avg = np.zeros((self.N,))
        w0 = np.zeros((self.nEVs,))
        w_err_max = 0
        gamma = self.consts.y_max - self.y0
        for i in range(self.nEVs):
            w_i, _ = self.lompc.solve_lompc(lmbd, lmbd_r, gamma[i])
            w_avg += w_i
            w0[i] = w_i[0]
            w_err_i = np.sqrt((w_i - w_ref) @ A_bar @ (w_i - w_ref))
            if w_err_i > w_err_max:
                w_err_max = w_err_i
        w_avg = w_avg / self.nEVs
        w_avg_err = np.sqrt((w_avg - w_ref) @ A_bar @ (w_avg - w_ref))
        # w0_err = np.max(w0) - np.min(w0) # This is not the correct error metric.
        w0_err = np.abs(w_avg[0] - w_ref[0])
        return w_err_max, w0_err, w_avg_err

    def _price_gradient_descent_step(
        self, A_bar_inv: np.ndarray, w_ref: np.ndarray, w: np.ndarray, lmbd: np.ndarray
    ) -> np.ndarray:
        """
        Inputs:
            A_bar_inv:  Inverse of the w-inner product metric.
            w_ref:      Reference w vector (team-optimal solution) from the BiMPC.
            w:          Current w iterate that is optimal for lmbd, i.e., w = w*(lmbd).
            lmbd:       Current price iterate.
        Outputs:
            lmbd_next:          Next price iterate.
            dual_cost_decrease: Decrease in majorization cost.
        """
        phi_ref = self.lompc.phi(w_ref)[: self.r]
        phi = self.lompc.phi(w)[: self.r]
        Dphi = self.lompc.Dphi(w)[: self.r, :]
        P_qp = 1 / (2 * self.m) * Dphi @ A_bar_inv @ Dphi.T + self.eps_reg * np.eye(
            self.r
        )
        q_qp = -2 * P_qp @ lmbd - (phi - phi_ref)
        dual_cost = lmbd @ P_qp @ lmbd + q_qp @ lmbd

        self._update_cvx_parameters(P_qp, q_qp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.prob.solve(solver=PRICE_SOLVER_SOLVER, warm_start=True)
        lmbd_next = self.lmbd.value
        dual_cost_new = self.cost.value
        dual_cost_decrease = dual_cost - dual_cost_new
        return lmbd_next, dual_cost_decrease

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

    def get_w0_price0(
        self, lmbd: np.ndarray, lmbd_r: float
    ) -> tuple[np.ndarray, float]:
        lmbd_ = np.zeros((3 * self.N))
        lmbd_[: self.r] = lmbd
        w0 = np.zeros((self.nEVs,))
        price0 = 0
        gamma = self.consts.y_max - self.y0
        for i in range(self.nEVs):
            w_i, _ = self.lompc.solve_lompc(lmbd_, lmbd_r, gamma[i])
            w0[i] = w_i[0]
            price0 += self.lompc.get_price0(w_i, lmbd_, lmbd_r)
        price0 = price0 / self.nEVs
        return w0, price0
