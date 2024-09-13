import warnings
from dataclasses import dataclass

import cvxpy as cv
import numpy as np

from chargingstation.settings import (LOMPC_SOLVER, MAX_BAT_CHARGE_RATE,
                                      MAX_MAX_BAT_SOC, MIN_MAX_BAT_SOC,
                                      PRINT_LEVEL)


@dataclass
class LoMPCConstants:
    """
    delta:      Relative weight of charging cost.
    theta:      Battery capacity [kWh].
    y_max:      Maximum allowed state of charge (SoC) as a fraction of capacity.
    w_max:      Maximum fraction of charge replenished per time step (normalized charging rate).
    ev_type:    EV type, either "small" or "large".
    """

    delta: float
    theta: float
    y_max: float
    w_max: float
    ev_type: str


class LoMPC:
    def __init__(self, N: int, consts: LoMPCConstants) -> None:
        """
        Inputs:
            N:      LoMPC horizon length.
            consts: LoMPC constants.
        """
        assert (consts.y_max >= MIN_MAX_BAT_SOC) and (consts.y_max <= MAX_MAX_BAT_SOC)
        assert (consts.w_max >= 0) and (consts.w_max <= MAX_BAT_CHARGE_RATE)
        assert (consts.ev_type == "small") or (consts.ev_type == "large")
        self._set_constants(N, consts)

        self._set_cvx_variables()
        self._set_cvx_parameters()
        self._update_cvx_parameters(np.zeros((3 * self.N,)), 0, 0)

        self.cons = []
        self._set_cvx_constraints()

        self.price = 0
        self.cost = 0
        self._set_cvx_bat_degradation_cost()
        self._set_cvx_charging_cost()
        self._set_cvx_prices()

        self.prob = cv.Problem(cv.Minimize(self.cost), self.cons)
        if PRINT_LEVEL >= 3:
            print("LoMPC problem is DCP:", self.prob.is_dcp())
        self.prob.solve(solver=LOMPC_SOLVER, warm_start=True)

    def _set_constants(self, N: int, consts: LoMPCConstants) -> None:
        self.N = N
        self.delta = consts.delta
        self.theta = consts.theta
        self.y_max = consts.y_max
        self.w_max = consts.w_max
        self.ev_type = consts.ev_type
        # Scaling factor for the quadratic electricity cost.
        self.q_scale = 3 * self.theta / (4 * self.w_max)
        # LoMPC input matrix, y = A w.
        self.A = np.tril(np.ones((self.N, self.N)))
        # Strong convexity modulus.
        self.m = 2 * self.delta * self.theta**2

    def _set_cvx_variables(self) -> None:
        self.w = cv.Variable(self.N, nonneg=True)

    def _set_cvx_parameters(self) -> None:
        # Unit electricity price (incentive) parameters.
        self.lmbd = cv.Parameter(3 * self.N, nonneg=True)
        # Additional robustness parameter.
        self.lmbd_r = cv.Parameter(nonneg=True)
        # Fraction of battery capacity remaining to be charged.
        self.gamma = cv.Parameter(nonneg=True)

    def _update_cvx_parameters(
        self, lmbd: np.ndarray, lmbd_r: float, gamma: float
    ) -> None:
        assert gamma <= self.y_max
        self.lmbd.value = lmbd
        self.lmbd_r.value = lmbd_r
        self.gamma.value = gamma

    def _set_cvx_constraints(self) -> None:
        self.cons += [self.w <= self.w_max]

    def _set_cvx_bat_degradation_cost(self) -> None:
        if self.ev_type == "small":
            self._set_cvx_small_bat_degradation_cost()
        else:
            self._set_cvx_large_bat_degradation_cost()

    def _set_cvx_small_bat_degradation_cost(self) -> None:
        # w_lim = 1.5 * self.w_max
        # scale = (self.theta * w_lim) ** 2
        # self.cost += -scale * cv.sum(cv.log(1 - cv.square(self.w / w_lim)))
        self.cost += self.theta**2 * cv.sum_squares(self.w / 0.9)

    def _set_cvx_large_bat_degradation_cost(self) -> None:
        w_rel = self.w / self.w_max
        pwl = cv.sum(
            cv.maximum(
                0.0 * w_rel, w_rel - 0.125, 1.5 * w_rel - 0.375, 2 * w_rel - 0.75
            )
        )
        self.cost += (self.theta * self.w_max) ** 2 * pwl

    def _set_cvx_charging_cost(self) -> None:
        y = self.A @ self.w
        self.cost += (
            self.delta
            * self.theta**2
            * (cv.sum_squares(y) - 2 * self.gamma * cv.sum(y))
        )

    def _set_cvx_prices(self) -> None:
        # Linear prices.
        l_price = self.theta * (
            self.lmbd[: self.N] @ self.w
            + self.lmbd[self.N : 2 * self.N] @ (self.w_max - self.w)
        )
        # Quadratic prices.
        q_price = self.q_scale * self.lmbd[2 * self.N :] @ cv.square(self.w)
        # Robustness prices.
        r_price = self.lmbd_r * self.theta**2 * cv.sum_squares(self.w)
        self.price = l_price + q_price + r_price
        self.cost += self.price

    def solve_lompc(
        self, lmbd: np.ndarray, lmbd_r: float, gamma: float
    ) -> tuple[np.ndarray, float]:
        """
        Inputs:
            lmbd:   Unit price (incentive) vector.
            lmbd_r: Robustness price parameter.
            gamma:  Fraction of battery capacity remaining to be charged.
        Outputs:
            w_opt:      Optimal w vector.
            cost_opt:   Optimal cost.
        """
        self._update_cvx_parameters(lmbd, lmbd_r, gamma)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.prob.solve(solver=LOMPC_SOLVER, warm_start=True, max_iter=int(1e5))
        w_opt = self.w.value
        cost_opt = self.cost.value
        return w_opt, cost_opt

    def get_sc_modulus(self) -> float:
        return self.m

    def get_input_mat(self) -> np.ndarray:
        return self.A

    def get_price0(self, w: np.ndarray, lmbd: np.ndarray, lmbd_r: float) -> float:
        price0 = (
            self.theta * (w[0] * lmbd[0] + (self.w_max - w[0]) * lmbd[self.N])
            + self.q_scale * w[0] ** 2 * lmbd[2 * self.N]
            + self.theta**2 * w[0] ** 2 * lmbd_r
        )
        return price0

    def phi(self, w: np.ndarray) -> np.ndarray:
        assert w.shape == (self.N,)
        # Linear + quadratic prices are given by: lmbd @ phi(w).
        return np.hstack(
            (self.theta * w, self.theta * (self.w_max - w), self.q_scale * (w * w))
        )

    def Dphi(self, w: np.ndarray) -> np.ndarray:
        assert w.shape == (self.N,)
        return np.block(
            [
                [self.theta * np.eye(self.N)],
                [-self.theta * np.eye(self.N)],
                [2 * self.q_scale * np.diag(w)],
            ]
        )
