from dataclasses import dataclass

import cvxpy as cv
import numpy as np

MIN_MAX_CHARGE = 0.5
MAX_MAX_CHARGE = 0.9
MAX_CHARGE_RATE = 0.25


@dataclass
class LoMPCConstants:
    delta: float  # Relative weight of tracking cost.
    theta: float  # Battery capacity [kWh].
    s_max: float  # Maximum allowed fraction of battery charge.
    w_max: float  # Maximum fraction of charge replenished per timestep.
    bat_type: str  # Battery type, either "small" or "large".


class LoMPC:
    def __init__(self, N: int, consts: LoMPCConstants) -> None:
        """
        Inputs:
            N: Horizon length.
            consts: LoMPC constants.
        """
        assert (consts.s_max >= MIN_MAX_CHARGE) and (consts.s_max <= MAX_MAX_CHARGE)
        assert (consts.w_max >= 0) and (consts.w_max <= MAX_CHARGE_RATE)
        assert (consts.bat_type == "small") or (consts.bat_type == "large")

        self._set_cvx_constants(N, consts)
        self._set_cvx_variables()
        self._set_cvx_parameters()
        self._update_cvx_parameters(np.zeros((3 * self.N,)), 0, 0)

        self.cons = []
        self._set_cvx_constraints()

        self.price = 0
        self.cost = 0
        self._set_cvx_bat_degradation_cost(consts)
        self._set_cvx_tracking_cost()
        self._set_cvx_prices()

        self.prob = cv.Problem(cv.Minimize(self.cost), self.cons)
        # print("LoMPC problem is DCP:", self.prob.is_dcp())
        self.prob.solve(warm_start=True)

    def _set_cvx_constants(self, N: int, consts: LoMPCConstants) -> None:
        self.N = N
        self.delta = consts.delta
        self.theta = consts.theta
        self.s_max = consts.s_max
        self.w_max = consts.w_max
        # Scaling factor for the quadratic price cost.
        self.q_scale = 3 * self.theta / (4 * self.w_max)
        # LoMPC input matrix, y = A w.
        self.A = np.tril(np.ones((self.N, self.N)))

    def _set_cvx_variables(self) -> None:
        self.w = cv.Variable(self.N, nonneg=True)

    def _set_cvx_parameters(self) -> None:
        # Price (incentive) parameters.
        self.lmbd = cv.Parameter(3 * self.N, nonneg=True)
        # Additional robustness parameter.
        self.lmbd_r = cv.Parameter(nonneg=True)
        # Fraction of battery capacity remaining to be charged.
        self.gamma = cv.Parameter(nonneg=True)

    def _update_cvx_parameters(
        self, lmbd: np.ndarray, lmbd_r: float, gamma: float
    ) -> None:
        assert gamma <= self.s_max
        self.lmbd.value = lmbd
        self.lmbd_r.value = lmbd_r
        self.gamma.value = gamma

    def _set_cvx_constraints(self) -> None:
        self.cons += [self.w <= self.w_max]

    def _set_cvx_bat_degradation_cost(self, consts: LoMPCConstants) -> None:
        if consts.bat_type == "small":
            self._set_cvx_small_bat_degradation_cost()
        else:
            self._set_cvx_large_bat_degradation_cost()

    def _set_cvx_small_bat_degradation_cost(self) -> None:
        w_lim = 1.5 * self.w_max
        scale = (self.theta * w_lim) ** 2
        self.cost += -scale * cv.sum(cv.log(1 - cv.square(self.w / w_lim)))

    def _set_cvx_large_bat_degradation_cost(self) -> None:
        w_rel = self.w / self.w_max
        pwl = cv.sum(
            cv.maximum(
                0.5 * w_rel, w_rel - 0.125, 1.5 * w_rel - 0.375, 2 * w_rel - 0.75
            )
        )
        self.cost += self.theta**2 * pwl

    def _set_cvx_tracking_cost(self) -> None:
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
        self._update_cvx_parameters(lmbd, lmbd_r, gamma)
        self.prob.solve(warm_start=True)
        w_opt = self.w.value
        price_opt = self.price.value
        return w_opt, price_opt

    def phi(self, w: np.ndarray) -> np.ndarray:
        # Linear + quadratic prices are given by: lmbd @ phi(w).
        return np.hstack(
            (self.theta * w, self.theta * (self.w_max - w), self.q_scale * (w * w))
        )

    def Dphi(self, w: np.ndarray) -> np.ndarray:
        return np.block(
            [
                [self.theta * np.eye(self.N)],
                [-self.theta * np.eye(self.N)],
                [2 * self.q_scale * np.diag(w)],
            ]
        )
