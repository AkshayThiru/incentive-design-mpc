import warnings
from dataclasses import dataclass
from enum import Enum

import cvxpy as cv
import numpy as np

from chargingstation.lompc import LoMPCConstants
from chargingstation.settings import BIMPC_SOLVER, PRINT_LEVEL


class BiMPCChargingCostType(Enum):
    WEIGHTED = 0
    UNWEIGHTED = 1
    EXP_UNWEIGHTED = 2


@dataclass
class BiMPCConstants:
    """
    delta:              Relative weight of charging cost.
    c_g:                Electricity generation cost coefficient.
    u_g_max:            Maximum electricity generation per timestep.
    u_b_max:            Maximum charge/discharge rate of the storage battery.
    x_max:              Battery storage capacity.
    charging_cost_type: Enum of type BiMPCChargingCostType.
    exp_rate:           Rate of expoenential growth for EXP_UNWEIGHTED charging cost.
    """

    delta: float
    c_g: float
    u_g_max: float
    u_b_max: float
    x_max: float
    charging_cost_type: BiMPCChargingCostType
    exp_rate: float = 1  # Use np.Inf if only the cost at the final timestep is needed.


@dataclass
class BiMPCParameters:
    """
    Mp_s:       Number of small EVs in each partition.
    Mp_l:       Number of large EVs in each partition.
    beta_s:     Robustness bounds, for each partition of small EVs.
    beta_l:     Robustness bounds, for each partition of large EVs.
    gamma_sm:   Average fraction of battery capacity to be charged, for each partition of small EVs.
    gamma_lm:   Average fraction of battery capacity to be charged, for each partition of large EVs.
    x0:         Current charge of the storage battery.
    demand:     External electricity demand forecast for the control horizon.
    """

    Mp_s: np.ndarray
    Mp_l: np.ndarray
    beta_s: np.ndarray
    beta_l: np.ndarray
    gamma_sm: np.ndarray
    gamma_lm: np.ndarray
    x0: float
    demand: np.ndarray


class BiMPC:
    def __init__(
        self,
        N: int,
        P: int,
        consts_bi: BiMPCConstants,
        consts_s: LoMPCConstants,
        consts_l: LoMPCConstants,
    ) -> None:
        """
        Inputs:
            N:                  Horizon length.
            P:                  Number of partitions per EV type.
            consts_bi:          BiMPC constants.
            consts_s:           LoMPC constants for small EVs.
            consts_l:           LoMPC constants for large EVs.
        """
        assert consts_bi.delta >= 0
        assert consts_bi.c_g >= 0
        assert consts_bi.u_g_max >= 0
        assert consts_bi.u_b_max >= 0
        assert consts_bi.x_max >= 0
        assert consts_bi.exp_rate >= 1
        self._set_constants(N, P, consts_bi, consts_s, consts_l)

        self._set_cvx_variables()
        self._set_cvx_parameters()
        _params = BiMPCParameters(
            np.zeros((self.P,)),
            np.zeros((self.P,)),
            np.zeros((self.P,)),
            np.zeros((self.P,)),
            np.zeros((self.P,)),
            np.zeros((self.P,)),
            0,
            np.zeros((self.N,)),
        )
        self._update_cvx_parameters(_params)

        self.cons = []
        self._set_cvx_w_constraints()
        self._set_cvx_electricity_generation_constraints()
        self._set_cvx_battery_storage_rate_constraints()
        self._set_cvx_battery_storage_constraints()

        self.cost = 0
        self._set_cvx_electricity_generation_cost()
        self._set_cvx_charging_cost()

        self.prob = cv.Problem(cv.Minimize(self.cost), self.cons)
        if PRINT_LEVEL >= 3:
            print("BiMPC problem is DCP:", self.prob.is_dcp())
        self.prob.solve(solver=BIMPC_SOLVER, warm_start=True)

    def _set_constants(
        self,
        N: int,
        P: int,
        consts_bi: BiMPCConstants,
        consts_s: LoMPCConstants,
        consts_l: LoMPCConstants,
    ) -> None:
        self.N = N
        self.P = P
        # BiMPC constants.
        self.delta = consts_bi.delta
        self.c_g = consts_bi.c_g
        self.u_g_max = consts_bi.u_g_max
        self.u_b_max = consts_bi.u_b_max
        self.x_max = consts_bi.x_max
        self.exp_rate = consts_bi.exp_rate * 1.0
        self.charging_cost_type = consts_bi.charging_cost_type
        # LoMPC constants.
        self.theta_s = consts_s.theta
        self.theta_l = consts_l.theta
        self.w_max_s = consts_s.w_max
        self.w_max_l = consts_l.w_max
        # BiMPC input matrix, x = A u_b + x0 1.
        self.A = np.tril(np.ones((self.N, self.N)))

    def _set_cvx_variables(self) -> None:
        self.w_hat_s = cv.Variable((self.P, self.N), nonneg=True)
        self.w_hat_l = cv.Variable((self.P, self.N), nonneg=True)
        self.u_g = cv.Variable(self.N, nonneg=True)

    def _set_cvx_parameters(self) -> None:
        # Number of EVs in each partition.
        self.Mp_s = cv.Parameter(self.P, nonneg=True)
        self.Mp_l = cv.Parameter(self.P, nonneg=True)
        # Average fraction of battery capacity to be charged, for each partition.
        self.gamma_sm = cv.Parameter(self.P, nonneg=True)
        self.gamma_lm = cv.Parameter(self.P, nonneg=True)
        # Initial battery charge.
        # self.x0 = cv.Parameter(nonneg=True)
        self.x0 = (
            cv.Parameter()
        )  # Due to numerical errors, x0 can become slightly negative.
        # External demand for the next time horizon.
        self.demand = cv.Parameter(self.N, nonneg=True)
        # Dependent parameters.
        #   Cumulative robustness bounds.
        self.Mp_dot_beta_s = cv.Parameter(nonneg=True)
        self.Mp_dot_beta_l = cv.Parameter(nonneg=True)
        #   Total fraction of battery capacity to be charged, for each partition.
        self.Mp_times_gamma_sm = cv.Parameter(self.P, nonneg=True)
        self.Mp_times_gamma_lm = cv.Parameter(self.P, nonneg=True)

    def _update_cvx_parameters(self, params: BiMPCParameters) -> None:
        self.Mp_s.value = params.Mp_s
        self.Mp_l.value = params.Mp_l
        self.gamma_sm.value = params.gamma_sm
        self.gamma_lm.value = params.gamma_lm
        self.x0.value = params.x0
        self.demand.value = params.demand
        # Dependent parameters.
        self.Mp_dot_beta_s.value = params.Mp_s @ params.beta_s
        self.Mp_dot_beta_l.value = params.Mp_l @ params.beta_l
        self.Mp_times_gamma_sm.value = params.Mp_s * params.gamma_sm
        self.Mp_times_gamma_lm.value = params.Mp_l * params.gamma_lm

    def _set_cvx_w_constraints(self) -> None:
        self.cons += [self.w_hat_s <= self.w_max_s, self.w_hat_l <= self.w_max_l]

    def _set_cvx_electricity_generation_constraints(self) -> None:
        self.cons += [self.u_g <= self.u_g_max]

    def _set_cvx_battery_storage_rate_constraints(self) -> None:
        u_b_hat = (
            self.u_g
            - self.demand
            - self.theta_s * self.Mp_s @ self.w_hat_s
            - self.theta_l * self.Mp_l @ self.w_hat_l
        )
        e1 = np.zeros((self.N,))
        e1[0] = 1
        delta_err = (
            self.theta_s * self.Mp_dot_beta_s + self.theta_l * self.Mp_dot_beta_l
        )
        # Maximum discharge rate constraint.
        self.cons += [u_b_hat - delta_err * e1 >= -self.u_b_max]
        # Maximum charge rate constraint.
        self.cons += [u_b_hat + delta_err * e1 <= self.u_b_max]

    def _set_cvx_battery_storage_constraints(self) -> None:
        x_hat = self.A @ (
            self.u_g
            - self.demand
            - self.theta_s * self.Mp_s @ self.w_hat_s
            - self.theta_l * self.Mp_l @ self.w_hat_l
        ) + self.x0 * np.ones((self.N,))
        delta_err = (
            self.theta_s * self.Mp_dot_beta_s + self.theta_l * self.Mp_dot_beta_l
        )
        # Battery storage lower bound constraint.
        self.cons += [x_hat - delta_err * np.ones((self.N,)) >= 0]
        # Battery storage upper bound constraint.
        self.cons += [x_hat + delta_err * np.ones((self.N,)) <= self.x_max]

    def _set_cvx_electricity_generation_cost(self) -> None:
        self.cost += self.c_g * cv.sum(cv.power(self.u_g, 1.7))

    def _set_cvx_charging_cost(self) -> None:
        if self.charging_cost_type == BiMPCChargingCostType.WEIGHTED:
            self._set_cvx_weighted_charging_cost()
        elif self.charging_cost_type == BiMPCChargingCostType.UNWEIGHTED:
            self._set_cvx_unweighted_charging_cost()
        elif self.charging_cost_type == BiMPCChargingCostType.EXP_UNWEIGHTED:
            self._set_cvx_exp_unweighted_charging_cost()
        else:
            raise NotImplementedError

    def _set_cvx_weighted_charging_cost(self) -> None:
        charging_cost = 0
        for p in range(self.P):
            charging_cost += self.theta_s ** 2 * cv.sum_squares(
                self.A @ self.w_hat_s[p, :] * self.Mp_s[p] - self.Mp_times_gamma_sm[p]
            )
            charging_cost += self.theta_l ** 2 * cv.sum_squares(
                self.A @ self.w_hat_l[p, :] * self.Mp_l[p] - self.Mp_times_gamma_lm[p]
            )
        self.cost += self.delta * charging_cost

    def _set_cvx_unweighted_charging_cost(self) -> None:
        charging_cost = 0
        for p in range(self.P):
            charging_cost += cv.sum_squares(
                self.A @ self.w_hat_s[p, :] - self.gamma_sm[p]
            )
            charging_cost += cv.sum_squares(
                self.A @ self.w_hat_l[p, :] - self.gamma_lm[p]
            )
        self.cost += self.delta * charging_cost

    def _set_cvx_exp_unweighted_charging_cost(self) -> None:
        charging_cost = 0
        exp_weight = np.power(self.exp_rate, np.arange(-self.N + 1, 1, 1))
        for p in range(self.P):
            charging_cost += exp_weight @ cv.square(
                self.A @ self.w_hat_s[p, :] - self.gamma_sm[p]
            )
            charging_cost += exp_weight @ cv.square(
                self.A @ self.w_hat_l[p, :] - self.gamma_lm[p]
            )
        self.cost += self.delta * charging_cost

    def solve_bimpc(
        self, params: BiMPCParameters
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inputs:
            params: BiMPC problem parameters.
        Outputs:
            w_hat_s_opt:    Team-optimal electricity output for small EVs.
            w_hat_l_opt:    Team-optimal electricity output for large EVs.
            u_g_opt:        Team-optimal electricity generation.
        """
        assert (params.Mp_s.shape == (self.P,)) and (params.Mp_l.shape == (self.P,))
        assert (params.beta_s.shape == (self.P,)) and (params.beta_l.shape == (self.P,))
        assert (params.gamma_sm.shape == (self.P,)) and (
            params.gamma_lm.shape == (self.P,)
        )
        assert params.demand.shape == (self.N,)
        self._update_cvx_parameters(params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.prob.solve(solver=BIMPC_SOLVER, warm_start=True)
        # print(f"BiMPC solve time: {self.prob.solver_stats.solve_time:.6f} s")
        w_hat_s_opt = self.w_hat_s.value
        w_hat_l_opt = self.w_hat_l.value
        u_g_opt = self.u_g.value
        return w_hat_s_opt, w_hat_l_opt, u_g_opt

    def get_bat_input_mat(self) -> np.ndarray:
        return self.A
