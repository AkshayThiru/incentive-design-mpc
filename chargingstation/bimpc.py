import warnings
from dataclasses import dataclass

import cvxpy as cv
import numpy as np

from chargingstation.lompc import LoMPCConstants
from chargingstation.settings import BIMPC_SOLVER, PRINT_LEVEL


@dataclass
class BiMPCConstants:
    """
    delta:              Relative weight of tracking cost.
    c_gen:              Energy generation cost coefficient.
    u_gen_max:          Maximum charge generation per timestep.
    u_bat_max:          Maximum charge/discharge rate of the storage battery.
    xi_bat_max:         Battery storage capacity.
    tracking_cost_type: "weighted" or "unweighted".
    """

    delta: float
    c_gen: float
    u_gen_max: float
    u_bat_max: float
    xi_bat_max: float
    tracking_cost_type: str = "unweighted"


@dataclass
class BiMPCParameters:
    """
    Nr_s:       Number of small robots in each partition.
    Nr_l:       Number of large robots in each partition.
    beta_s:     Robustness bounds, for each partition of small robots.
    beta_l:     Robustness bounds, for each partition of large robots.
    gamma_s:    Average fraction of battery capacity to be charged, for each partition of small robots.
    gamma_l:    Average fraction of battery capacity to be charged, for each partition of large robots.
    xi0_bat:    Current charge of the storage battery.
    demand:     External energy demand forecast for the control horizon.
    """

    Nr_s: np.ndarray
    Nr_l: np.ndarray
    beta_s: np.ndarray
    beta_l: np.ndarray
    gamma_s: np.ndarray
    gamma_l: np.ndarray
    xi0_bat: float
    demand: np.ndarray


class BiMPC:
    def __init__(
        self,
        N: int,
        Np: int,
        consts: BiMPCConstants,
        consts_s: LoMPCConstants,
        consts_l: LoMPCConstants,
    ) -> None:
        """
        Inputs:
            N:                  Horizon length.
            Np:                 Number of partitions per battery type.
            consts:             BiMPC constants.
            consts_s:           LoMPC constants for small battery type.
            consts_l:           LoMPC constants for large battery type.
        """
        assert consts.delta >= 0
        assert consts.c_gen >= 0
        assert consts.u_gen_max >= 0
        assert consts.u_bat_max >= 0
        assert consts.xi_bat_max >= 0
        assert (consts.tracking_cost_type == "weighted") or (
            consts.tracking_cost_type == "unweighted"
        )
        self._set_constants(N, Np, consts, consts_s, consts_l)

        self._set_cvx_variables()
        self._set_cvx_parameters()
        _params = BiMPCParameters(
            np.zeros((self.Np,)),
            np.zeros((self.Np,)),
            np.zeros((self.Np,)),
            np.zeros((self.Np,)),
            np.zeros((self.Np,)),
            np.zeros((self.Np,)),
            0,
            np.zeros((self.N,)),
        )
        self._update_cvx_parameters(_params)

        self.cons = []
        self._set_cvx_w_constraints()
        self._set_cvx_energy_generation_constraints()
        self._set_cvx_battery_storage_rate_constraints()
        self._set_cvx_battery_storage_constraints()

        self.cost = 0
        self._set_cvx_energy_generation_cost()
        self._set_cvx_tracking_cost(consts)

        self.prob = cv.Problem(cv.Minimize(self.cost), self.cons)
        if PRINT_LEVEL >= 3:
            print("BiMPC problem is DCP:", self.prob.is_dcp())
        self.prob.solve(solver=BIMPC_SOLVER, warm_start=True)

    def _set_constants(
        self,
        N: int,
        Np: int,
        consts: BiMPCConstants,
        consts_s: LoMPCConstants,
        consts_l: LoMPCConstants,
    ) -> None:
        self.N = N
        self.Np = Np
        # BiMPC constants.
        self.delta = consts.delta
        self.c_gen = consts.c_gen
        self.u_gen_max = consts.u_gen_max
        self.u_bat_max = consts.u_bat_max
        self.xi_bat_max = consts.xi_bat_max
        # LoMPC constants.
        self.theta_s = consts_s.theta
        self.theta_l = consts_l.theta
        self.w_max_s = consts_s.w_max
        self.w_max_l = consts_l.w_max
        # BiMPC input matrix, xi_bat = A u_bat + xi0_bat np.ones(N,).
        self.A = np.tril(np.ones((self.N, self.N)))

    def _set_cvx_variables(self) -> None:
        self.w_s = cv.Variable((self.Np, self.N), nonneg=True)
        self.w_l = cv.Variable((self.Np, self.N), nonneg=True)
        self.u_gen = cv.Variable(self.N, nonneg=True)

    def _set_cvx_parameters(self) -> None:
        # Number of robots in each partition.
        self.Nr_s = cv.Parameter(self.Np, nonneg=True)
        self.Nr_l = cv.Parameter(self.Np, nonneg=True)
        # Average fraction of battery capacity to be charged, for each partition.
        self.gamma_s = cv.Parameter(self.Np, nonneg=True)
        self.gamma_l = cv.Parameter(self.Np, nonneg=True)
        # Initial battery charge.
        self.xi0_bat = cv.Parameter(nonneg=True)
        # Demand for the next time horizon.
        self.demand = cv.Parameter(self.N, nonneg=True)
        # Dependent parameters.
        #   Cumulative robustness bounds.
        self.Nr_dot_beta_s = cv.Parameter(nonneg=True)
        self.Nr_dot_beta_l = cv.Parameter(nonneg=True)
        #   Total fraction of battery capacity to be charged, for each partition.
        self.Nr_times_gamma_s = cv.Parameter(self.Np, nonneg=True)
        self.Nr_times_gamma_l = cv.Parameter(self.Np, nonneg=True)

    def _update_cvx_parameters(self, params: BiMPCParameters) -> None:
        self.Nr_s.value = params.Nr_s
        self.Nr_l.value = params.Nr_l
        self.gamma_s.value = params.gamma_s
        self.gamma_l.value = params.gamma_l
        self.xi0_bat.value = params.xi0_bat
        self.demand.value = params.demand
        # Dependent parameters.
        self.Nr_dot_beta_s.value = params.Nr_s @ params.beta_s
        self.Nr_dot_beta_l.value = params.Nr_l @ params.beta_l
        self.Nr_times_gamma_s.value = params.Nr_s * params.gamma_s
        self.Nr_times_gamma_l.value = params.Nr_l * params.gamma_l

    def _set_cvx_w_constraints(self) -> None:
        self.cons += [
            self.w_s <= self.w_max_s,
            self.w_l <= self.w_max_l,
        ]

    def _set_cvx_energy_generation_constraints(self) -> None:
        self.cons += [self.u_gen <= self.u_gen_max]

    def _set_cvx_battery_storage_rate_constraints(self) -> None:
        u_bat_avg = (
            self.u_gen
            - self.demand
            - self.theta_s * self.Nr_s @ self.w_s
            - self.theta_l * self.Nr_l @ self.w_l
        )
        e1 = np.zeros((self.N,))
        e1[0] = 1
        u_bat_err = e1 * (
            self.theta_s * self.Nr_dot_beta_s + self.theta_l * self.Nr_dot_beta_l
        )
        # Maximum discharge rate constraint.
        self.cons += [u_bat_avg - u_bat_err >= -self.u_bat_max]
        # Maximum charge rate constraint.
        self.cons += [u_bat_avg + u_bat_err <= self.u_bat_max]

    def _set_cvx_battery_storage_constraints(self) -> None:
        xi_bat_avg = self.A @ (
            self.u_gen
            - self.demand
            - self.theta_s * self.Nr_s @ self.w_s
            - self.theta_l * self.Nr_l @ self.w_l
        ) + self.xi0_bat * np.ones((self.N,))
        xi_bat_err = np.ones((self.N,)) * (
            self.theta_s * self.Nr_dot_beta_s + self.theta_l * self.Nr_dot_beta_l
        )
        # Battery storage lower bound constraint.
        self.cons += [xi_bat_avg - xi_bat_err >= 0]
        # Battery storage upper bound constraint.
        self.cons += [xi_bat_avg + xi_bat_err <= self.xi_bat_max]

    def _set_cvx_energy_generation_cost(self) -> None:
        self.cost += self.c_gen * cv.sum(cv.power(self.u_gen, 1.7))

    def _set_cvx_tracking_cost(self, consts: BiMPCConstants) -> None:
        if consts.tracking_cost_type == "weighted":
            self._set_cvx_weighted_tracking_cost()
        else:
            self._set_cvx_unweighted_tracking_cost()

    def _set_cvx_weighted_tracking_cost(self) -> None:
        tracking_err_cost = 0
        for p in range(self.Np):
            tracking_err_cost += self.theta_s**2 * cv.sum_squares(
                self.A @ self.w_s[p, :] * self.Nr_s[p] - self.Nr_times_gamma_s[p]
            )
            tracking_err_cost += self.theta_l**2 * cv.sum_squares(
                self.A @ self.w_l[p, :] * self.Nr_l[p] - self.Nr_times_gamma_l[p]
            )
        self.cost += self.delta * tracking_err_cost

    def _set_cvx_unweighted_tracking_cost(self) -> None:
        tracking_err_cost = 0
        for p in range(self.Np):
            tracking_err_cost += cv.sum_squares(
                self.A @ self.w_s[p, :] - self.gamma_s[p]
            )
            tracking_err_cost += cv.sum_squares(
                self.A @ self.w_l[p, :] - self.gamma_l[p]
            )
        self.cost += self.delta * tracking_err_cost

    def solve_bimpc(
        self, params: BiMPCParameters
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inputs:
            params: BiMPC problem parameters.
        Outputs:
            w_s_opt:    Optimal energy output for small robots.
            w_l_opt:    Optimal energy output for large robots.
            u_gen_opt:  Optimal energy generation.
        """
        assert (params.Nr_s.shape == (self.Np,)) and (params.Nr_l.shape == (self.Np,))
        assert (params.beta_s.shape == (self.Np,)) and (
            params.beta_l.shape == (self.Np,)
        )
        assert (params.gamma_s.shape == (self.Np,)) and (
            params.gamma_l.shape == (self.Np,)
        )
        assert params.demand.shape == (self.N,)
        self._update_cvx_parameters(params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.prob.solve(solver=BIMPC_SOLVER, warm_start=True)
        w_s_opt = self.w_s.value
        w_l_opt = self.w_l.value
        u_gen_opt = self.u_gen.value
        return w_s_opt, w_l_opt, u_gen_opt

    def get_bat_input_mat(self) -> np.ndarray:
        return self.A
