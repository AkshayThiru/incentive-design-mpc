from dataclasses import dataclass

import numpy as np
import time

from chargingstation.bimpc import BiMPC, BiMPCConstants, BiMPCParameters
from chargingstation.lompc import LoMPCConstants
from chargingstation.price_solver import PriceSolver
from chargingstation.settings import (ADD_RESIDUAL_CHARGE_TO_BATTERY,
                                      MAX_INITIAL_SOC,
                                      MIN_FULL_CHARGE_FRACTION,
                                      MIN_INITIAL_SOC, PRINT_LEVEL)


@dataclass
class ChargingStationConstants:
    """
    simulation_length:  Length of the simulation [hours].
    horizon_bimpc:      BiMPC horizon.
    horizon_lompc:      LoMPC horizon (<= BiMPC horizon).
    nEVs_per_EV_type:   Number of small (and large) EVs.
    npartitions:        Number of partitions per EV type.
    demand:             External demand vector.
    bimpc_consts:       Normalized constants for the BiMPC.
    small_EV_consts:    Constants for the small EV LoMPC.
    large_EV_consts:    Constants for the large EV LoMPC.
    price_type:         "linear" or "linear-convex".
    """

    simulation_length: int
    horizon_bimpc: int
    horizon_lompc: int
    nEVs_per_EV_type: int
    npartitions: int
    demand: np.ndarray
    bimpc_consts: BiMPCConstants
    small_EV_consts: LoMPCConstants
    large_EV_consts: LoMPCConstants
    price_type: str


class ChargingStation:
    def __init__(self, consts: ChargingStationConstants) -> None:
        assert consts.simulation_length >= 1
        assert (consts.horizon_bimpc >= consts.horizon_lompc) and (
            consts.horizon_lompc >= 1
        )
        assert consts.nEVs_per_EV_type >= 1
        assert consts.npartitions >= 1
        assert (len(consts.demand.shape) == 1) and (
            consts.demand.shape[0]
            >= consts.simulation_length + consts.horizon_bimpc + 1
        )
        # Set constants, initialize PriceSolvers and BiMPC.
        self._set_constants(consts)
        self.bimpc = BiMPC(
            self.N_bi, self.P, self.consts_bi, self.consts_s, self.consts_l
        )
        self.price_solver_s = PriceSolver(self.N_lo, self.consts_s, self.price_type)
        self.price_solver_l = PriceSolver(self.N_lo, self.consts_l, self.price_type)

        # Initialize state variables = (EV SoCs, charge stored).
        self._init_states()
        # Initialize logs.
        self._init_logs(consts)

    def _set_constants(self, consts: ChargingStationConstants) -> None:
        self.Tf = consts.simulation_length
        self.N_bi = consts.horizon_bimpc
        self.N_lo = consts.horizon_lompc
        self.M_2 = consts.nEVs_per_EV_type
        self.P = consts.npartitions
        self.demand = consts.demand
        self.consts_bi = consts.bimpc_consts
        self.consts_s = consts.small_EV_consts
        self.consts_l = consts.large_EV_consts
        self.price_type = consts.price_type
        if self.price_type == "linear":
            self.r = 2 * self.N_lo
        else:
            self.r = 3 * self.N_lo
        # Range of initial SoCs of EVs.
        self.y0_min = MIN_INITIAL_SOC
        self.y0_max = MAX_INITIAL_SOC
        self.y0_s_rng = np.linspace(
            self.y0_min, self.consts_s.y_max, self.P + 1
        )  # Partition definition (small EVs).
        self.y0_l_rng = np.linspace(
            self.y0_min, self.consts_l.y_max, self.P + 1
        )  # Partition definition (large EVs).
        # Total charge capacity of EVs.
        self.B = (self.consts_s.theta + self.consts_l.theta) * self.M_2

    def _init_states(self) -> None:
        self.y_s = self.y0_min + (self.y0_max - self.y0_min) * np.random.random(
            (self.M_2,)
        )
        self.y_l = self.y0_min + (self.y0_max - self.y0_min) * np.random.random(
            (self.M_2,)
        )
        self.x = 0  # Storage battery SoC, normalized wrt B.

        self.t = 0
        self.ncharged_s = 0
        self.ncharged_l = 0

        self.idx_s = np.zeros((self.M_2,), dtype=int)
        self.idx_l = np.zeros((self.M_2,), dtype=int)
        self._update_indices()

    def _update_indices(self) -> None:
        for p in range(self.P):
            mask_s = (self.y_s >= self.y0_s_rng[p]) & (self.y_s <= self.y0_s_rng[p + 1])
            self.idx_s[mask_s] = p
            mask_l = (self.y_l >= self.y0_l_rng[p]) & (self.y_l <= self.y0_l_rng[p + 1])
            self.idx_l[mask_l] = p

    def _init_logs(self, consts: ChargingStationConstants) -> None:
        self.logs = {}
        self.logs["constants"] = consts
        self.logs["inputs"] = {
            "w_s": np.zeros((self.P, self.Tf)),
            "w_l": np.zeros((self.P, self.Tf)),
            "w_hat_s": np.zeros((self.P, self.Tf)),
            "w_hat_l": np.zeros((self.P, self.Tf)),
            "u_g": np.zeros((self.Tf,)),
        }
        self.logs["states"] = {"x": np.zeros((self.Tf,))}
        self.logs["bounds"] = {
            "beta_s": np.zeros((self.P, self.Tf)),
            "beta_l": np.zeros((self.P, self.Tf)),
        }
        self.logs["statistics"] = {
            "ncharged_s": 0,
            "ncharged_l": 0,
            "gamma_sm": np.zeros((self.P, self.Tf)),
            "gamma_lm": np.zeros((self.P, self.Tf)),
            "niter_s": np.zeros((self.P, self.Tf), dtype=int),
            "niter_l": np.zeros((self.P, self.Tf), dtype=int),
            "Mp_s": np.zeros((self.P, self.Tf), dtype=int),
            "Mp_l": np.zeros((self.P, self.Tf), dtype=int),
        }
        self.logs["prices"] = {
            "lmbd_r": np.zeros((self.Tf)),
            "avg_price_s": np.zeros((self.P, self.Tf)),
            "avg_price_l": np.zeros((self.P, self.Tf)),
            "price_red_s": np.zeros((self.P, self.Tf)),
            "price_red_l": np.zeros((self.P, self.Tf)),
        }

    def simulate(self) -> dict:
        for _ in range(self.Tf):
            self._step()
        return self.logs

    def _step(self):
        if PRINT_LEVEL >= 1:
            print("-" * 50)
            print(f"Iteration {self.t}")
            print("-" * 50)
        # (Optional) Feasibility conditon for kappa.
        lmbd_r = 0
        #   Compute BiMPC solution.
        start_time = time.time()
        w_hat_s, w_hat_l, u_g, stats_bi = self._get_bimpc_solution(lmbd_r)
        end_time = time.time()
        # print(f"BiMPC setup + solve time: {end_time - start_time:.6f} s")
        # Compute prices from PriceSolver.
        start_time = time.time()
        prices_s, prices_l, stats_s, stats_l = self._get_optimal_prices(
            w_hat_s, w_hat_l, lmbd_r
        )
        end_time = time.time()
        # print(f"Optimal incentive (price) setup + solve time: {end_time - start_time:.6f} s\n")
        # Get w0 and price paid by LoMPCs.
        w0_s, w0_l, price0_s, price0_l = self._get_w0_price0(prices_s, prices_l, lmbd_r)
        # Update logs.
        nu = (w_hat_s, w_hat_l, u_g, w0_s, w0_l)
        stats = (stats_bi, stats_s, stats_l)
        price0 = (price0_s, price0_l)
        self._update_logs(lmbd_r, nu, stats, price0)
        # Update state variables.
        self._update_state(w0_s, w0_l, u_g[0])
        # Update time.
        self.t += 1

    def _get_bimpc_solution(
        self, lmbd_r: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        # (Optional): Check feasibility condition for kappa.
        # Set normalized BiMPC parameters.
        #   Set normalized Mp, beta, and gamma_m.
        Mp_s, Mp_l = np.zeros((self.P,), dtype=int), np.zeros((self.P,), dtype=int)
        beta_s, beta_l = np.zeros((self.P,)), np.zeros((self.P,))
        gamma_sm, gamma_lm = np.zeros((self.P,)), np.zeros((self.P,))
        for p in range(self.P):
            mask_s = self.idx_s == p
            Mp_s[p] = mask_s.sum()
            if Mp_s[p] > 0:
                y0p_s = self.y_s[mask_s]
                self.price_solver_s.set_charge_levels(y0p_s)
                _, beta_s[p] = self.price_solver_s.get_robustness_bounds(lmbd_r)
                gamma_sm[p] = self.price_solver_s.get_gamma_sm()
            mask_l = self.idx_l == p
            Mp_l[p] = mask_l.sum()
            if Mp_l[p] > 0:
                y0p_l = self.y_l[mask_l]
                self.price_solver_l.set_charge_levels(y0p_l)
                _, beta_l[p] = self.price_solver_l.get_robustness_bounds(lmbd_r)
                gamma_lm[p] = self.price_solver_l.get_gamma_sm()
        Mp_s_ = Mp_s / self.B
        Mp_l_ = Mp_l / self.B
        #   Set normalized demand.
        demand = self.demand[self.t : self.t + self.N_bi] / self.B
        #   Normalized BiMPC parameters.
        bimpc_params = BiMPCParameters(
            Mp_s_, Mp_l_, beta_s, beta_l, gamma_sm, gamma_lm, self.x, demand
        )
        # Compute BiMPC solution.
        w_hat_s, w_hat_l, u_g = self.bimpc.solve_bimpc(bimpc_params)
        # Store statistics.
        stats_bi = {
            "Mp_s": Mp_s,
            "Mp_l": Mp_l,
            "beta_s": beta_s,
            "beta_l": beta_l,
            "gamma_sm": gamma_sm,
            "gamma_lm": gamma_lm,
        }
        if PRINT_LEVEL >= 1:
            total_w0_hat = (
                self.consts_s.theta * Mp_s_ @ w_hat_s[:, 0]
                + self.consts_l.theta * Mp_l_ @ w_hat_l[:, 0]
            )
            u0_b_hat = u_g[0] - demand[0] - total_w0_hat
            u0_b_err = (
                self.consts_s.theta * Mp_s_ @ beta_s
                + self.consts_l.theta * Mp_l_ @ beta_l
            )
            x_hat = self.x + u0_b_hat
            print(
                "EV distribution (small): "
                + " + ".join("{:4d}".format(n) for n in Mp_s)
                + " = {:4d}".format(np.sum(Mp_s))
            )
            print(
                "EV distribution (large): "
                + " + ".join("{:4d}".format(n) for n in Mp_l)
                + " = {:4d}".format(np.sum(Mp_l))
            )
            print(
                f"Electricity generated  : {u_g[0]:13.8f} | Max: {self.consts_bi.u_g_max:13.8f}"
            )
            print(f"Demand                 : {demand[0]:13.8f}")
            print(f"Predicted output (EVs) : {total_w0_hat:13.8f}")
            print(
                f"Predicted battery input: [{u0_b_hat - u0_b_err:8.5f}, {u0_b_hat + u0_b_err:8.5f}] | Max (mag): {self.consts_bi.u_b_max:8.5f}"
            )
            print(f"Current battery state  : {self.x}")
            print(
                f"Predicted battery state: Min: 0 | [{x_hat - u0_b_err:8.5f}, {x_hat + u0_b_err:8.5f}] | Max: {self.consts_bi.x_max:8.5f}"
            )

            if PRINT_LEVEL >= 2:
                print("")
        return w_hat_s, w_hat_l, u_g, stats_bi

    def _get_optimal_prices(
        self, w_hat_s: np.ndarray, w_hat_l: np.ndarray, lmbd_r: float
    ) -> tuple[np.ndarray, np.ndarray, list, list]:
        # Reduce the horizon for the LoMPC.
        w_hat_s_opt, w_hat_l_opt = w_hat_s[:, : self.N_lo], w_hat_l[:, : self.N_lo]
        prices_s, prices_l = np.zeros((self.P, self.r)), np.zeros((self.P, self.r))
        stats_s, stats_l = [], []
        for p in range(self.P):
            y0p_s = self.y_s[self.idx_s == p]
            if len(y0p_s) > 0:
                self.price_solver_s.set_charge_levels(y0p_s)
                if PRINT_LEVEL >= 1:
                    print(f"Small EVs, partition {p:2d}: ", end="")
                    if PRINT_LEVEL >= 2:
                        print("\n" + "-" * 27)
                lmbd_, stats_ = self.price_solver_s.compute_optimal_prices(
                    w_hat_s_opt[p, :], lmbd_r
                )
                prices_s[p, :] = lmbd_[: self.r]
                stats_s.append(stats_)
                if PRINT_LEVEL >= 2:
                    print("")
            else:
                stats_s.append({})
            y0p_l = self.y_l[self.idx_l == p]
            if len(y0p_l) > 0:
                self.price_solver_l.set_charge_levels(y0p_l)
                if PRINT_LEVEL >= 1:
                    print(f"Large EVs, partition {p:2d}: ", end="")
                    if PRINT_LEVEL >= 2:
                        print("\n" + "-" * 27)
                lmbd_, stats_ = self.price_solver_l.compute_optimal_prices(
                    w_hat_l_opt[p, :], lmbd_r
                )
                prices_l[p, :] = lmbd_[: self.r]
                stats_l.append(stats_)
                if PRINT_LEVEL >= 2:
                    print("")
            else:
                stats_l.append({})
        return prices_s, prices_l, stats_s, stats_l

    def _get_w0_price0(
        self, prices_s: np.ndarray, prices_l: np.ndarray, lmbd_r: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Use prices to get w0 for small and large battery types.
        w0_s, w0_l = np.zeros((self.M_2,)), np.zeros((self.M_2,))
        price0_s, price0_l = np.zeros((self.P,)), np.zeros((self.P,))
        for p in range(self.P):
            y0p_s = self.y_s[self.idx_s == p]
            if len(y0p_s) > 0:
                self.price_solver_s.set_charge_levels(y0p_s)
                w0_s[self.idx_s == p], price0_s[p] = self.price_solver_s.get_w0_price0(
                    prices_s[p, :], lmbd_r
                )
            y0p_l = self.y_l[self.idx_l == p]
            if len(y0p_l) > 0:
                self.price_solver_l.set_charge_levels(y0p_l)
                w0_l[self.idx_l == p], price0_l[p] = self.price_solver_l.get_w0_price0(
                    prices_l[p, :], lmbd_r
                )
        return w0_s, w0_l, price0_s, price0_l

    def _update_state(self, w0_s: np.ndarray, w0_l: np.ndarray, u0_g: float) -> None:
        # Update EV SoCs and indices.
        residual_charge = 0
        self.y_s += w0_s
        mask_s = self.y_s > MIN_FULL_CHARGE_FRACTION * self.consts_s.y_max
        residual_charge += self.consts_s.theta * np.sum(
            self.y_s[mask_s] - MIN_FULL_CHARGE_FRACTION * self.consts_s.y_max
        )
        self.y_s[mask_s] = self.y0_min + (self.y0_max - self.y0_min) * np.random.random(
            (mask_s.sum(),)
        )
        self.ncharged_s += mask_s.sum()
        self.y_l += w0_l
        mask_l = self.y_l > MIN_FULL_CHARGE_FRACTION * self.consts_l.y_max
        residual_charge += self.consts_l.theta * np.sum(
            self.y_l[mask_l] - MIN_FULL_CHARGE_FRACTION * self.consts_l.y_max
        )
        self.y_l[mask_l] = self.y0_min + (self.y0_max - self.y0_min) * np.random.random(
            (mask_l.sum(),)
        )
        self.ncharged_l += mask_l.sum()
        self._update_indices()
        if not ADD_RESIDUAL_CHARGE_TO_BATTERY:
            residual_charge = 0
        # Update battery charge state.
        u0_b = (
            u0_g
            + (
                -self.consts_s.theta * np.sum(w0_s)
                - self.consts_l.theta * np.sum(w0_l)
                + residual_charge
                - self.demand[self.t]
            )
            / self.B
        )
        self.x += u0_b
        if PRINT_LEVEL >= 1:
            print(f"# small EVs charged    : {self.ncharged_s:5d}")
            print(f"# large EVs charged    : {self.ncharged_l:5d}")
            print("")

    def _update_logs(
        self, lmbd_r: float, nu: tuple, stats: tuple, price0: tuple
    ) -> None:
        w_hat_s, w_hat_l, u_g, w0_s, w0_l = nu
        stats_bi, stats_s, stats_l = stats
        price0_s, price0_l = price0
        # Log inputs.
        #   Log w_s and w_l.
        for p in range(self.P):
            if len(w0_s[self.idx_s == p]) > 0:
                self.logs["inputs"]["w_s"][p, self.t] = np.mean(w0_s[self.idx_s == p])
            if len(w0_l[self.idx_l == p]) > 0:
                self.logs["inputs"]["w_l"][p, self.t] = np.mean(w0_l[self.idx_l == p])
        #   Log other inputs.
        self.logs["inputs"]["w_hat_s"][:, self.t] = w_hat_s[:, 0]
        self.logs["inputs"]["w_hat_l"][:, self.t] = w_hat_l[:, 0]
        self.logs["inputs"]["u_g"][self.t] = u_g[0]
        # Log battery state.
        self.logs["states"]["x"][self.t] = self.x
        # Log bounds.
        self.logs["bounds"]["beta_s"][:, self.t] = stats_bi["beta_s"]
        self.logs["bounds"]["beta_l"][:, self.t] = stats_bi["beta_l"]
        # Log statistics.
        #   Update number of EVs charged.
        self.logs["statistics"]["ncharged_s"] = self.ncharged_s
        self.logs["statistics"]["ncharged_l"] = self.ncharged_l
        #   Log gamma.
        self.logs["statistics"]["gamma_sm"][:, self.t] = stats_bi["gamma_sm"]
        self.logs["statistics"]["gamma_lm"][:, self.t] = stats_bi["gamma_lm"]
        #   Log number of iterations for convergence.
        for p in range(self.P):
            if stats_s[p]:
                self.logs["statistics"]["niter_s"][p, self.t] = stats_s[p]["iter"]
            else:
                self.logs["statistics"]["niter_s"][p, self.t] = -1
            if stats_l[p]:
                self.logs["statistics"]["niter_l"][p, self.t] = stats_l[p]["iter"]
            else:
                self.logs["statistics"]["niter_l"][p, self.t] = -1
        #   Log number of EVs in each partition.
        self.logs["statistics"]["Mp_s"][:, self.t] = stats_bi["Mp_s"]
        self.logs["statistics"]["Mp_l"][:, self.t] = stats_bi["Mp_l"]
        # Log prices.
        #   Log lmbd_r.
        self.logs["prices"]["lmbd_r"][self.t] = lmbd_r
        #   Log average prices paid at time t.
        self.logs["prices"]["avg_price_s"][:, self.t] = price0_s
        self.logs["prices"]["avg_price_l"][:, self.t] = price0_l
        #   Log average price reduction from price regulation.
        for p in range(self.P):
            if stats_s[p]:
                self.logs["prices"]["price_red_s"][p, self.t] = (
                    stats_s[p]["price_after_reg"] - stats_s[p]["price_before_reg"]
                )
            else:
                self.logs["prices"]["price_red_s"][p, self.t] = np.nan
            if stats_l[p]:
                self.logs["prices"]["price_red_l"][p, self.t] = (
                    stats_l[p]["price_after_reg"] - stats_l[p]["price_before_reg"]
                )
            else:
                self.logs["prices"]["price_red_l"][p, self.t] = np.nan
