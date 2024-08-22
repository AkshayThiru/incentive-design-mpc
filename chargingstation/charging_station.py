from dataclasses import dataclass

import numpy as np

from chargingstation.bimpc import BiMPC, BiMPCConstants, BiMPCParameters
from chargingstation.lompc import LoMPCConstants
from chargingstation.price_coordinator import PriceCoordinator
from chargingstation.settings import (ADD_RESIDUAL_CHARGE_TO_BATTERY,
                                      MAX_INITIAL_CHARGE,
                                      MIN_FULL_CHARGE_FRACTION,
                                      MIN_INITIAL_CHARGE, PRINT_LEVEL)


@dataclass
class ChargingStationConstants:
    """
    simulation_length:      Length of the simulation [hours].
    horizon_bimpc:          BiMPC horizon.
    horizon_lompc:          LoMPC horizon (<= BiMPC horizon).
    nrobots_per_bat_type:   Number of small (and large) robots.
    npartitions:            Number of partitions per robot type.
    demand:                 External demand vector.
    bimpc_consts:           Normalized constants for the BiMPC.
    small_bat_consts:       Constants for the small robot LoMPC.
    large_bat_consts:       Constants for the large robot LoMPC.
    price_type:             "linear" or "linear-convex".
    """

    simulation_length: int
    horizon_bimpc: int
    horizon_lompc: int
    nrobots_per_bat_type: int
    npartitions: int
    demand: np.ndarray
    bimpc_consts: BiMPCConstants
    small_bat_consts: LoMPCConstants
    large_bat_consts: LoMPCConstants
    price_type: str


class ChargingStation:
    def __init__(self, consts: ChargingStationConstants) -> None:
        assert consts.simulation_length >= 1
        assert (consts.horizon_bimpc >= consts.horizon_lompc) and (
            consts.horizon_lompc >= 1
        )
        assert consts.nrobots_per_bat_type >= 1
        assert consts.npartitions >= 1
        assert (len(consts.demand.shape) == 1) and (
            consts.demand.shape[0]
            >= consts.simulation_length + consts.horizon_bimpc + 1
        )
        # Set constants, initialize PriceCoordinators and BiMPC.
        self._set_constants(consts)
        self.bimpc = BiMPC(
            self.N_bi, self.Np, self.consts_bi, self.consts_s, self.consts_l
        )
        self.price_coord_s = PriceCoordinator(self.N_lo, self.consts_s, self.price_type)
        self.price_coord_l = PriceCoordinator(self.N_lo, self.consts_l, self.price_type)

        # Initialize state variables = (robot charge levels, energy stored).
        self._init_states()
        # Initialize logs.
        self._init_logs(consts)

    def _set_constants(self, consts: ChargingStationConstants) -> None:
        self.Tf = consts.simulation_length
        self.N_bi = consts.horizon_bimpc
        self.N_lo = consts.horizon_lompc
        self.nr = consts.nrobots_per_bat_type
        self.Np = consts.npartitions
        self.demand = consts.demand
        self.consts_bi = consts.bimpc_consts
        self.consts_s = consts.small_bat_consts
        self.consts_l = consts.large_bat_consts
        self.price_type = consts.price_type
        if self.price_type == "linear":
            self.Nr = 2 * self.N_lo
        else:
            self.Nr = 3 * self.N_lo
        # Range of initial charges of robots.
        self.s0_min = MIN_INITIAL_CHARGE
        self.s0_max = MAX_INITIAL_CHARGE
        self.s_s_rng = np.linspace(self.s0_min, self.consts_s.s_max, self.Np + 1)
        self.s_l_rng = np.linspace(self.s0_min, self.consts_l.s_max, self.Np + 1)
        # Total charge capacity of  robots.
        self.B = (self.consts_s.theta + self.consts_l.theta) * self.nr

    def _init_states(self) -> None:
        self.s_s = self.s0_min + (self.s0_max - self.s0_min) * np.random.random(
            (self.nr,)
        )
        self.s_l = self.s0_min + (self.s0_max - self.s0_min) * np.random.random(
            (self.nr,)
        )
        self.s_bat = 0  # Normalized.

        self.t = 0
        self.ncharged_s = 0
        self.ncharged_l = 0

        self.idx_s = np.zeros((self.nr,), dtype=int)
        self.idx_l = np.zeros((self.nr,), dtype=int)
        self._update_indices()

    def _update_indices(self) -> None:
        for p in range(self.Np):
            mask_s = (self.s_s >= self.s_s_rng[p]) & (self.s_s <= self.s_s_rng[p + 1])
            self.idx_s[mask_s] = p
            mask_l = (self.s_l >= self.s_l_rng[p]) & (self.s_l <= self.s_l_rng[p + 1])
            self.idx_l[mask_l] = p

    def _init_logs(self, consts: ChargingStationConstants) -> None:
        self.logs = {}
        self.logs["constants"] = consts
        self.logs["inputs"] = {
            "w_s": np.zeros((self.Np, self.Tf)),
            "w_l": np.zeros((self.Np, self.Tf)),
            "w_s_bimpc": np.zeros((self.Np, self.Tf)),
            "w_l_bimpc": np.zeros((self.Np, self.Tf)),
            "u_gen": np.zeros((self.Tf,)),
        }
        self.logs["states"] = {
            "xi_bat": np.zeros((self.Tf,)),
        }
        self.logs["bounds"] = {
            "beta_s": np.zeros((self.Np, self.Tf)),
            "beta_l": np.zeros((self.Np, self.Tf)),
        }
        self.logs["statistics"] = {
            "num_charged_s": 0,
            "num_charged_l": 0,
            "gamma_s": np.zeros((self.Np, self.Tf)),
            "gamma_l": np.zeros((self.Np, self.Tf)),
            "num_iter_s": np.zeros((self.Np, self.Tf), dtype=int),
            "num_iter_l": np.zeros((self.Np, self.Tf), dtype=int),
            "Nr_s": np.zeros((self.Np, self.Tf), dtype=int),
            "Nr_l": np.zeros((self.Np, self.Tf), dtype=int),
        }
        self.logs["prices"] = {
            "lmbd_r": np.zeros((self.Tf)),
            "avg_price_s": np.zeros((self.Np, self.Tf)),
            "avg_price_l": np.zeros((self.Np, self.Tf)),
            "price_red_s": np.zeros((self.Np, self.Tf)),
            "price_red_l": np.zeros((self.Np, self.Tf)),
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
        w_s_bi, w_l_bi, u_gen, stats_bi = self._get_bimpc_solution(lmbd_r)
        # Compute prices from PriceCoordinator.
        prices_s, prices_l, stats_s, stats_l = self._get_optimal_prices(
            w_s_bi, w_l_bi, lmbd_r
        )
        # Get w0 and price paid by LoMPCs.
        w0_s, w0_l, price0_s, price0_l = self._get_w0_price0(prices_s, prices_l, lmbd_r)
        # Update logs.
        nu = (w_s_bi, w_l_bi, u_gen, w0_s, w0_l)
        stats = (stats_bi, stats_s, stats_l)
        price0 = (price0_s, price0_l)
        self._update_logs(lmbd_r, nu, stats, price0)
        # Update state variables.
        self._update_state(w0_s, w0_l, u_gen[0])
        # Update time.
        self.t += 1

    def _get_bimpc_solution(
        self, lmbd_r: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        # (Optional): Check feasibility condition for kappa.
        # Set normalized BiMPC parameters.
        #   Set normalized Nr, beta, and gamma.
        Nr_s, Nr_l = np.zeros((self.Np,), dtype=int), np.zeros((self.Np,), dtype=int)
        beta_s, beta_l = np.zeros((self.Np,)), np.zeros((self.Np,))
        gamma_s, gamma_l = np.zeros((self.Np,)), np.zeros((self.Np,))
        for p in range(self.Np):
            mask_s = self.idx_s == p
            Nr_s[p] = mask_s.sum()
            if Nr_s[p] > 0:
                s0p_s = self.s_s[mask_s]
                self.price_coord_s.set_charge_levels(s0p_s)
                _, beta_s[p] = self.price_coord_s.get_robustness_bounds(lmbd_r)
                gamma_s[p] = self.price_coord_s.get_gamma()
            mask_l = self.idx_l == p
            Nr_l[p] = mask_l.sum()
            if Nr_l[p] > 0:
                s0p_l = self.s_l[mask_l]
                self.price_coord_l.set_charge_levels(s0p_l)
                _, beta_l[p] = self.price_coord_l.get_robustness_bounds(lmbd_r)
                gamma_l[p] = self.price_coord_l.get_gamma()
        Nr_s_ = Nr_s / self.B
        Nr_l_ = Nr_l / self.B
        #   Set normalized demand.
        demand = self.demand[self.t : self.t + self.N_bi] / self.B
        #   Normalized BiMPC parameters.
        bimpc_params = BiMPCParameters(
            Nr_s_, Nr_l_, beta_s, beta_l, gamma_s, gamma_l, self.s_bat, demand
        )
        # Compute BiMPC solution.
        w_s_bi, w_l_bi, u_gen = self.bimpc.solve_bimpc(bimpc_params)
        # Store statistics.
        stats_bi = {
            "Nr_s": Nr_s,
            "Nr_l": Nr_l,
            "beta_s": beta_s,
            "beta_l": beta_l,
            "gamma_s": gamma_s,
            "gamma_l": gamma_l,
        }
        if PRINT_LEVEL >= 1:
            total_w0 = (
                self.consts_s.theta * Nr_s_ @ w_s_bi[:, 0]
                + self.consts_l.theta * Nr_l_ @ w_l_bi[:, 0]
            )
            u0_bat_avg = u_gen[0] - demand[0] - total_w0
            u0_bat_err = (
                self.consts_s.theta * Nr_s_ @ beta_s
                + self.consts_l.theta * Nr_l_ @ beta_l
            )
            xi_bat_avg = self.s_bat + u0_bat_avg
            print(
                "Robot distribution (small): "
                + " + ".join("{:4d}".format(n) for n in Nr_s)
                + " = {:4d}".format(np.sum(Nr_s))
            )
            print(
                "Robot distribution (large): "
                + " + ".join("{:4d}".format(n) for n in Nr_l)
                + " = {:4d}".format(np.sum(Nr_l))
            )
            print(
                f"Energy generated          : {u_gen[0]:13.8f} | Max: {self.consts_bi.u_gen_max:13.8f}"
            )
            print(f"Demand                    : {demand[0]:13.8f}")
            print(f"Predicted output (robots) : {total_w0:13.8f}")
            print(
                f"Predicted battery input   : [{u0_bat_avg - u0_bat_err:8.5f}, {u0_bat_avg + u0_bat_err:8.5f}] | Max (mag): {self.consts_bi.u_bat_max:8.5f}"
            )
            print(f"Current battery state     : {self.s_bat}")
            print(
                f"Predicted battery state   : Min: 0 | [{xi_bat_avg - u0_bat_err:8.5f}, {xi_bat_avg + u0_bat_err:8.5f}] | Max: {self.consts_bi.xi_bat_max:8.5f}"
            )

            if PRINT_LEVEL >= 2:
                print("")
        return w_s_bi, w_l_bi, u_gen, stats_bi

    def _get_optimal_prices(
        self, w_s_bi: np.ndarray, w_l_bi: np.ndarray, lmbd_r: float
    ) -> tuple[np.ndarray, np.ndarray, list, list]:
        # Reduce the horizon for the LoMPC.
        w_s_opt, w_l_opt = w_s_bi[:, : self.N_lo], w_l_bi[:, : self.N_lo]
        prices_s, prices_l = np.zeros((self.Np, self.Nr)), np.zeros((self.Np, self.Nr))
        stats_s, stats_l = [], []
        for p in range(self.Np):
            s0p_s = self.s_s[self.idx_s == p]
            if len(s0p_s) > 0:
                self.price_coord_s.set_charge_levels(s0p_s)
                if PRINT_LEVEL >= 1:
                    print(f"Small robots, partition {p:2d}: ", end="")
                    if PRINT_LEVEL >= 2:
                        print("\n" + "-" * 27)
                lmbd_, stats_ = self.price_coord_s.compute_optimal_prices(
                    w_s_opt[p, :], lmbd_r
                )
                prices_s[p, :] = lmbd_[: self.Nr]
                stats_s.append(stats_)
                if PRINT_LEVEL >= 2:
                    print("")
            else:
                stats_s.append({})
            s0p_l = self.s_l[self.idx_l == p]
            if len(s0p_l) > 0:
                self.price_coord_l.set_charge_levels(s0p_l)
                if PRINT_LEVEL >= 1:
                    print(f"Large robots, partition {p:2d}: ", end="")
                    if PRINT_LEVEL >= 2:
                        print("\n" + "-" * 27)
                lmbd_, stats_ = self.price_coord_l.compute_optimal_prices(
                    w_l_opt[p, :], lmbd_r
                )
                prices_l[p, :] = lmbd_[: self.Nr]
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
        w0_s, w0_l = np.zeros((self.nr,)), np.zeros((self.nr,))
        price0_s, price0_l = np.zeros((self.Np,)), np.zeros((self.Np,))
        for p in range(self.Np):
            s0p_s = self.s_s[self.idx_s == p]
            if len(s0p_s) > 0:
                self.price_coord_s.set_charge_levels(s0p_s)
                w0_s[self.idx_s == p], price0_s[p] = self.price_coord_s.get_w0_price0(
                    prices_s[p, :], lmbd_r
                )
            s0p_l = self.s_l[self.idx_l == p]
            if len(s0p_l) > 0:
                self.price_coord_l.set_charge_levels(s0p_l)
                w0_l[self.idx_l == p], price0_l[p] = self.price_coord_l.get_w0_price0(
                    prices_l[p, :], lmbd_r
                )
        return w0_s, w0_l, price0_s, price0_l

    def _update_state(self, w0_s: np.ndarray, w0_l: np.ndarray, u0_gen: float) -> None:
        # Update robot charge states and indices.
        residual_charge = 0
        self.s_s += w0_s
        mask_s = self.s_s > MIN_FULL_CHARGE_FRACTION * self.consts_s.s_max
        residual_charge += self.consts_s.theta * np.sum(
            self.s_s[mask_s] - MIN_FULL_CHARGE_FRACTION * self.consts_s.s_max
        )
        self.s_s[mask_s] = self.s0_min + (self.s0_max - self.s0_min) * np.random.random(
            (mask_s.sum(),)
        )
        self.ncharged_s += mask_s.sum()
        self.s_l += w0_l
        mask_l = self.s_l > MIN_FULL_CHARGE_FRACTION * self.consts_l.s_max
        residual_charge += self.consts_l.theta * np.sum(
            self.s_l[mask_l] - MIN_FULL_CHARGE_FRACTION * self.consts_l.s_max
        )
        self.s_l[mask_l] = self.s0_min + (self.s0_max - self.s0_min) * np.random.random(
            (mask_l.sum(),)
        )
        self.ncharged_l += mask_l.sum()
        self._update_indices()
        if not ADD_RESIDUAL_CHARGE_TO_BATTERY:
            residual_charge = 0
        # Update battery charge state.
        u_bat = (
            u0_gen
            + (
                -self.consts_s.theta * np.sum(w0_s)
                - self.consts_l.theta * np.sum(w0_l)
                + residual_charge
                - self.demand[self.t]
            )
            / self.B
        )
        self.s_bat += u_bat
        if PRINT_LEVEL >= 1:
            print(f"# small robots charged    : {self.ncharged_s:5d}")
            print(f"# large robots charged    : {self.ncharged_l:5d}")
            print("")

    def _update_logs(
        self, lmbd_r: float, nu: tuple, stats: tuple, price0: tuple
    ) -> None:
        w_s_bi, w_l_bi, u_gen, w0_s, w0_l = nu
        stats_bi, stats_s, stats_l = stats
        price0_s, price0_l = price0
        # Log inputs.
        #   Log w_s and w_l.
        for p in range(self.Np):
            if len(w0_s[self.idx_s == p]) > 0:
                self.logs["inputs"]["w_s"][p, self.t] = np.mean(w0_s[self.idx_s == p])
            if len(w0_l[self.idx_l == p]) > 0:
                self.logs["inputs"]["w_l"][p, self.t] = np.mean(w0_l[self.idx_l == p])
        #   Log other inputs.
        self.logs["inputs"]["w_s_bimpc"][:, self.t] = w_s_bi[:, 0]
        self.logs["inputs"]["w_l_bimpc"][:, self.t] = w_l_bi[:, 0]
        self.logs["inputs"]["u_gen"][self.t] = u_gen[0]
        # Log battery state.
        self.logs["states"]["xi_bat"][self.t] = self.s_bat
        # Log bounds.
        self.logs["bounds"]["beta_s"][:, self.t] = stats_bi["beta_s"]
        self.logs["bounds"]["beta_l"][:, self.t] = stats_bi["beta_l"]
        # Log statistics.
        #   Update number of robots charged.
        self.logs["statistics"]["num_charged_s"] = self.ncharged_s
        self.logs["statistics"]["num_charged_l"] = self.ncharged_l
        #   Log gamma.
        self.logs["statistics"]["gamma_s"][:, self.t] = stats_bi["gamma_s"]
        self.logs["statistics"]["gamma_l"][:, self.t] = stats_bi["gamma_l"]
        #   Log number of iterations for convergence.
        for p in range(self.Np):
            if stats_s[p]:
                self.logs["statistics"]["num_iter_s"][p, self.t] = stats_s[p]["iter"]
            else:
                self.logs["statistics"]["num_iter_s"][p, self.t] = -1
            if stats_l[p]:
                self.logs["statistics"]["num_iter_l"][p, self.t] = stats_l[p]["iter"]
            else:
                self.logs["statistics"]["num_iter_l"][p, self.t] = -1
        #   Log Nr.
        self.logs["statistics"]["Nr_s"][:, self.t] = stats_bi["Nr_s"]
        self.logs["statistics"]["Nr_l"][:, self.t] = stats_bi["Nr_l"]
        # Log prices.
        #   Log lmbd_r.
        self.logs["prices"]["lmbd_r"][self.t] = lmbd_r
        #   Log average prices paid at time t.
        self.logs["prices"]["avg_price_s"][:, self.t] = price0_s
        self.logs["prices"]["avg_price_l"][:, self.t] = price0_l
        #   Log average price reduction from price regulation.
        for p in range(self.Np):
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
