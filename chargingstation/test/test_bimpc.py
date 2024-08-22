import matplotlib.pyplot as plt
import numpy as np

from chargingstation.bimpc import (BiMPC, BiMPCConstants, BiMPCParameters,
                                   BiMPCTrackingCost)
from chargingstation.demand_data import medium_term_demand_forecast
from chargingstation.lompc import LoMPCConstants


def _get_lompc_constants() -> tuple[LoMPCConstants, LoMPCConstants]:
    theta_s = 10
    w_max_s = 0.25
    theta_l = 50
    w_max_l = 0.15
    consts_s = LoMPCConstants(0.05, theta_s, 0.9, w_max_s, "small")
    consts_l = LoMPCConstants(0.025, theta_l, 0.9, w_max_l, "large")
    return consts_s, consts_l


def _get_normalized_bimpc_consts(
    tracking_cost_type: BiMPCTrackingCost,
) -> BiMPCConstants:
    delta = 1e3
    c_gen = 1
    u_gen_max = 1.5
    u_bat_max = 0.3
    xi_bat_max = 1.5
    exp_rate = 5
    tracking_cost_type = tracking_cost_type
    return BiMPCConstants(
        delta, c_gen, u_gen_max, u_bat_max, xi_bat_max, tracking_cost_type, exp_rate
    )


def _get_random_probability_simplex_vector(n: int) -> np.ndarray:
    v = np.random.random((n,)) + 1e-6
    return v / np.sum(v)


def _get_normalized_bimpc_parameters(
    N: int,
    Np: int,
    nr_s: int,
    nr_l: int,
    consts_s: LoMPCConstants,
    consts_l: LoMPCConstants,
    random_Nr: bool,
    random_gamma: bool,
    early_peak_demand: bool,
) -> BiMPCParameters:
    B = consts_s.theta * nr_s + consts_l.theta * nr_l
    if random_Nr:
        Nr_s = nr_s * _get_random_probability_simplex_vector(Np) / B
        Nr_l = nr_l * _get_random_probability_simplex_vector(Np) / B
    else:
        Nr_s = nr_s * np.ones((Np,)) / (Np * B)
        Nr_l = nr_l * np.ones((Np,)) / (Np * B)
    beta_s = np.sqrt(N) * 0.3 / Np * np.ones((Np,))
    beta_l = np.sqrt(N) * 0.3 / Np * np.ones((Np,))
    if random_gamma:
        gamma_s = 0.6 * np.random.random((Np,))
        gamma_l = 0.6 * np.random.random((Np,))
    else:
        gamma_s = 0.6 * np.ones((Np,))
        gamma_l = 0.6 * np.ones((Np,))
    xi0_bat = 0
    if early_peak_demand:
        demand = medium_term_demand_forecast(24 + N, 1 / 4, interpolate=False) / B
        demand = demand[17 : 17 + N]
    else:
        demand = medium_term_demand_forecast(N, 1 / 4, interpolate=False) / B
    # demand = 0.5 * np.array([0] * (N // 2) + [1] * (N - N // 2))
    return BiMPCParameters(
        Nr_s, Nr_l, beta_s, beta_l, gamma_s, gamma_l, xi0_bat, demand
    )


def _test_random_robot_distributions(
    random_Nr: bool, random_gamma: bool, early_peak_demand: bool
) -> None:
    N = 24
    Np = 12
    nr_s = 500
    nr_l = 500
    consts_s, consts_l = _get_lompc_constants()
    consts = _get_normalized_bimpc_consts(
        tracking_cost_type=BiMPCTrackingCost.EXP_UNWEIGHTED
    )
    params = _get_normalized_bimpc_parameters(
        N,
        Np,
        nr_s,
        nr_l,
        consts_s,
        consts_l,
        random_Nr,
        random_gamma,
        early_peak_demand,
    )
    print(f"Small robot distribution: {np.round(params.Nr_s / np.sum(params.Nr_s), 4)}")
    print(f"Large robot distribution: {np.round(params.Nr_l / np.sum(params.Nr_l), 4)}")
    print(f"Gamma (small): {np.round(params.gamma_s, 4)}")
    print(f"Gamma (large): {np.round(params.gamma_l, 4)}")

    bimpc = BiMPC(N, Np, consts, consts_s, consts_l)
    w_s_opt, w_l_opt, u_gen_opt = bimpc.solve_bimpc(params)
    horizons_ = (N, Np)
    constants_ = (consts_s, consts_l, consts, params)
    results_ = (w_s_opt, w_l_opt, u_gen_opt)
    _plot_figure(*horizons_, *constants_, bimpc, *results_)


def _plot_figure(
    N: int,
    Np: int,
    consts_s: LoMPCConstants,
    consts_l: LoMPCConstants,
    consts: BiMPCConstants,
    params: BiMPCParameters,
    bimpc: BiMPC,
    w_s_opt: np.ndarray,
    w_l_opt: np.ndarray,
    u_gen_opt: np.ndarray,
) -> None:
    _, ax = plt.subplots(3, 2)
    A = bimpc.get_bat_input_mat()
    # Plots.
    t = np.arange(N)
    #   w_s_opt.
    for p in range(Np):
        ax[0, 0].plot(t, w_s_opt[p, :])
    ax[0, 0].plot(t, consts_s.w_max * np.ones((N,)), "--")
    ax[0, 0].set_title("Optimal energy output (small)")
    #   cumulative w_s_opt: y_s_opt / gamma_s.
    for p in range(Np):
        ax[0, 1].plot(t, A @ w_s_opt[p, :] / params.gamma_s[p])
    ax[0, 1].plot(t, np.ones((N,)), "--")
    ax[0, 1].set_title("Optimal battery charge (small)")
    #   w_l_opt.
    for p in range(Np):
        ax[1, 0].plot(t, w_l_opt[p, :])
    ax[1, 0].plot(t, consts_l.w_max * np.ones((N,)), "--")
    ax[1, 0].set_title("Optimal energy output (large)")
    #   cumulate w_l_opt: y_l_opt / gamma_l.
    for p in range(Np):
        ax[1, 1].plot(t, A @ w_l_opt[p, :] / params.gamma_l[p])
    ax[1, 1].plot(t, np.ones((N,)), "--")
    ax[1, 1].set_title("Optimal battery charge (large)")
    #   Total enery output (+ error), external demand.
    w_total = (
        consts_s.theta * params.Nr_s @ w_s_opt + consts_l.theta * params.Nr_l @ w_l_opt
    )
    robustness_bound = (
        consts_s.theta * params.Nr_s @ params.beta_s
        + consts_l.theta * params.Nr_l @ params.beta_l
    )
    w_total_min = w_total_max = w_total
    w_total_min[0] -= robustness_bound
    w_total_max[0] += robustness_bound
    w_max = (
        consts_s.theta * np.sum(params.Nr_s) * consts_s.w_max
        + consts_l.theta * np.sum(params.Nr_l) * consts_l.w_max
    )
    ax[2, 0].plot(t, params.demand, "-b", label="external demand")
    ax[2, 0].plot(t, w_total, "-r", label="total output")
    ax[2, 0].fill_between(t, w_total_min, w_total_max, alpha=0.2, color="r")
    ax[2, 0].plot(t, w_max * np.ones((N,)), "--r")
    ax[2, 0].legend()
    ax[2, 0].set_title("Total output, external demand")
    #   Combined energy output (+ error), energy generated, battery storage (+error).
    en_comb = w_total + params.demand
    xi_bat_avg_opt = A @ (
        u_gen_opt
        - params.demand
        - consts_s.theta * params.Nr_s @ w_s_opt
        - consts_l.theta * params.Nr_l @ w_l_opt
    )
    xi_bat_err = np.ones((N,)) * (
        consts_s.theta * params.Nr_s @ params.beta_s
        + consts_l.theta * params.Nr_l @ params.beta_l
    )
    ax[2, 1].plot(t, en_comb, "-r", label="combined energy output")
    ax[2, 1].plot(t, u_gen_opt, "-g", label="energy generated")
    ax[2, 1].plot(t, consts.u_gen_max * np.ones((N,)), "--g")
    ax[2, 1].plot(t, xi_bat_avg_opt, "-b", label="storage battery charge")
    ax[2, 1].fill_between(
        t,
        xi_bat_avg_opt - xi_bat_err,
        xi_bat_avg_opt + xi_bat_err,
        alpha=0.2,
        color="b",
    )
    ax[2, 1].plot(t, consts.xi_bat_max * np.ones((N,)), "--b")
    ax[2, 1].legend()
    ax[2, 1].set_title(
        "Combined energy output, energy generated, storage battery charge"
    )

    plt.show()


def main() -> None:
    _test_random_robot_distributions(
        random_Nr=True, random_gamma=True, early_peak_demand=True
    )


if __name__ == "__main__":
    main()
