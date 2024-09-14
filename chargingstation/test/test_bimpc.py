import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from chargingstation.bimpc import (BiMPC, BiMPCChargingCostType,
                                   BiMPCConstants, BiMPCParameters)
from chargingstation.demand_data import medium_term_demand_forecast
from chargingstation.lompc import LoMPCConstants

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "STIXGeneral"


def _get_lompc_constants() -> tuple[LoMPCConstants, LoMPCConstants]:
    theta_s = 10
    w_max_s = 0.25
    theta_l = 50
    w_max_l = 0.15
    consts_s = LoMPCConstants(0.05, theta_s, 0.9, w_max_s, "small")
    consts_l = LoMPCConstants(0.025, theta_l, 0.9, w_max_l, "large")
    return consts_s, consts_l


def _get_normalized_bimpc_consts(
    charging_cost_type: BiMPCChargingCostType,
) -> BiMPCConstants:
    delta = 1e3
    c_g = 1
    u_g_max = 1.5
    u_b_max = 0.3
    x_max = 1.5
    exp_rate = 5
    charging_cost_type = charging_cost_type
    return BiMPCConstants(
        delta, c_g, u_g_max, u_b_max, x_max, charging_cost_type, exp_rate
    )


def _get_random_probability_simplex_vector(n: int) -> np.ndarray:
    v = np.random.random((n,)) + 1e-6
    return v / np.sum(v)


def _get_normalized_bimpc_parameters(
    N: int,
    P: int,
    M_s: int,
    M_l: int,
    consts_s: LoMPCConstants,
    consts_l: LoMPCConstants,
    random_Mp: bool,
    random_gamma: bool,
    early_peak_demand: bool,
) -> BiMPCParameters:
    B = consts_s.theta * M_s + consts_l.theta * M_l
    if random_Mp:
        Mp_s = M_s * _get_random_probability_simplex_vector(P) / B
        Mp_l = M_l * _get_random_probability_simplex_vector(P) / B
    else:
        Mp_s = M_s * np.ones((P,)) / (P * B)
        Mp_l = M_l * np.ones((P,)) / (P * B)
    beta_s = np.sqrt(N) * 0.3 / P * np.ones((P,))
    beta_l = np.sqrt(N) * 0.3 / P * np.ones((P,))
    if random_gamma:
        gamma_sm = 0.6 * np.random.random((P,))
        gamma_lm = 0.6 * np.random.random((P,))
    else:
        gamma_sm = 0.6 * np.ones((P,))
        gamma_lm = 0.6 * np.ones((P,))
    x0 = 0
    if early_peak_demand:
        demand = medium_term_demand_forecast(24 + N, 1 / 4, interpolate=False) / B
        demand = demand[17 : 17 + N]
    else:
        demand = medium_term_demand_forecast(N, 1 / 4, interpolate=False) / B
    # demand = 0.5 * np.array([0] * (N // 2) + [1] * (N - N // 2))
    return BiMPCParameters(Mp_s, Mp_l, beta_s, beta_l, gamma_sm, gamma_lm, x0, demand)


def _test_random_EV_distributions(
    random_Mp: bool, random_gamma: bool, early_peak_demand: bool
) -> None:
    N = 24
    P = 12
    M_s = 500
    M_l = 500
    consts_s, consts_l = _get_lompc_constants()
    consts_bi = _get_normalized_bimpc_consts(
        charging_cost_type=BiMPCChargingCostType.EXP_UNWEIGHTED
    )
    params = _get_normalized_bimpc_parameters(
        N,
        P,
        M_s,
        M_l,
        consts_s,
        consts_l,
        random_Mp,
        random_gamma,
        early_peak_demand,
    )
    print(f"Small EV distribution: {np.round(params.Mp_s / np.sum(params.Mp_s), 4)}")
    print(f"Large EV distribution: {np.round(params.Mp_l / np.sum(params.Mp_l), 4)}")
    print(f"Avg gamma (small): {np.round(params.gamma_sm, 4)}")
    print(f"Avg gamma (large): {np.round(params.gamma_lm, 4)}")

    bimpc = BiMPC(N, P, consts_bi, consts_s, consts_l)
    w_hat_s_opt, w_hat_l_opt, u_g_opt = bimpc.solve_bimpc(params)
    horizons_ = (N, P)
    constants_ = (consts_s, consts_l, consts_bi, params)
    results_ = (w_hat_s_opt, w_hat_l_opt, u_g_opt)
    _plot_figure(*horizons_, *constants_, bimpc, *results_)


def _plot_figure(
    N: int,
    P: int,
    consts_s: LoMPCConstants,
    consts_l: LoMPCConstants,
    consts_bi: BiMPCConstants,
    params: BiMPCParameters,
    bimpc: BiMPC,
    w_hat_s_opt: np.ndarray,
    w_hat_l_opt: np.ndarray,
    u_g_opt: np.ndarray,
) -> None:
    _, ax = plt.subplots(3, 2, sharex=True, layout="constrained")
    A = bimpc.get_bat_input_mat()
    # Plots.
    t = np.arange(N)
    #   w_hat_s_opt.
    for p in range(P):
        ax[0, 0].plot(t, w_hat_s_opt[p, :])
    ax[0, 0].plot(t, consts_s.w_max * np.ones((N,)), "--")
    ax[0, 0].set_title("Team-optimal electricity output (small EV)")
    #   cumulative w_hat_s_opt: (y_hat_s_opt - y0) / gamma_sm.
    for p in range(P):
        ax[0, 1].plot(t, A @ w_hat_s_opt[p, :] / params.gamma_sm[p])
    ax[0, 1].plot(t, np.ones((N,)), "--")
    ax[0, 1].set_title("Team-optimal battery charge remaining (small EV)")
    #   w_hat_l_opt.
    for p in range(P):
        ax[1, 0].plot(t, w_hat_l_opt[p, :])
    ax[1, 0].plot(t, consts_l.w_max * np.ones((N,)), "--")
    ax[1, 0].set_title("Team-optimal electricity output (large EV)")
    #   cumulate w_hat_l_opt: (y_hat_l_opt - y0) / gamma_lm.
    for p in range(P):
        ax[1, 1].plot(t, A @ w_hat_l_opt[p, :] / params.gamma_lm[p])
    ax[1, 1].plot(t, np.ones((N,)), "--")
    ax[1, 1].set_title("Team-optimal battery charge remaining (large EV)")
    #   Total EV electricity consumption (+ error), external demand.
    w_hat_opt = (
        consts_s.theta * params.Mp_s @ w_hat_s_opt
        + consts_l.theta * params.Mp_l @ w_hat_l_opt
    )
    error_bound = (
        consts_s.theta * params.Mp_s @ params.beta_s
        + consts_l.theta * params.Mp_l @ params.beta_l
    )
    w_hat_min, w_hat_max = w_hat_opt, w_hat_opt
    w_hat_min[0] -= error_bound
    w_hat_max[0] += error_bound
    w_max = (
        consts_s.theta * np.sum(params.Mp_s) * consts_s.w_max
        + consts_l.theta * np.sum(params.Mp_l) * consts_l.w_max
    )
    ax[2, 0].plot(t, params.demand, "-b", label="external demand")
    ax[2, 0].plot(t, w_hat_opt, "-r", label=r"$\hat{w}^*$")
    ax[2, 0].fill_between(t, w_hat_min, w_hat_max, alpha=0.2, color="r")
    ax[2, 0].plot(t, w_max * np.ones((N,)), "--r")
    ax[2, 0].legend()
    ax[2, 0].set_title("Aggregate EV electricity consumption, external demand")
    #   Total electricity output (+ error), electricity generated, battery storage (+error).
    supply = w_hat_opt + params.demand
    x_hat_opt = A @ (
        u_g_opt
        - params.demand
        - consts_s.theta * params.Mp_s @ w_hat_s_opt
        - consts_l.theta * params.Mp_l @ w_hat_l_opt
    )
    delta_err = (
        consts_s.theta * params.Mp_s @ params.beta_s
        + consts_l.theta * params.Mp_l @ params.beta_l
    )
    ax[2, 1].plot(t, supply, "-r", label="total supply")
    ax[2, 1].plot(t, u_g_opt, "-g", label=r"$u^{g*}$")
    ax[2, 1].plot(t, consts_bi.u_g_max * np.ones((N,)), "--g")
    ax[2, 1].plot(t, x_hat_opt, "-b", label=r"$\hat{x}^*$")
    ax[2, 1].fill_between(
        t,
        x_hat_opt - delta_err * np.ones((N,)),
        x_hat_opt + delta_err * np.ones((N,)),
        alpha=0.2,
        color="b",
    )
    ax[2, 1].plot(t, consts_bi.x_max * np.ones((N,)), "--b")
    ax[2, 1].legend()
    ax[2, 1].set_title("Total supply, electricity generated, storage battery charge")

    plt.show()


def main() -> None:
    _test_random_EV_distributions(
        random_Mp=True, random_gamma=True, early_peak_demand=True
    )


if __name__ == "__main__":
    main()
