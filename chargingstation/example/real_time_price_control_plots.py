import pickle

import matplotlib.pyplot as plt
import numpy as np


def _plot_graphs(logs: dict) -> None:
    consts = logs["constants"]
    consts_bi = consts.bimpc_consts
    consts_s, consts_l = consts.small_bat_consts, consts.large_bat_consts
    Tf = consts.simulation_length - 1
    nr = consts.nrobots_per_bat_type
    B = (consts_s.theta + consts_l.theta) * nr
    demand = consts.demand[: Tf + 1]

    # Robot distributions.
    Nr_s, Nr_l = logs["statistics"]["Nr_s"], logs["statistics"]["Nr_l"]
    # Robustness bounds.
    beta_s, beta_l = logs["bounds"]["beta_s"], logs["bounds"]["beta_l"]
    # Number of robots charged.
    ncharged_s, ncharged_l = (
        logs["statistics"]["num_charged_s"],
        logs["statistics"]["num_charged_l"],
    )

    # Energy generated.
    u_gen = logs["inputs"]["u_gen"]
    # (Actual / predicted) robot consumptions.
    w_s, w_l = logs["inputs"]["w_s"], logs["inputs"]["w_l"]
    w_s_bi, w_l_bi = logs["inputs"]["w_s_bimpc"], logs["inputs"]["w_l_bimpc"]
    # Total normalized (actual / predicted) energy consumption by robots (and errors).
    total_w_s = consts_s.theta * np.sum(Nr_s * w_s, axis=0) / B
    total_w_l = consts_l.theta * np.sum(Nr_l * w_l, axis=0) / B
    total_w_s_bi = consts_s.theta * np.sum(Nr_s * w_s_bi, axis=0) / B
    total_w_l_bi = consts_l.theta * np.sum(Nr_l * w_l_bi, axis=0) / B
    err_w_s = consts_s.theta * np.sum(Nr_s * beta_s, axis=0) / B
    err_w_l = consts_l.theta * np.sum(Nr_l * beta_l, axis=0) / B

    # Normalized (actual / predicted) battery charge and input.
    u_bat_max = consts_bi.u_bat_max
    xi_bat_max = consts_bi.xi_bat_max
    xi_bat = logs["states"]["xi_bat"]
    xi0_bat = xi_bat[0]
    u_bat = xi_bat[1:] - xi_bat[:-1]
    u_bat_bi = u_gen - demand / B - total_w_s_bi - total_w_l_bi
    xi_bat_bi = np.zeros((Tf + 1,))
    xi_bat_bi[0] = xi0_bat
    xi_bat_bi[1:] = xi_bat[:-1] + u_bat_bi[:-1]

    # Prices.
    avg_price_s = np.sum(Nr_s * logs["prices"]["avg_price_s"], axis=0) / nr
    avg_price_l = np.sum(Nr_l * logs["prices"]["avg_price_s"], axis=0) / nr
    price_red_s = np.nan_to_num(np.zeros_like(logs["prices"]["price_red_s"]))
    price_red_l = np.nan_to_num(np.zeros_like(logs["prices"]["price_red_l"]))
    price_before_reg_s = avg_price_s + np.sum(Nr_s * price_red_s, axis=0) / nr
    price_before_reg_l = avg_price_l + np.sum(Nr_l * price_red_l, axis=0) / nr

    ### Statistics.
    print(f"# small robots charged: {ncharged_s}")
    print(f"# large robots charged: {ncharged_l}")

    ### Plots.
    t = np.arange(Tf)

    _, ax = plt.subplots(2, 2)
    # w_s with error.
    ax[0][0].plot(t, total_w_s[:Tf], "-b", label="actual w_s")
    ax[0][0].plot(t, total_w_s_bi[:Tf], "-r", label="pred w_s")
    total_w_s_min = np.max(((total_w_s_bi - err_w_s)[:Tf], np.zeros((Tf,))), axis=0)
    total_w_s_max = np.min(
        (
            (total_w_s_bi + err_w_s)[:Tf],
            nr * consts_s.w_max * consts_s.theta / B * np.ones((Tf)),
        ),
        axis=0,
    )
    ax[0][0].fill_between(t, total_w_s_min, total_w_s_max, alpha=0.2, color="r")
    ax[0][0].plot(t, nr * consts_s.w_max * consts_s.theta / B * np.ones((Tf,)), "--b")
    ax[0][0].legend()

    # w_l with error.
    ax[0][1].plot(t, total_w_l[:Tf], "-b", label="actual w_l")
    ax[0][1].plot(t, total_w_l_bi[:Tf], "-r", label="pred w_l")
    total_w_l_min = np.max(((total_w_l_bi - err_w_l)[:Tf], np.zeros((Tf,))), axis=0)
    total_w_l_max = np.min(
        (
            (total_w_l_bi + err_w_l)[:Tf],
            nr * consts_l.w_max * consts_l.theta / B * np.ones((Tf)),
        ),
        axis=0,
    )
    ax[0][1].fill_between(t, total_w_l_min, total_w_l_max, alpha=0.2, color="r")
    ax[0][1].plot(t, nr * consts_l.w_max * consts_l.theta / B * np.ones((Tf,)), "--b")
    ax[0][1].legend()

    # Energy generated, demand, and battery state.
    ax[1][0].plot(t, np.zeros((Tf,)), "-.y")
    ax[1][0].plot(t, demand[:Tf] / B, "-r", label="demand")
    ax[1][0].plot(t, u_gen[:Tf], "-b", label="u_gen")
    ax[1][0].plot(t, (total_w_s + total_w_l)[:Tf], "-m", label="w")
    # ax[1][0].plot(t, u_bat[:Tf], "-k", label="u_bat")
    # ax[1][0].plot(t, u_bat_bi[:Tf], "--k", label="u_bat")
    u_bat_bi_min = np.max(
        (u_bat_bi[:Tf] - err_w_s[:Tf] - err_w_l[:Tf], -u_bat_max * np.ones((Tf,))),
        axis=0,
    )
    u_bat_bi_max = np.min(
        (u_bat_bi[:Tf] + err_w_s[:Tf] + err_w_l[:Tf], u_bat_max * np.ones((Tf,))),
        axis=0,
    )
    # ax[1][0].fill_between(t, u_bat_bi_min, u_bat_bi_max, alpha=0.2, color="grey")
    ax[1][0].plot(t, xi_bat[:Tf], "-g", label="xi_bat")
    ax[1][0].plot(t, xi_bat_bi[:Tf], "--g")
    xi_bat_bi_min = np.max(
        ((xi_bat_bi - err_w_s - err_w_l)[:Tf], np.zeros((Tf))), axis=0
    )
    xi_bat_bi_max = np.min(
        ((xi_bat_bi + err_w_s + err_w_l)[:Tf], xi_bat_max * np.ones((Tf,))), axis=0
    )
    ax[1][0].fill_between(t, xi_bat_bi_min, xi_bat_bi_max, alpha=0.2, color="g")
    ax[1][0].legend()

    # Average prices paid.
    ax[1][1].plot(t, avg_price_s[:Tf], "-r", label="price_s")
    # ax[1][1].plot(t, price_before_reg_s[: Tf], "--r")
    ax[1][1].plot(t, avg_price_l[:Tf], "-b", label="price_l")
    # ax[1][1].plot(t, price_before_reg_l[: Tf], "--b")
    ax[1][1].legend()

    plt.show()


def main() -> None:
    price_type = "linear-convex"
    # price_type = "linear"
    file_name = (
        "chargingstation/example/real-time-price-control_logs_" + price_type + ".pkl"
    )
    with open(file_name, "rb") as file:
        logs = pickle.load(file)
    _plot_graphs(logs)


if __name__ == "__main__":
    main()
