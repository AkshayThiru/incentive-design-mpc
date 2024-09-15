import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from chargingstation.charging_station import ChargingStationConstants
from chargingstation.settings import MIN_INITIAL_SOC

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

COL_WIDTH = 3.54  # [inch].
FIG_DPI = 200
SAVE_DPI = 1000  # >= 600.
SAVE_FIG = True

# mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "STIXGeneral"
# mpl.rcParams["font.family"] = "cmr10"


def _plot_graphs(logs: dict) -> None:
    consts: ChargingStationConstants = logs["constants"]
    consts_bi = consts.bimpc_consts
    consts_s = consts.small_EV_consts
    consts_l = consts.large_EV_consts
    Tf = consts.simulation_length - 1
    M_2 = consts.nEVs_per_EV_type
    B = (consts_s.theta + consts_l.theta) * M_2
    demand = consts.demand[: Tf + 1]

    t = np.arange(Tf)

    # EV distributions.
    Mp_s, Mp_l = logs["statistics"]["Mp_s"], logs["statistics"]["Mp_l"]
    # Robustness bounds.
    beta_s, beta_l = logs["bounds"]["beta_s"], logs["bounds"]["beta_l"]
    # Number of EVs charged.
    ncharged_s, ncharged_l = (
        logs["statistics"]["ncharged_s"],
        logs["statistics"]["ncharged_l"],
    )
    # Price solver iterations.
    niter_s, niter_l = (
        logs["statistics"]["niter_s"],
        logs["statistics"]["niter_l"],
    )
    niter_s = niter_s[niter_s >= 1]
    niter_l = niter_l[niter_l >= 1]

    # Electricity generated.
    u_g_max = consts_bi.u_g_max
    u_g = logs["inputs"]["u_g"]
    # (Actual / predicted) EV electricity consumptions.
    w_s, w_l = logs["inputs"]["w_s"], logs["inputs"]["w_l"]
    w_hat_s, w_hat_l = logs["inputs"]["w_hat_s"], logs["inputs"]["w_hat_l"]
    # Total normalized (actual / predicted) electricity consumption by EVs (and errors).
    total_w_s = consts_s.theta * np.sum(Mp_s * w_s, axis=0) / B
    total_w_l = consts_l.theta * np.sum(Mp_l * w_l, axis=0) / B
    total_w_hat_s = consts_s.theta * np.sum(Mp_s * w_hat_s, axis=0) / B
    total_w_hat_l = consts_l.theta * np.sum(Mp_l * w_hat_l, axis=0) / B
    err_w_s = consts_s.theta * np.sum(Mp_s * beta_s, axis=0) / B
    err_w_l = consts_l.theta * np.sum(Mp_l * beta_l, axis=0) / B
    total_w_hat_s_min = np.max(
        ((total_w_hat_s - err_w_s)[:Tf], np.zeros((Tf,))), axis=0
    )
    total_w_hat_s_max = np.min(
        (
            (total_w_hat_s + err_w_s)[:Tf],
            M_2 * consts_s.w_max * consts_s.theta / B * np.ones((Tf)),
        ),
        axis=0,
    )
    total_w_hat_l_min = np.max(
        ((total_w_hat_l - err_w_l)[:Tf], np.zeros((Tf,))), axis=0
    )
    total_w_hat_l_max = np.min(
        (
            (total_w_hat_l + err_w_l)[:Tf],
            M_2 * consts_l.w_max * consts_l.theta / B * np.ones((Tf)),
        ),
        axis=0,
    )
    total_w = total_w_s + total_w_l
    total_w_hat = total_w_hat_s + total_w_hat_l
    total_w_hat_min = total_w_hat_s_min + total_w_hat_l_min
    total_w_hat_max = total_w_hat_s_max + total_w_hat_l_max
    total_w_max = (
        M_2
        * (consts_s.w_max * consts_s.theta + consts_l.w_max * consts_l.theta)
        / B
        * np.ones((Tf,))
    )

    # Normalized (actual / predicted) battery charge and input.
    u_b_max = consts_bi.u_b_max
    x_max = consts_bi.x_max
    x = logs["states"]["x"]
    x0 = x[0]
    u_b = x[1:] - x[:-1]
    u_hat_b = u_g - demand / B - total_w_hat_s - total_w_hat_l
    x_hat = np.zeros((Tf + 1,))
    x_hat[0] = x0
    x_hat[1:] = x[:-1] + u_hat_b[:-1]
    u_hat_b_bi_min = np.max(
        (u_hat_b[:Tf] - err_w_s[:Tf] - err_w_l[:Tf], -u_b_max * np.ones((Tf,))),
        axis=0,
    )
    u_hat_b_max = np.min(
        (u_hat_b[:Tf] + err_w_s[:Tf] + err_w_l[:Tf], u_b_max * np.ones((Tf,))),
        axis=0,
    )
    x_hat_min = np.max(((x_hat - err_w_s - err_w_l)[:Tf], np.zeros((Tf))), axis=0)
    x_hat_max = np.min(
        ((x_hat + err_w_s + err_w_l)[:Tf], x_max * np.ones((Tf,))), axis=0
    )

    # Prices.
    avg_price_s = np.sum(Mp_s * logs["prices"]["avg_price_s"], axis=0) / M_2
    avg_price_l = np.sum(Mp_l * logs["prices"]["avg_price_s"], axis=0) / M_2
    price_red_s = np.nan_to_num(np.zeros_like(logs["prices"]["price_red_s"]))
    price_red_l = np.nan_to_num(np.zeros_like(logs["prices"]["price_red_l"]))
    price_before_reg_s = avg_price_s + np.sum(Mp_s * price_red_s, axis=0) / M_2
    price_before_reg_l = avg_price_l + np.sum(Mp_l * price_red_l, axis=0) / M_2

    ### Statistics.
    ncharged_s_max = M_2 * (Tf - 1) / ((consts_s.y_max - MIN_INITIAL_SOC) / consts_s.w_max)
    ncharged_l_max = M_2 * (Tf - 1) / ((consts_l.y_max - MIN_INITIAL_SOC) / consts_l.w_max)
    print(f"# small robots charged: {ncharged_s:6d}, ({100 * ncharged_s / ncharged_s_max:6.2f}%)")
    print(f"# large robots charged: {ncharged_l:6d}, ({100 * ncharged_l / ncharged_l_max:6.2f}%)")

    print(f"Average # iterations for small robots: {np.mean(niter_s):7.2f}")
    print(f"Average # iterations for large robots: {np.mean(niter_l):7.2f}")

    ### Plotting.
    # Figure properties.
    font_size = 8
    font_dict = {
        "fontsize": font_size,  # [pt]
        "fontstyle": "normal",
        "fontweight": "normal",
    }
    axis_margins = 0.05

    # Input plots.
    # Charging rate with error.
    fig_height = 1.4  # [inch].
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(COL_WIDTH, fig_height),
        dpi=FIG_DPI,
        sharey=False,
        layout="constrained",
    )
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.plot(t, total_w[:Tf], "-b", lw=1, label=r"$w$")
    ax.plot(t, total_w_hat[:Tf], "--r", lw=1, label=r"$\hat{w}$")
    ax.fill_between(
        t,
        total_w_hat_min,
        total_w_hat_max,
        alpha=0.1,
        color="r",
        lw=1,
        label="error\nbound",
    )
    ax.plot(t, total_w_max, "-.b", lw=1, label=r"$w_{\text{max}}$")
    ax.grid(axis="y", lw=0.25, alpha=0.5)
    ax.set_xlabel(r"time $\ (\text{hrs})$", **font_dict)
    ax.set_ylabel(r"normalized aggregate" "\n" r"EV charging rate $\ ()$", **font_dict)
    ax.legend(
        loc="center right",
        fontsize=font_size,
        framealpha=0.5,
        fancybox=False,
        edgecolor="black",
        labelspacing=0.15,
    )
    ax.set_xticks([0, 12, 24, 36, 47])
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    ax.margins(axis_margins, axis_margins)

    plt.show()

    if SAVE_FIG:
        fig.savefig(
            DIR_PATH + "/../plots/aggregate_ev_charging_rate.png", dpi=SAVE_DPI
        )  # , bbox_inches='tight')

    # Electricity generated and demand.
    fig_height = 1.5  # [inch].
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(COL_WIDTH, fig_height),
        dpi=FIG_DPI,
        sharey=False,
        layout="constrained",
    )
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.plot(t, u_g[:Tf], "-b", lw=1, label=r"$u^{\text{g}}$", zorder=3)
    ax.plot(
        t,
        u_g_max * np.ones((Tf,)),
        "-.b",
        lw=1,
        label=r"$u^{\text{g}}_\text{max}$",
        zorder=2,
    )
    ax.plot(t, demand[:Tf] / B, "--r", lw=1, label="external\ndemand", zorder=1)
    # ax.plot(t, total_w[:Tf], "-m", lw=1, label=r"$w$")
    # ax.plot(t, total_w_max[:Tf], "-.m", lw=1, label=r"$w_\text{max}$")
    # ax.plot(t, u_b[:Tf], "-k", lw=1, label=r"$u^b$")
    # ax.plot(t, u_hat_b[:Tf], "--k", lw=1, label=r"$\hat{u}^b$")
    # ax.fill_between(t, u_hat_b_min, u_hat_b_max, alpha=0.2, color="grey", lw=1)
    # ax.plot(t, x[:Tf], "-g", lw=1, label=r"$x$")
    # ax.plot(t, x_hat[:Tf], "--m", lw=1, label=r"$\hat{x}$")
    # ax.plot(t, x_max * np.ones((Tf,)), "-.g", lw=1, label=r"$x_\text{max}$")
    # ax.fill_between(t, x_hat_b_min, x_hat_b_max, alpha=0.2, color="g", lw=1)
    ax.grid(axis="y", lw=0.25, alpha=0.5)
    ax.set_xlabel(r"time $\ (\text{hrs})$", **font_dict)
    ax.set_ylabel(
        r"normalized demand and" "\n" r"energy generation $\ ()$", **font_dict
    )
    leg = ax.legend(
        loc="upper right",
        bbox_to_anchor=(1, 0.98),
        ncol=2,
        fontsize=font_size,
        framealpha=0.5,
        fancybox=False,
        edgecolor="black",
    )
    leg.legend_handles[0].set_ydata([6] * 3)
    leg.texts[0].set_y(60)
    leg.legend_handles[1].set_ydata([6] * 3)
    leg.texts[1].set_y(60)
    leg._legend_box.set_height(300)
    ax.set_xticks([0, 12, 24, 36, 47])
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    ax.margins(axis_margins, axis_margins)

    plt.show()

    if SAVE_FIG:
        fig.savefig(
            DIR_PATH + "/../plots/demand_energy_generation.png", dpi=SAVE_DPI
        )  # , bbox_inches='tight')

    # Storage battery state.
    fig_height = 1.5  # [inch].
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(COL_WIDTH, fig_height),
        dpi=FIG_DPI,
        sharey=False,
        layout="constrained",
    )
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    # ax.plot(t, u_b[:Tf], "-k", lw=1, label=r"$u^b$")
    # ax.plot(t, u_hat_b[:Tf], "--k", lw=1, label=r"$\hat{u}^b$")
    # ax.fill_between(t, u_hat_b_min, u_hat_b_max, alpha=0.2, color="grey", lw=1)
    ax.plot(t, x[:Tf], "-b", lw=1, label=r"$x$")
    ax.plot(t, x_hat[:Tf], "--r", lw=1, label=r"$\hat{x}$")
    ax.plot(t, x_max * np.ones((Tf,)), "-.b", lw=1, label=r"$x_\text{max}$")
    ax.fill_between(
        t,
        x_hat_min,
        x_hat_max,
        alpha=0.1,
        color="r",
        lw=1,
        label="error\nbound",
    )
    ax.grid(axis="y", lw=0.25, alpha=0.5)
    ax.set_xlabel(r"time $\ (\text{hrs})$", **font_dict)
    ax.set_ylabel(r"normalized storage" "\n" r"battery state $\ ()$", **font_dict)
    leg = ax.legend(
        loc="upper right",
        bbox_to_anchor=(1, 0.98),
        ncol=2,
        fontsize=font_size,
        framealpha=0.5,
        fancybox=False,
        edgecolor="black",
    )
    # leg.legend_handles[0].set_ydata([6] * 3)
    # leg.texts[0].set_y(9)
    # leg.legend_handles[1].set_ydata([6] * 3)
    # leg.texts[1].set_y(9)
    # leg._legend_box.set_height(57)
    ax.set_xticks([0, 12, 24, 36, 47])
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    ax.margins(axis_margins, axis_margins)

    plt.show()

    if SAVE_FIG:
        fig.savefig(
            DIR_PATH + "/../plots/storage_battery_state.png", dpi=SAVE_DPI
        )  # , bbox_inches='tight')


def main() -> None:
    price_type = "linear-convex"
    # price_type = "linear"
    file_name = DIR_PATH + "/real-time-price-control_logs_" + price_type + ".pkl"
    with open(file_name, "rb") as file:
        logs = pickle.load(file)
    _plot_graphs(logs)


if __name__ == "__main__":
    main()
