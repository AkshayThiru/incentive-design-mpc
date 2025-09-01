import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from chargingstation.charging_station import ChargingStationConstants
from chargingstation.example.real_time_price_control import \
    get_chargingstation_consts
from chargingstation.lompc import LoMPC
from chargingstation.price_solver import PriceSolver
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


def _plot_robustness_bounds_large_robots(consts: ChargingStationConstants) -> None:
    N = consts.horizon_lompc
    M_2 = 10
    consts_l = consts.large_EV_consts
    lompc = LoMPC(N, consts_l)
    A = lompc.get_input_mat()

    # w error bound vs gamma.
    gamma_max_arr = consts_l.y_max * np.arange(1, 0, -0.01)
    len_arr = len(gamma_max_arr)
    lmbd = 5 * consts_l.theta * np.random.random((3 * N,))
    kappa = 1e-5
    lmbd_r = consts_l.delta * kappa
    A_bar = A.T @ A + kappa * np.eye(N)  # Metric for the w-inner product.
    w_err, w0_err = np.zeros((len_arr,)), np.zeros((len_arr,))
    w_err_max = np.zeros((len_arr,))
    w_err_bound, w0_err_bound = np.zeros((len_arr,)), np.zeros((len_arr,))
    for j in tqdm(range(len_arr)):
        gamma_arr = gamma_max_arr[j] * np.random.random((M_2,))
        gamma_bar = gamma_max_arr[j] / 2  # (np.max(gamma_arr) - np.min(gamma_arr)) / 2
        gamma_sc = (np.max(gamma_arr) + np.min(gamma_arr)) / 2
        w_opt_ref, _ = lompc.solve_lompc(lmbd, lmbd_r, gamma_sc)
        w_opt_avg = np.zeros((N,))
        w_err_max_j = 0
        for i in range(M_2):
            w_opt, _ = lompc.solve_lompc(lmbd, lmbd_r, gamma_arr[i])
            w_opt_avg += w_opt
            w_err_i = np.sqrt((w_opt - w_opt_ref) @ A_bar @ (w_opt - w_opt_ref))
            w_err_max_j = np.max((w_err_max_j, w_err_i))
        w_opt_avg = w_opt_avg / M_2
        w_err[j] = np.sqrt((w_opt_avg - w_opt_ref) @ A_bar @ (w_opt_avg - w_opt_ref))
        w_err_max[j] = w_err_max_j
        w_err_bound[j] = np.sqrt(N) * gamma_sc
        w0_err[j] = np.abs(w_opt_avg[0] - w_opt_ref[0])
        w0_err_bound[j] = np.sqrt(N) * gamma_sc * np.min((1, 1 / np.sqrt(kappa)))

    # Figure properties.
    fig_height = 1.4  # [inch].
    font_size = 8
    font_dict = {
        "fontsize": font_size,  # [pt]
        "fontstyle": "normal",
        "fontweight": "normal",
    }
    axis_margins = 0.05
    # Plot.
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
    ax.plot(gamma_max_arr, w_err, "-b", lw=1, label=r"$\Vert w - \hat{w}\Vert$")
    # ax.plot(gamma_max_arr, w_err_max, "-g", lw=1, label=r"$\max_i \{\Vert w^i - \hat{w}\Vert\}$")
    ax.plot(gamma_max_arr, w_err_bound, "--r", lw=1, label=r"$\sqrt{N}\bar{\Gamma}$")
    ax.grid(axis="y", lw=0.25, alpha=0.5)
    ax.set_xlabel(r"range of initial normalized SoC, $\bar{\Gamma} \ ()$", **font_dict)
    ax.set_ylabel(r"error bound $\ ()$", **font_dict)
    ax.legend(
        loc="lower right",
        fontsize=font_size,
        framealpha=0.5,
        fancybox=False,
        edgecolor="black",
    )
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    ax.set_yscale("log")
    ax.set_xticks([0, 0.3, 0.6, consts_l.y_max])
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=3))
    ax.yaxis.set_minor_locator(
        mpl.ticker.LogLocator(base=10, subs=np.linspace(0.2, 1, 5), numticks=50)
    )
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.margins(axis_margins, axis_margins)

    plt.show()

    if SAVE_FIG:
        fig.savefig(
            DIR_PATH + "/robustness_bounds.png", dpi=SAVE_DPI
        )  # , bbox_inches='tight')


def _plot_dual_cost_decrease_large_robots(consts: ChargingStationConstants) -> None:
    N = consts.horizon_lompc
    M_2 = 100
    lmbd_r = 0
    consts_l = consts.large_EV_consts

    price_solver = PriceSolver(N, consts_l, "linear-convex")
    y0 = MIN_INITIAL_SOC + 1 / 24 * consts_l.y_max * np.random.random((M_2,))
    price_solver.set_charge_levels(y0)
    w_ref = consts_l.w_max * np.random.random((N,))
    _, stats = price_solver.compute_optimal_prices(w_ref, lmbd_r)
    dual_cost_decrease_ac = stats["dual_cost_decrease_actual"]
    dual_cost_decrease_pred = stats["dual_cost_decrease_predicted"]
    niter = len(dual_cost_decrease_ac)

    # Figure properties.
    fig_height = 1.4  # [inch].
    font_size = 8
    font_dict = {
        "fontsize": font_size,  # [pt]
        "fontstyle": "normal",
        "fontweight": "normal",
    }
    axis_margins = 0.05
    # Plot.
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
    ax.plot(np.arange(niter), dual_cost_decrease_ac, "-b", lw=1, label="actual")
    ax.plot(np.arange(niter), dual_cost_decrease_pred, "--r", lw=1, label="guaranteed")
    # ax.plot(np.arange(niter), np.zeros((niter,)), "--k", lw=1)
    ax.grid(axis="y", lw=0.25, alpha=0.5)
    ax.set_xlabel(r"number of iterations", **font_dict)
    ax.set_ylabel(r"dual cost decrease" "\n" r"per iteration $\ ()$", **font_dict)
    ax.legend(
        loc="upper right",
        fontsize=font_size,
        framealpha=0.5,
        fancybox=False,
        edgecolor="black",
    )
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=5))
    ax.yaxis.set_minor_locator(
        mpl.ticker.LogLocator(base=10, subs=np.linspace(0.1, 1, 10), numticks=50)
    )
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.margins(axis_margins, axis_margins)

    plt.show()

    if SAVE_FIG:
        fig.savefig(
            DIR_PATH + "/dual_cost_decrease.png", dpi=SAVE_DPI
        )  # , bbox_inches='tight')


def _price_comparison_large_robots(consts: ChargingStationConstants) -> None:
    N = consts.horizon_lompc
    M_2 = 1
    consts_l = consts.large_EV_consts
    lmbd_r = 0
    nsamples = 100

    price_solver_l = PriceSolver(N, consts_l, "linear")
    price_solver_lc = PriceSolver(N, consts_l, "linear-convex")
    avg_price_l, avg_price_lc = 0, 0
    for _ in tqdm(range(nsamples)):
        y0 = consts_l.y_max * np.random.random((M_2,))
        price_solver_l.set_charge_levels(y0)
        price_solver_lc.set_charge_levels(y0)
        w_ref = consts_l.w_max * np.random.random((N,))
        _, stats_l = price_solver_l.compute_optimal_prices(w_ref, lmbd_r)
        _, stats_lc = price_solver_lc.compute_optimal_prices(w_ref, lmbd_r)
        avg_price_l = avg_price_l + stats_l["price_after_reg"]
        avg_price_lc = avg_price_lc + stats_lc["price_after_reg"]
    avg_price_l = avg_price_l / nsamples
    avg_price_lc = avg_price_lc / nsamples
    print(f"Average price: linear       : {avg_price_l:13.8f}")
    print(f"Average price: linear-convex: {avg_price_lc:13.8f}")


def main() -> None:
    # Get charging station constants.
    consts = get_chargingstation_consts()

    # Plots.
    # _plot_robustness_bounds_large_robots(consts)
    _plot_dual_cost_decrease_large_robots(consts)

    # Statistics: Price comparison.
    # _price_comparison_large_robots(consts)


if __name__ == "__main__":
    main()
