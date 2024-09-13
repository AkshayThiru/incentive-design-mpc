from timeit import default_timer

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from chargingstation.lompc import LoMPC, LoMPCConstants

mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "STIXGeneral"


def _get_lompc_consts(ev_type: str) -> LoMPCConstants:
    if ev_type == "small":
        delta = 0.05
        theta = 10
        y_max = 0.9
        w_max = 0.25
    elif ev_type == "large":
        delta = 0.025
        theta = 50
        y_max = 0.9
        w_max = 0.15
    else:
        raise ValueError("Invalid EV type")
    return LoMPCConstants(delta, theta, y_max, w_max, ev_type)


def _print_lompc_solve_time(N: int, consts: LoMPCConstants, lompc: LoMPC) -> None:
    nEVs = 100
    start_time = default_timer()
    for _ in range(nEVs):
        lmbd = consts.theta * np.random.random((3 * N,))
        lmbd_r = (3 * N) * consts.delta * np.random.random()
        gamma = consts.y_max * np.random.random()
        lompc.solve_lompc(lmbd, lmbd_r, gamma)
    end_time = default_timer()
    print(f"LoMPC Solve time for 100 instances: {end_time - start_time} s")
    print(f"Average LoMPC solve time          : {(end_time - start_time) / nEVs} s")


def _plot_unpriced_electricity_consumption(
    N: int, consts: LoMPCConstants, lompc: LoMPC
) -> None:
    t = np.arange(0, N)
    gamma = consts.y_max
    w_opt, _ = lompc.solve_lompc(np.zeros((3 * N,)), 0, gamma)

    _, ax = plt.subplots(2, layout="constrained", sharex=True)
    ax[0].plot(t, w_opt, "-b", label=r"$w^*$")
    ax[0].plot(t, consts.w_max * np.ones((N,)), "--r", label=r"$w_\text{max}$")
    ax[0].legend()
    ax[1].plot(t, np.cumsum(w_opt), "-b", label=r"$y - y_0$")
    ax[1].plot(t, gamma * np.ones((N,)), "--r", label=r"$\Gamma$")
    ax[1].set_xlabel(r"$t$")
    ax[1].legend()
    plt.show()


def _check_error_bounds(N: int, consts: LoMPCConstants, lompc: LoMPC) -> None:
    gamma_max_arr = consts.y_max * np.arange(1, 0, -0.01)
    len_arr = len(gamma_max_arr)
    A = lompc.get_input_mat()
    lmbd = consts.theta * np.random.random((3 * N,))
    kappa = (3 * N) * np.random.random() + 1e-5
    lmbd_r = consts.delta * kappa
    A_bar = A.T @ A + kappa * np.eye(N)  # Metric for the w-inner product.

    nEVs = 10
    w_err, w0_err = np.zeros((len_arr,)), np.zeros((len_arr,))
    w_err_bound, w0_err_bound = np.zeros((len_arr,)), np.zeros((len_arr,))
    for j in tqdm(range(len_arr)):
        gamma_arr = gamma_max_arr[j] * np.random.random((nEVs,))
        gamma_rng = gamma_max_arr[j] / 2  # (np.max(gamma_arr) - np.min(gamma_arr)) / 2
        gamma_ref = (np.max(gamma_arr) + np.min(gamma_arr)) / 2
        w_opt_avg = np.zeros((N,))
        for i in range(nEVs):
            w_opt, _ = lompc.solve_lompc(lmbd, lmbd_r, gamma_arr[i])
            w_opt_avg += w_opt
        w_opt_avg = w_opt_avg / nEVs
        w_opt_ref, _ = lompc.solve_lompc(lmbd, lmbd_r, gamma_ref)
        w_err[j] = np.sqrt((w_opt_avg - w_opt_ref) @ A_bar @ (w_opt_avg - w_opt_ref))
        w0_err[j] = np.abs(w_opt_avg[0] - w_opt_ref[0])
        w_err_bound[j] = np.sqrt(N) * gamma_rng
        w0_err_bound[j] = w_err_bound[j] * np.min((1, 1 / np.sqrt(kappa)))

    _, ax = plt.subplots(2, layout="constrained", sharex=True)
    ax[0].plot(gamma_max_arr, w_err, "-b", label=r"$\Vert w - \hat{w}\Vert$")
    ax[0].plot(gamma_max_arr, w_err_bound, "--r", label=r"$\sqrt{N}\bar{\Gamma}$")
    ax[0].set_yscale("log")
    ax[0].legend()
    ax[1].plot(gamma_max_arr, w0_err, "-b", label=r"$|w_0 - \hat{w}_0|$")
    ax[1].plot(gamma_max_arr, w0_err_bound, "--r", label=r"$\sqrt{N}\bar{\Gamma}$")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$\bar{\Gamma}$")
    ax[1].legend()
    plt.show()


def main() -> None:
    N = 12
    ev_type = "small"
    # ev_type = "large"
    consts = _get_lompc_consts(ev_type)
    lompc = LoMPC(N, consts)

    _print_lompc_solve_time(N, consts, lompc)
    _plot_unpriced_electricity_consumption(N, consts, lompc)
    _check_error_bounds(N, consts, lompc)


if __name__ == "__main__":
    main()
