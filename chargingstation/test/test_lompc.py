from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from chargingstation.lompc import LoMPC, LoMPCConstants


def _get_lompc_consts(bat_type: str) -> LoMPCConstants:
    if bat_type == "small":
        delta = 0.05
        theta = 10
        s_max = 0.9
        w_max = 0.25
    elif bat_type == "large":
        delta = 1
        theta = 50
        s_max = 0.9
        w_max = 0.15
    else:
        raise ValueError("Invalid battery type")
    return LoMPCConstants(delta, theta, s_max, w_max, bat_type)


def _print_lompc_solve_time(N: int, consts: LoMPCConstants, lompc: LoMPC) -> None:
    nrobots = 100
    start_time = default_timer()
    for _ in range(nrobots):
        lmbd = consts.theta * np.random.random((3 * N,))
        lmbd_m = (3 * N) * consts.delta * np.random.random()
        gamma = consts.s_max * np.random.random()
        lompc.solve_lompc(lmbd, lmbd_m, gamma)
    end_time = default_timer()
    print(f"LoMPC Solve time for 100 instances: {end_time - start_time} s")
    print(f"Average LoMPC solve time          : {(end_time - start_time) / nrobots} s")


def _plot_unpriced_energy_consumption(
    N: int, consts: LoMPCConstants, lompc: LoMPC
) -> None:
    t = np.arange(0, N)
    w_opt, _ = lompc.solve_lompc(np.zeros((3 * N,)), 0, consts.s_max)

    _, ax = plt.subplots(2)
    ax[0].plot(t, w_opt, "-b", label="optimal w")
    ax[0].plot(t, consts.w_max * np.ones((N,)), "--r")
    ax[0].legend()
    ax[1].plot(t, np.cumsum(w_opt), "-b", label="optimal y")
    ax[1].plot(t, consts.s_max * np.ones((N,)), "--r")
    ax[1].legend()
    plt.show()


def _check_robustness_bounds(N: int, consts: LoMPCConstants, lompc: LoMPC) -> None:
    gamma_max_arr = consts.s_max * np.arange(1, 0, -0.01)
    len_arr = len(gamma_max_arr)
    A = np.tril(np.ones((N, N)))
    lmbd = consts.theta * np.random.random((3 * N,))
    kappa = (3 * N) * np.random.random()
    lmbd_r = consts.delta * kappa
    A_bar = A.T @ A + kappa * np.eye(N)  # Metric for the w-inner product.

    nrobots = 10
    w_err, w0_err = np.zeros((len_arr,)), np.zeros((len_arr,))
    w_err_bound, w0_err_bound = np.zeros((len_arr,)), np.zeros((len_arr,))
    for j in tqdm(range(len_arr)):
        gamma_arr = gamma_max_arr[j] * np.random.random((nrobots,))
        gamma_rng = gamma_max_arr[j] / 2  # (np.max(gamma_arr) - np.min(gamma_arr)) / 2
        gamma_ref = (np.max(gamma_arr) + np.min(gamma_arr)) / 2
        w_opt_avg = np.zeros((N,))
        for i in range(nrobots):
            w_opt, _ = lompc.solve_lompc(lmbd, lmbd_r, gamma_arr[i])
            w_opt_avg += w_opt
        w_opt_avg /= nrobots
        w_opt_ref, _ = lompc.solve_lompc(lmbd, lmbd_r, gamma_ref)
        w_err[j] = np.sqrt((w_opt_avg - w_opt_ref) @ A_bar @ (w_opt_avg - w_opt_ref))
        w0_err[j] = np.abs(w_opt_avg[0] - w_opt_ref[0])
        w_err_bound[j] = np.sqrt(N) * gamma_rng
        w0_err_bound[j] = np.sqrt(N) / np.sqrt(N + kappa) * gamma_rng

    _, ax = plt.subplots(2)
    ax[0].plot(gamma_max_arr, w_err, "-b", label="w-error")
    ax[0].plot(gamma_max_arr, w_err_bound, "--r", label="w-error bound")
    ax[0].set_yscale("log")
    ax[0].legend()
    ax[1].plot(gamma_max_arr, w0_err, "-b", label="w0-error")
    ax[1].plot(gamma_max_arr, w0_err_bound, "--r", label="w0-error bound")
    ax[1].set_yscale("log")
    ax[1].legend()
    plt.show()


def main() -> None:
    N = 12
    # bat_type = "small"
    bat_type = "large"
    consts = _get_lompc_consts(bat_type)
    lompc = LoMPC(N, consts)

    _print_lompc_solve_time(N, consts, lompc)
    _plot_unpriced_energy_consumption(N, consts, lompc)
    _check_robustness_bounds(N, consts, lompc)


if __name__ == "__main__":
    main()
