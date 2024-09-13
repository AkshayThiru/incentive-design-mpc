import numpy as np

from chargingstation.lompc import LoMPCConstants
from chargingstation.price_solver import PriceSolver


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


def _solve_price_coordination(
    nEVs: int,
    N: int,
    consts: LoMPCConstants,
    price_type: str,
    lmbd_r: float,
    max_initial_charge: float,
) -> None:
    price_coord = PriceSolver(N, consts, price_type)
    y0 = max_initial_charge * consts.y_max * np.random.random((nEVs,))
    price_coord.set_charge_levels(y0)
    w_ref = consts.w_max * np.random.random((N,))
    price_coord.compute_optimal_prices(w_ref, lmbd_r)


def _test_single_EV() -> None:
    print("-" * 75)
    print("Single EV convergence test:")
    nEVs = 1
    N = 12
    ev_types = ["small", "large"]
    price_types = ["linear", "linear-convex"]
    lmbd_r = 0

    for ev_type in ev_types:
        consts = _get_lompc_consts(ev_type)
        for price_type in price_types:
            print("-" * 50)
            print(f"EV type: {ev_type} | Price type: {price_type}")
            print("-" * 50)
            _solve_price_coordination(nEVs, N, consts, price_type, lmbd_r, 1 / 3.0)


def _test_multiple_EVs() -> None:
    print("-" * 75)
    print("Multiple EV convergence test:")
    nEVs = 100
    N = 12
    ev_types = ["small", "large"]
    price_type = "linear-convex"
    lmbd_r = 0

    for ev_type in ev_types:
        consts = _get_lompc_consts(ev_type)
        print("-" * 50)
        print(f"EV type: {ev_type}")
        print("-" * 50)
        _solve_price_coordination(nEVs, N, consts, price_type, lmbd_r, 1 / 36.0)


def _test_horizon_length() -> None:
    print("-" * 75)
    print("Horizon length convergence test:")
    nEVs = 10
    horizon_lengths = [12, 24]
    ev_types = ["small", "large"]
    price_type = "linear-convex"
    lmbd_r = 0

    for ev_type in ev_types:
        consts = _get_lompc_consts(ev_type)
        for N in horizon_lengths:
            print("-" * 50)
            print(f"EV type: {ev_type} | Horizon length: {N}")
            print("-" * 50)
            _solve_price_coordination(nEVs, N, consts, price_type, lmbd_r, 1 / 36.0)


def _test_robustness_parameter() -> None:
    print("-" * 75)
    print("Robustness parameter convergence test:")
    nEVs = 10
    N = 12
    ev_types = ["small", "large"]
    price_type = "linear-convex"
    robustness_parameters = [0, N, 2 * N, 3 * N]

    for ev_type in ev_types:
        consts = _get_lompc_consts(ev_type)
        for lmbd_r in robustness_parameters:
            print("-" * 50)
            print(f"EV type: {ev_type} | \lambda_r: {lmbd_r:7.2f}")
            print("-" * 50)
            _solve_price_coordination(nEVs, N, consts, price_type, lmbd_r, 1 / 36.0)


def main() -> None:
    # _test_single_EV()
    _test_multiple_EVs()

    # _test_horizon_length()

    # _test_robustness_parameter()


if __name__ == "__main__":
    main()
