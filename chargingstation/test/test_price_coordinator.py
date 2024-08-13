import numpy as np

from chargingstation.lompc import LoMPCConstants
from chargingstation.price_coordinator import PriceCoordinator


def _get_lompc_consts(bat_type: str) -> LoMPCConstants:
    if bat_type == "small":
        delta = 0.05
        theta = 10
        s_max = 0.9
        w_max = 0.25
    elif bat_type == "large":
        delta = 0.025
        theta = 50
        s_max = 0.9
        w_max = 0.15
    else:
        raise ValueError("Invalid battery type")
    return LoMPCConstants(delta, theta, s_max, w_max, bat_type)


def _solve_price_coordination(
    nrobots: int,
    N: int,
    consts: LoMPCConstants,
    price_type: str,
    lmbd_r: float,
    max_initial_charge: float,
) -> None:
    price_coord = PriceCoordinator(N, consts, price_type)
    s0 = max_initial_charge * consts.s_max * np.random.random((nrobots,))
    price_coord.set_charge_levels(s0)
    w_ref = consts.w_max * np.random.random((N,))
    price_coord.compute_optimal_prices(w_ref, lmbd_r)


def _test_single_robot() -> None:
    print("-" * 75)
    print("Single robot convergence test:")
    nrobots = 1
    N = 12
    bat_types = ["small", "large"]
    price_types = ["linear", "linear-convex"]
    lmbd_r = 0

    for bat_type in bat_types:
        consts = _get_lompc_consts(bat_type)
        for price_type in price_types:
            print("-" * 50)
            print(f"Battery type: {bat_type} | Price type: {price_type}")
            print("-" * 50)
            _solve_price_coordination(nrobots, N, consts, price_type, lmbd_r, 1 / 3.0)


def _test_multiple_robot() -> None:
    print("-" * 75)
    print("Multiple robot convergence test:")
    nrobots = 100
    N = 12
    bat_types = ["small", "large"]
    price_type = "linear-convex"
    lmbd_r = 0

    for bat_type in bat_types:
        consts = _get_lompc_consts(bat_type)
        print("-" * 50)
        print(f"Battery type: {bat_type}")
        print("-" * 50)
        _solve_price_coordination(nrobots, N, consts, price_type, lmbd_r, 1 / 9.0)


def _test_horizon_length() -> None:
    print("-" * 75)
    print("Horizon length convergence test:")
    nrobots = 10
    horizon_lengths = [12, 24]
    bat_types = ["small", "large"]
    price_type = "linear-convex"
    lmbd_r = 0

    for bat_type in bat_types:
        consts = _get_lompc_consts(bat_type)
        for N in horizon_lengths:
            print("-" * 50)
            print(f"Battery type: {bat_type} | Horizon length: {N}")
            print("-" * 50)
            _solve_price_coordination(nrobots, N, consts, price_type, lmbd_r, 1 / 9.0)


def _test_robustness_parameter() -> None:
    print("-" * 75)
    print("Robustness parameter convergence test:")
    nrobots = 10
    N = 12
    bat_types = ["small", "large"]
    price_type = "linear-convex"
    robustness_parameters = [0, N, 2 * N, 3 * N]

    for bat_type in bat_types:
        consts = _get_lompc_consts(bat_type)
        for lmbd_r in robustness_parameters:
            print("-" * 50)
            print(f"Battery type: {bat_type} | \lambda_r: {lmbd_r:7.2f}")
            print("-" * 50)
            _solve_price_coordination(nrobots, N, consts, price_type, lmbd_r, 1 / 9.0)


def main() -> None:
    _test_single_robot()
    _test_multiple_robot()

    _test_horizon_length()

    _test_robustness_parameter()


if __name__ == "__main__":
    main()
