import pickle

import numpy as np

from chargingstation.bimpc import BiMPCConstants, BiMPCTrackingCost
from chargingstation.charging_station import (ChargingStation,
                                              ChargingStationConstants)
from chargingstation.demand_data import medium_term_demand_forecast
from chargingstation.lompc import LoMPCConstants

## Simulation parameters.
SIMULATION_LENGTH = 49  # [hours].

HORIZON_LOMPC = 12
HORIZON_BIMPC = 16

NUM_ROBOTS_PER_BAT_TYPE = 500
NUM_PARTITIONS = 12

PRICE_TYPE = "linear-convex"
# PRICE_TYPE = "linear"

DEMAND_SCALE = 1 / 4  # {1 / 4, 1 / 3}.


def _get_lompc_consts() -> tuple[LoMPCConstants, LoMPCConstants]:
    delta_s = 0.05
    theta_s = 10
    s_max_s = 0.9
    w_max_s = 0.25
    consts_s = LoMPCConstants(delta_s, theta_s, s_max_s, w_max_s, "small")

    delta_l = 0.025
    theta_l = 50
    s_max_l = 0.9
    w_max_l = 0.15
    consts_l = LoMPCConstants(delta_l, theta_l, s_max_l, w_max_l, "large")

    return consts_s, consts_l


def _get_normalized_bimpc_consts() -> BiMPCConstants:
    delta = 1e3
    c_gen = 1
    u_gen_max = 1
    u_bat_max = 0.3  # {0.15, 0.3}.
    xi_bat_max = 0.3  # {0.5}.
    tracking_cost_type = BiMPCTrackingCost.EXP_UNWEIGHTED
    exp_rate = 5  # {5, np.Inf}.
    return BiMPCConstants(
        delta, c_gen, u_gen_max, u_bat_max, xi_bat_max, tracking_cost_type, exp_rate
    )


def _get_unnormalized_external_demand() -> np.ndarray:
    demand = medium_term_demand_forecast(
        SIMULATION_LENGTH + HORIZON_BIMPC + 1, DEMAND_SCALE, interpolate=False
    )
    return demand


def main() -> None:
    # Set charging station constants.
    consts_s, consts_l = _get_lompc_consts()
    consts_bi = _get_normalized_bimpc_consts()
    demand = _get_unnormalized_external_demand()
    consts = ChargingStationConstants(
        SIMULATION_LENGTH,
        HORIZON_BIMPC,
        HORIZON_LOMPC,
        NUM_ROBOTS_PER_BAT_TYPE,
        NUM_PARTITIONS,
        demand,
        consts_bi,
        consts_s,
        consts_l,
        PRICE_TYPE,
    )
    # Initialize simulation.
    cs = ChargingStation(consts)
    logs = cs.simulate()

    # Save log file.
    file_name = (
        "chargingstation/example/real-time-price-control_logs_" + PRICE_TYPE + ".pkl"
    )
    with open(file_name, "wb") as file:
        pickle.dump(logs, file)


if __name__ == "__main__":
    main()
