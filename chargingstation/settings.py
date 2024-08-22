from cvxpy import CLARABEL

# Global settings.
PRINT_LEVEL = 2  # 3 is highest.

# LoMPC settings.
MIN_MAX_BAT_CHARGE = 0.75  # Lower bound of s_max.
MAX_MAX_BAT_CHARGE = 0.9  # Upper bound of s_max.
MAX_BAT_CHARGE_RATE = 0.25  # Upper bound of w_max.

LOMPC_SOLVER = CLARABEL

# PriceCoordinator settings.
MAX_PRICE_COORD_ITERATIONS = 1000
PRICE_COORD_EPS_REG = 0.01
PRICE_COORD_EPS_TOL = 0.01

PRICE_COORD_SOLVER = CLARABEL

# BiMPC settings.
BIMPC_SOLVER = CLARABEL

# ChargingStation settings.
MIN_INITIAL_CHARGE = 0.3
MAX_INITIAL_CHARGE = 0.5

MIN_FULL_CHARGE_FRACTION = 0.95
