from cvxpy import CLARABEL

# Global settings.
PRINT_LEVEL = 1  # 3 is highest.

# LoMPC settings.
MIN_MAX_BAT_SOC = 0.75  # Lower bound of y_max.
MAX_MAX_BAT_SOC = 0.9  # Upper bound of y_max.
MAX_BAT_CHARGE_RATE = 0.25  # Upper bound of w_max.

LOMPC_SOLVER = CLARABEL

# PriceSolver settings.
MAX_PRICE_SOLVER_ITERATIONS = 1000
PRICE_SOLVER_EPS_REG = 0.01
PRICE_SOLVER_EPS_TOL = 0.01

PRICE_SOLVER_SOLVER = CLARABEL

# BiMPC settings.
BIMPC_SOLVER = CLARABEL

# ChargingStation settings.
MIN_INITIAL_SOC = 0.3  # y_{min, 1}.
MAX_INITIAL_SOC = 0.5  # y_{min, 2}.

# EVs leave after this fraction of y_max is reached.
MIN_FULL_CHARGE_FRACTION = 0.95

ADD_RESIDUAL_CHARGE_TO_BATTERY = True
