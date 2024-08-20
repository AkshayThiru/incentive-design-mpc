import csv
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def _get_csv_data() -> list:
    file_name = "data/Real-Time Total Load.csv"
    path = Path(__file__).parent / file_name
    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        data = list(reader)
    return data


def medium_term_demand_forecast(
    hours: int, scale: float, interpolate: bool = False
) -> np.ndarray:
    data = _get_csv_data()
    # Mid-hour forecasts every hour, starting at 00:00.
    forecast_24 = np.asarray(data[30:54]).astype(float)
    # Interpolated demand forecasts every 30 mins, starting from 00:00.
    forecast_48 = np.zeros((48,))
    forecast_48[1::2] = forecast_24[:, 1]
    forecast_48[0::2] = (
        forecast_24[:, 1] + forecast_24[:, 1].take(range(-1, 23), mode="wrap")
    ) / 2
    forecast_48_ = forecast_48.tolist()
    demand = forecast_48_ * (hours // 24) + forecast_48_[: 2 * (hours % 24)]
    if not interpolate:
        demand = demand[0::2]
    return scale * np.array(demand)


def main() -> None:
    hours = 48
    demand = medium_term_demand_forecast(hours, 1 / 4, interpolate=False)
    demand_interp = medium_term_demand_forecast(hours, 1 / 4, interpolate=True)
    _, ax = plt.subplots(1)
    ax.plot(np.arange(len(demand)), demand, "-b", label="uninterpolated")
    ax.plot(
        np.arange(len(demand_interp)) / 2, demand_interp, "-r", label="interpolated"
    )
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
