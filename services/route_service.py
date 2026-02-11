import numpy as np
from utils.map import generate_latlon_grid
from datetime import datetime
from services.wind_service import load_wind_history
from services.predrnn_service import predict_wind
from services.astar_service import find_optimized_path


def get_nearest_index(target_lat, target_lon, lat_grid, lon_grid):
    i = np.abs(lat_grid - target_lat).argmin()
    j = np.abs(lon_grid - target_lon).argmin()
    return int(i), int(j)


def optimize_route_service(req):
    from data import denormalize_wind

    dt_str = f"{req.flight_date} {req.departure_time}"
    try:
        target_datetime = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        target_datetime = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")

    lat_grid, lon_grid = generate_latlon_grid(
        req.dep_lat, req.dep_lon,
        req.arr_lat, req.arr_lon,
        size=64
    )

    start_idx = get_nearest_index(req.dep_lat, req.dep_lon, lat_grid, lon_grid)
    goal_idx = get_nearest_index(req.arr_lat, req.arr_lon, lat_grid, lon_grid)

    Tin = 12
    H, W = len(lat_grid), len(lon_grid)

    lat_min, lat_max = float(lat_grid.min()), float(lat_grid.max())
    lon_min, lon_max = float(lon_grid.min()), float(lon_grid.max())
    wind_history, data_source = load_wind_history(
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            H,
            W,
            Tin,
            target_datetime,
        )
    wind_pred = denormalize_wind(predict_wind(wind_history))
    wind_u, wind_v = wind_pred[0, 0] * 3.6, wind_pred[0, 1] * 3.6

    path = find_optimized_path(
        start_idx, goal_idx,
        lat_grid, lon_grid,
        wind_u, wind_v,
        v_air=250.0
    )

    return {
        "data_source": data_source,
        "path": path,
        "bounds": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        }
    }
