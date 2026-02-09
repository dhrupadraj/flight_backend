import numpy as np
from datetime import datetime
import logging

from model import predrnn
from model.Astarinference import astar_search
from utils.map import generate_latlon_grid,generate_synthetic_wind_field
from data import denormalize_wind,get_wind_history_for_region,_load_mean_std


logger = logging.getLogger("uvicorn.error")


def get_nearest_index(target_lat, target_lon, lat_grid, lon_grid):
    i = np.abs(lat_grid - target_lat).argmin()
    j = np.abs(lon_grid - target_lon).argmin()
    return int(i), int(j)


def optimize_route_service(req):
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

    try:
        wind_history = get_wind_history_for_region(
            lat_min, lat_max, lon_min, lon_max,
            H, W, Tin, target_datetime
        )
        data_source = "era5_grib"
    except Exception:
        u, v, _, _, _ = generate_synthetic_wind_field(
            lat_min, lat_max, lon_min, lon_max, grid_size=H
        )
        mean, std = _load_mean_std()
        wind_raw = np.stack([[u, v]] * Tin)
        wind_history = (wind_raw - mean) / (std + 1e-6)
        data_source = "synthetic"

    wind_pred = denormalize_wind(predrnn.predict(wind_history))
    wind_u, wind_v = wind_pred[0, 0] * 3.6, wind_pred[0, 1] * 3.6

    path = astar_search(
        start_idx, goal_idx,
        lat_grid, lon_grid,
        wind_u, wind_v,
        v_air=250.0
    )

    return {
        "data_source": source,
        "path": path,
        "bounds": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        }
    }
    except Exception as e:
        logger.error(f"Error optimizing route: {e}")
        raise HTTPException(status_code=500, detail=str(e))