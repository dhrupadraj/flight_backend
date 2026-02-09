import numpy as np
import logging
from utils.map import generate_synthetic_wind_field
from data import (
    get_wind_history_for_region,
    get_wind_field_for_display,
    _load_mean_std
)

logger = logging.getLogger("uvicorn.error")

def load_wind_history(lat_min, lat_max, lon_min, lon_max, H, W, Tin, target_datetime):
    try:
        wind = get_wind_history_for_region(
            lat_min, lat_max, lon_min, lon_max,
            target_h=H, target_w=W,
            num_timesteps=Tin,
            target_datetime=target_datetime
        )
        return wind, "era5_grib"

    except Exception as e:
        logger.warning(f"Wind fallback to synthetic: {e}")
        u, v, _, _, _ = generate_synthetic_wind_field(
            lat_min, lat_max, lon_min, lon_max, grid_size=H
        )
        wind_raw = np.stack(
            [np.stack([u, v], axis=0)] * Tin, axis=0
        ).astype(np.float32)

        mean, std = _load_mean_std()
        return (wind_raw - mean) / (std + 1e-6), "synthetic"


def get_wind_field_for_display_service(lat_min, lat_max, lon_min, lon_max, grid_size=28):
    """
    Return wind field for map display with fallback to synthetic wind.
    Returns (u, v, lat_mesh, lon_mesh, source).
    """
    try:
        u, v, lat_mesh, lon_mesh = get_wind_field_for_display(
            lat_min, lat_max, lon_min, lon_max, grid_size=grid_size
        )
        return u, v, lat_mesh, lon_mesh, "era5"
    except Exception as e:
        logger.warning(f"Wind display fallback to synthetic: {e}")
        u, v, _, lat_mesh, lon_mesh = generate_synthetic_wind_field(
            lat_min, lat_max, lon_min, lon_max, grid_size=grid_size
        )
        return u, v, lat_mesh, lon_mesh, "synthetic"
