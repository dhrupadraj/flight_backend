
"""
Load wind data from GRIB file for PredRNN inference.
Uses same preprocessing as training: 500 hPa u/v, normalization with mean/std.
"""
import os
from typing import Optional
import numpy as np
import xarray as xr
GRIB_PATH = os.path.join(os.path.dirname(__file__), "data.grib")
MEAN_PATH = os.path.join(os.path.dirname(__file__), "processed", "mean.npy")
STD_PATH = os.path.join(os.path.dirname(__file__), "processed", "std.npy")
LEVEL = 500  # hPa
INPUT_STEPS = 12  # match training

_cached_wind = None
_cached_mean: Optional[np.ndarray] = None
_cached_std: Optional[np.ndarray] = None
_resample_plan_cache = {}


def _load_grib():
    """Load GRIB and return wind (T, 2, H, W), lat, lon, and times. Cached."""
    global _cached_wind
    if _cached_wind is not None:
        return _cached_wind
    # Use a context manager (with) to ensure the file closes after reading values
    try:
        with xr.open_dataset(
            GRIB_PATH,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa", "level": LEVEL}},
        ) as ds:
            u = ds["u"].values
            v = ds["v"].values
            lat = np.asarray(ds.latitude.values)
            lon = np.asarray(ds.longitude.values)
            grib_times = ds.time.values  # Access before closing
    except Exception:
        with xr.open_dataset(GRIB_PATH, engine="cfgrib") as ds:
            u = ds["u"].values
            v = ds["v"].values
            lat = np.asarray(ds.latitude.values)
            lon = np.asarray(ds.longitude.values)
            grib_times = ds.time.values

    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    
    if lat[0] > lat[-1]:
        u = u[:, ::-1, :]
        v = v[:, ::-1, :]
        lat = lat[::-1]
    if lon[0] > lon[-1]:
        u = u[:, :, ::-1]
        v = v[:, :, ::-1]
        lon = lon[::-1]
        
    wind = np.stack([u, v], axis=1)  # (T, 2, H, W)
    _cached_wind = (wind, lat, lon, grib_times)
    return _cached_wind


def _make_cache_key(src_lat, src_lon, dst_lat, dst_lon):
    return (
        float(src_lat[0]),
        float(src_lat[-1]),
        len(src_lat),
        float(src_lon[0]),
        float(src_lon[-1]),
        len(src_lon),
        float(dst_lat[0]),
        float(dst_lat[-1]),
        len(dst_lat),
        float(dst_lon[0]),
        float(dst_lon[-1]),
        len(dst_lon),
    )


def _get_resample_plan(src_lat, src_lon, dst_lat, dst_lon):
    """Build/cached bilinear interpolation indices and weights for a fixed grid pair."""
    key = _make_cache_key(src_lat, src_lon, dst_lat, dst_lon)
    plan = _resample_plan_cache.get(key)
    if plan is not None:
        return plan

    src_lat = np.asarray(src_lat, dtype=np.float64)
    src_lon = np.asarray(src_lon, dtype=np.float64)
    dst_lat = np.asarray(dst_lat, dtype=np.float64)
    dst_lon = np.asarray(dst_lon, dtype=np.float64)

    lat_i1 = np.searchsorted(src_lat, dst_lat, side="right")
    lon_j1 = np.searchsorted(src_lon, dst_lon, side="right")

    valid_lat = (dst_lat >= src_lat[0]) & (dst_lat <= src_lat[-1])
    valid_lon = (dst_lon >= src_lon[0]) & (dst_lon <= src_lon[-1])

    lat_i1 = np.clip(lat_i1, 1, len(src_lat) - 1)
    lon_j1 = np.clip(lon_j1, 1, len(src_lon) - 1)
    lat_i0 = lat_i1 - 1
    lon_j0 = lon_j1 - 1

    lat0 = src_lat[lat_i0]
    lat1 = src_lat[lat_i1]
    lon0 = src_lon[lon_j0]
    lon1 = src_lon[lon_j1]

    lat_den = np.maximum(lat1 - lat0, 1e-12)
    lon_den = np.maximum(lon1 - lon0, 1e-12)

    wy = (dst_lat - lat0) / lat_den
    wx = (dst_lon - lon0) / lon_den
    wy = np.clip(wy, 0.0, 1.0).astype(np.float32)
    wx = np.clip(wx, 0.0, 1.0).astype(np.float32)

    plan = {
        "lat_i0": lat_i0,
        "lat_i1": lat_i1,
        "lon_j0": lon_j0,
        "lon_j1": lon_j1,
        "wy": wy,
        "wx": wx,
        "valid_lat": valid_lat,
        "valid_lon": valid_lon,
    }
    _resample_plan_cache[key] = plan
    return plan


def _bilinear_resample(data: np.ndarray, src_lat, src_lon, dst_lat, dst_lon) -> np.ndarray:
    """
    Vectorized bilinear resampling on rectilinear grids.
    data: (..., H, W)
    returns: (..., dst_h, dst_w)
    """
    plan = _get_resample_plan(src_lat, src_lon, dst_lat, dst_lon)

    row0 = np.take(data, plan["lat_i0"], axis=-2)
    row1 = np.take(data, plan["lat_i1"], axis=-2)
    v00 = np.take(row0, plan["lon_j0"], axis=-1)
    v01 = np.take(row0, plan["lon_j1"], axis=-1)
    v10 = np.take(row1, plan["lon_j0"], axis=-1)
    v11 = np.take(row1, plan["lon_j1"], axis=-1)

    wy = plan["wy"].reshape((1,) * (data.ndim - 2) + (-1, 1))
    wx = plan["wx"].reshape((1,) * (data.ndim - 2) + (1, -1))

    top = v00 * (1.0 - wx) + v01 * wx
    bottom = v10 * (1.0 - wx) + v11 * wx
    out = top * (1.0 - wy) + bottom * wy

    valid = np.outer(plan["valid_lat"], plan["valid_lon"])
    valid = valid.reshape((1,) * (data.ndim - 2) + valid.shape)
    out = np.where(valid, out, 0.0)
    return out.astype(np.float32, copy=False)


def _load_mean_std() -> tuple[np.ndarray, np.ndarray]:
    """Load mean and std. Cached."""
    global _cached_mean, _cached_std
    if _cached_mean is None or _cached_std is None:
        _cached_mean = np.load(MEAN_PATH)
        _cached_std = np.load(STD_PATH)
    if _cached_mean is None or _cached_std is None:
        raise RuntimeError("Mean/std not loaded.")
    return _cached_mean, _cached_std


def _resample_to_grid(data, src_lat, src_lon, dst_lat, dst_lon):
    """Resample 2D or 3D data from src grid to dst grid via linear interpolation."""
    if data.ndim == 2:
        data = data[np.newaxis, ...]
        squeeze = True
    else:
        squeeze = False

    out = _bilinear_resample(
        np.asarray(data, dtype=np.float32),
        src_lat,
        src_lon,
        dst_lat,
        dst_lon,
    )

    if squeeze:
        out = out[0]
    return out


def get_wind_history_for_region(
    lat_min, lat_max, lon_min, lon_max,
    target_h, target_w,
    num_timesteps=INPUT_STEPS,
    target_datetime=None
) -> np.ndarray:
    
    # 1. Load data
    wind, grib_lat, grib_lon, grib_times = _load_grib() # Ensure _load_grib returns 4 values!
    mean, std = _load_mean_std()

    # 2. Time Indexing (The 500-Error Fix)
    if target_datetime is not None:
        # Force conversion to numpy datetime64[ns] to avoid type errors
        target_np = np.datetime64(target_datetime)
        
        # Calculate index of the closest time
        time_diffs = np.abs(grib_times - target_np)
        end_idx = int(np.argmin(time_diffs))
        
        # Ensure we get exactly num_timesteps ending at end_idx
        start_idx = end_idx - num_timesteps + 1
        
        if start_idx < 0:
            # If flight is too early in the dataset, take first available 12
            wind_slice = wind[0 : num_timesteps]
        else:
            wind_slice = wind[start_idx : end_idx + 1]
    else:
        wind_slice = wind[-num_timesteps:]

    # 3. Handle data shape (Safety Check)
    # If the file is shorter than 12 hours total
    if wind_slice.shape[0] < num_timesteps:
        diff = num_timesteps - wind_slice.shape[0]
        padding = np.repeat(wind_slice[:1], diff, axis=0)
        wind_slice = np.concatenate([padding, wind_slice], axis=0)

    # 4. Spatial Check
    grib_lat_min, grib_lat_max = float(grib_lat.min()), float(grib_lat.max())
    grib_lon_min, grib_lon_max = float(grib_lon.min()), float(grib_lon.max())
    
    if lat_min < grib_lat_min or lat_max > grib_lat_max or \
       lon_min < grib_lon_min or lon_max > grib_lon_max:
         raise ValueError(f"Region ({lat_min}, {lon_min}) outside GRIB domain")

    # 5. Resampling (Tin, 2, H, W)
    dst_lat = np.linspace(lat_min, lat_max, target_h)
    dst_lon = np.linspace(lon_min, lon_max, target_w)
    resampled = _bilinear_resample(
        np.asarray(wind_slice, dtype=np.float32),
        grib_lat,
        grib_lon,
        dst_lat,
        dst_lon,
    )

    # 6. Final Normalization
    norm = (resampled - mean) / (std + 1e-6)
    return norm.astype(np.float32)


def denormalize_wind(wind_norm: np.ndarray) -> np.ndarray:
    """Convert normalized wind back to m/s."""
    mean, std = _load_mean_std()
    return wind_norm * std + mean


def get_wind_field_for_display(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    grid_size: int = 28,
) -> tuple:
    """
    Get ERA5 (GRIB) wind field for map display: raw u, v in m/s on a regular grid.
    Returns (u, v, lat_mesh, lon_mesh) for the given region. Raises if outside GRIB domain.
    """
    # _load_grib returns (wind, lat, lon, times)
    wind, grib_lat, grib_lon, _ = _load_grib()
    grib_lat_min, grib_lat_max = float(grib_lat.min()), float(grib_lat.max())
    grib_lon_min, grib_lon_max = float(grib_lon.min()), float(grib_lon.max())
    if lat_min < grib_lat_min or lat_max > grib_lat_max or lon_min < grib_lon_min or lon_max > grib_lon_max:
        raise ValueError(
            f"Region outside GRIB domain [{grib_lat_min},{grib_lat_max}] x [{grib_lon_min},{grib_lon_max}]"
        )
    dst_lat = np.linspace(lat_min, lat_max, grid_size)
    dst_lon = np.linspace(lon_min, lon_max, grid_size)
    # Last timestep (most recent) for display
    u_src = wind[-1, 0]  # (H, W)
    v_src = wind[-1, 1]
    u = _resample_to_grid(u_src, grib_lat, grib_lon, dst_lat, dst_lon)
    v = _resample_to_grid(v_src, grib_lat, grib_lon, dst_lat, dst_lon)
    lon_mesh, lat_mesh = np.meshgrid(dst_lon, dst_lat)
    return u.astype(np.float32), v.astype(np.float32), lat_mesh, lon_mesh
