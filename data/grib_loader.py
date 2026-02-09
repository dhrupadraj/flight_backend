
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
        
    wind = np.stack([u, v], axis=1)  # (T, 2, H, W)
    _cached_wind = (wind, lat, lon, grib_times)
    return _cached_wind


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
    from scipy.interpolate import RegularGridInterpolator

    if data.ndim == 2:
        data = data[np.newaxis, ...]
        squeeze = True
    else:
        squeeze = False

    n_t, *spatial = data.shape
    dst_h, dst_w = len(dst_lat), len(dst_lon)
    dst_lon_2d, dst_lat_2d = np.meshgrid(dst_lon, dst_lat)

    out = np.zeros((n_t, dst_h, dst_w), dtype=np.float32)
    for t in range(n_t):
        interp = RegularGridInterpolator(
            (src_lat, src_lon),
            data[t],
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        pts = np.column_stack([dst_lat_2d.ravel(), dst_lon_2d.ravel()])
        out[t] = interp(pts).reshape(dst_h, dst_w)
        if np.any(np.isnan(out[t])):
            out[t] = np.nan_to_num(out[t], nan=0.0)

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

    # 5. Resampling (Tin, 2, H, W) - Optimized: batch resample
    dst_lat = np.linspace(lat_min, lat_max, target_h)
    dst_lon = np.linspace(lon_min, lon_max, target_w)
    
    # Create interpolator once and reuse for all timesteps/channels
    from scipy.interpolate import RegularGridInterpolator
    dst_lon_2d, dst_lat_2d = np.meshgrid(dst_lon, dst_lat)
    pts = np.column_stack([dst_lat_2d.ravel(), dst_lon_2d.ravel()])
    
    resampled = np.zeros((num_timesteps, 2, target_h, target_w), dtype=np.float32)
    for t in range(num_timesteps):
        for c in range(2):
            # Reuse interpolator creation for better performance
            interp = RegularGridInterpolator(
                (grib_lat, grib_lon),
                wind_slice[t, c],
                method="linear",
                bounds_error=False,
                fill_value=0.0,  # Use 0.0 instead of nan for faster processing
            )
            resampled[t, c] = interp(pts).reshape(target_h, target_w)

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
