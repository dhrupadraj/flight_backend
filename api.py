from fastapi import FastAPI, HTTPException
from schemas import RouteRequest
from services.route_service import optimize_route_service
from services.wind_service import get_wind_field_for_display_service
from data import _load_grib
from fastapi.middleware.cors import CORSMiddleware
import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows ALL origins
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger = logging.getLogger("uvicorn.error")


@app.get("/")
def root():
    return {"message": "Fast API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/optimize-route")
def optimize_route(req: RouteRequest):
    try:
        result = optimize_route_service(req)
        path = result["path"]
        source = result["data_source"]
        bounds = result["bounds"]

        if not path:
            raise HTTPException(status_code=404, detail="No optimized path found")

        return {
            "status": "success",
            "data_source": source,
            "optimized_route": [{"lat": float(a), "lon": float(b)} for a, b in path],
            "direct_route": [
                {"lat": req.dep_lat, "lon": req.dep_lon},
                {"lat": req.arr_lat, "lon": req.arr_lon},
            ],
            "num_waypoints": len(path),
            "region_bounds": {
                "lat_min": bounds["lat_min"],
                "lat_max": bounds["lat_max"],
                "lon_min": bounds["lon_min"],
                "lon_max": bounds["lon_max"],
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error in /optimize-route")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/grib-bounds")
def get_grib_bounds():
    """
    Get the geographic bounds of the GRIB file to check coverage.
    Returns the latitude and longitude ranges available in the GRIB data.
    """
    try:
        wind, grib_lat, grib_lon, grib_times = _load_grib()
        return {
            "status": "success",
            "bounds": {
                "lat_min": float(grib_lat.min()),
                "lat_max": float(grib_lat.max()),
                "lon_min": float(grib_lon.min()),
                "lon_max": float(grib_lon.max())
            },
            "time_range": {
                "start": str(grib_times[0]) if len(grib_times) > 0 else None,
                "end": str(grib_times[-1]) if len(grib_times) > 0 else None,
                "num_timesteps": int(len(grib_times))
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting GRIB bounds: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading GRIB file: {str(e)}")
        
@app.post("/wind-field")
def wind_field(req: RouteRequest):
    """
    Return ERA5 (data.grib) wind field for the route region for map display.
    Used by the app to show wind heatmap and vectors from real data.
    Falls back to synthetic wind if GRIB is unavailable or region outside domain.
    """
    try:
        lat_min = min(req.dep_lat, req.arr_lat) - 2.0
        lat_max = max(req.dep_lat, req.arr_lat) + 2.0
        lon_min = min(req.dep_lon, req.arr_lon) - 2.0
        lon_max = max(req.dep_lon, req.arr_lon) + 2.0
        grid_size = 28

        u, v, lat_mesh, lon_mesh, source = get_wind_field_for_display_service(
            lat_min, lat_max, lon_min, lon_max, grid_size=grid_size
        )

        return {
            "source": source,
            "lat_grid": lat_mesh[:, 0].tolist(),
            "lon_grid": lon_mesh[0, :].tolist(),
            "wind_u": u.tolist(),
            "wind_v": v.tolist(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error in /wind-field")
        raise HTTPException(status_code=500, detail=str(e))
