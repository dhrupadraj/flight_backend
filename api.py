from fastapi import FastAPI, HTTPException
from schemas import RouteRequest
from services.route_service import optimize_route_service
import logging

app = FastAPI()
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
        path, source, lat_min, lat_max, lon_min, lon_max = optimize_route_service(req)

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
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
