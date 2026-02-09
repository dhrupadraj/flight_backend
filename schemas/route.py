from pydantic import BaseModel

class RouteRequest(BaseModel):
    dep_lat: float
    dep_lon: float
    arr_lat: float
    arr_lon: float
    departure_time: str
    flight_date: str
