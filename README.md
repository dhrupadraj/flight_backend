# AI Flight Route Optimization System

An end‑to‑end intelligent flight routing platform that predicts future wind conditions using deep learning and computes time and fuel‑efficient flight paths between airports inside the indian airspace .

The system combines spatiotemporal weather forecasting using PredRNN and reverse scheduled sampling with A* pathfinding algorithm and provides an interactive stteamlit visualization dashboard for pilots and analysts.

---

## Live streamlit Application

[backend api](http://16.176.208.177:8000)
[streamlit app](https://dhrupadraj-flight-path-optimization-app-3vctir.streamlit.app/)

---

## Core Idea

Aircraft routes are typically planned using fixed airways and shortest distance assumptions. However, wind patterns significantly affect:

* Fuel consumption
* Flight duration
* Emissions

This project predicts future wind fields and integrates them into route planning to generate a more efficient flight trajectory.

---

## System Architecture

User → Streamlit UI → FastAPI Backend → ML Model → Route Optimizer → Response → Visualization

### Pipeline

1. User selects departure and arrival airports
2. Backend predicts future atmospheric wind conditions
3. A* algorithm computes optimal path using wind cost
4. Waypoints and route statistics returned
5. UI visualizes direct vs optimized route

---

## Machine Learning Model

**Model:** PredRNN

### Input

* Historical wind vectors (U, V components)
* Multi‑step time sequence
* Indian airspace ERA5 grid dataset

### Output

* Future wind prediction for upcoming hours


---

## Route Optimization

**Algorithm:** A* Search

### Cost Function

Total Cost = Distance + Wind Resistance Penalty

The optimizer finds a path that may be longer in distance but cheaper in fuel consumption.

---

## Features

### Visualization

* Direct route vs optimized route
* Interactive map
* Waypoints along optimized path
* Weather‑aware trajectory

### Pilot Navigation Support

* Generated waypoint list
* Latitude & longitude coordinates
* Sequential navigation guidance

### Analytics

* Route distance comparison
* Fuel efficiency estimate
* Wind impact visualization

### Export Options

* CSV waypoint download
* Route report generation

---

## API Endpoints

## API Endpoints

### `GET /`
Basic service check endpoint.
Returns a simple message confirming the FastAPI server is running.

### `GET /health`
Health-check endpoint for monitoring.
Returns `{"status": "ok"}` when the API is available.

### `POST /optimize-route`
Computes an optimized flight path between departure and arrival points using wind-aware routing (ERA5/GRIB data with synthetic fallback).  
Returns optimized waypoints, direct route, waypoint count, selected data source, and region bounds.

### `GET /grib-bounds`
Returns geographic and time coverage of the loaded GRIB dataset.
Useful to verify whether requested routes are inside available weather-data bounds.

### `POST /wind-field`
Returns wind field data for map visualization over the route region.
Provides latitude/longitude grids and wind_u / wind_v vectors, with source metadata (era5 or synthetic).

---

## Running Locally

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
streamlit run app.py
```

---

## Current Status

The system is functional and publicly accessible.

Ongoing improvements:

* Better generalization across seasons
* Faster inference
* Larger airport coverage

---

## Tech Stack

* Python
* PyTorch
* FastAPI
* Streamlit
* Plotly
* NumPy 
* Docker
* AWS EC2

---

## Future Work
* Aircraft‑specific performance modeling
* Multi‑altitude optimization

---

## Author

Dhrupad Raj

This project is actively being developed and improved.

