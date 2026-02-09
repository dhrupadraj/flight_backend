import plotly.graph_objects as go
import numpy as np
import plotly.colors

def generate_synthetic_wind_field(lat_min, lat_max, lon_min, lon_max, grid_size=40):
    """
    Generate a physically plausible synthetic wind field:
    - Mid-latitude westerly jet (stronger at 30–60° latitude)
    - Large-scale Rossby-like waves (meandering jet)
    - Trade-wind style flow in tropics
    - Speeds in realistic range ~5–45 m/s
    """
    lats = np.linspace(lat_min, lat_max, grid_size)
    lons = np.linspace(lon_min, lon_max, grid_size)
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    lat_rad = np.radians(lat_mesh)
    lon_rad = np.radians(lon_mesh)

    # Mid-latitude westerly jet: peak around 40–50° N/S, decays toward poles and equator
    jet_lat = 45.0
    jet_width = 12.0
    jet_strength = 35.0  # m/s
    jet_profile = jet_strength * np.exp(-((lat_mesh - jet_lat) ** 2) / (2 * jet_width**2))
    jet_profile += 0.6 * jet_strength * np.exp(-((lat_mesh + jet_lat) ** 2) / (2 * (jet_width * 1.2) ** 2))

    # Meandering jet: wave pattern along latitude (Rossby-like)
    wave = 0.4 * np.sin(2 * lon_rad) * np.cos(np.radians(lat_mesh - 45))
    jet_u = (jet_profile * (1 + 0.3 * np.cos(lon_rad * 2))) * (1 + wave)
    jet_v = jet_profile * 0.25 * np.sin(3 * lon_rad) * np.exp(-np.abs(np.radians(lat_mesh)) / 0.8)

    # Trade-wind component: easterlies in tropics (negative u), weaker
    tropic_mask = np.exp(-((np.abs(lat_mesh) - 25) ** 2) / 400)
    trade_u = -18 * tropic_mask * (1 + 0.2 * np.sin(lon_rad * 4))
    trade_v = 4 * tropic_mask * np.cos(lon_rad * 2)

    # Small-scale variation (mesoscale/turbulence)
    np.random.seed(42)
    noise_scale = 4.0
    small_u = noise_scale * (
        np.sin(lat_mesh * 0.5) * np.cos(lon_mesh * 0.4)
        + 0.5 * np.sin(lat_mesh * 1.2) * np.sin(lon_mesh * 0.9)
    )
    small_v = noise_scale * (
        np.cos(lat_mesh * 0.4) * np.sin(lon_mesh * 0.5)
        + 0.5 * np.cos(lat_mesh * 0.9) * np.cos(lon_mesh * 1.1)
    )

    u_component = jet_u + trade_u + small_u
    v_component = jet_v + trade_v + small_v
    wind_speed = np.sqrt(u_component**2 + v_component**2)
    # Clamp to plausible range
    wind_speed = np.clip(wind_speed, 2.0, 55.0)
    return u_component, v_component, wind_speed, lat_mesh, lon_mesh

def _wind_direction_label(deg):
    """Convert wind-from direction (degrees) to compass label."""
    labels = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = int((deg / 22.5) + 0.5) % 16
    return labels[idx]


def create_wind_heatmap_and_vectors(fig, dep_coords, arr_coords, grid_size=28, wind_data=None):
    """
    Add a single combined wind layer: one semi-transparent heatmap + vectors on top,
    inserted at the start of the figure so the route is drawn on top and stays visible.
    If wind_data is provided (from ERA5/GRIB via API), use it; otherwise use synthetic wind.
    wind_data: dict with keys 'wind_u', 'wind_v', 'lat_grid', 'lon_grid' (lists/arrays).
    """
    if wind_data is not None:
        u = np.asarray(wind_data["wind_u"], dtype=np.float64)
        v = np.asarray(wind_data["wind_v"], dtype=np.float64)
        lat_1d = np.asarray(wind_data["lat_grid"])
        lon_1d = np.asarray(wind_data["lon_grid"])
        lon_mesh, lat_mesh = np.meshgrid(lon_1d, lat_1d)
        wind_speed = np.sqrt(u**2 + v**2)
    else:
        lat_min = min(dep_coords[0], arr_coords[0]) - 2.0
        lat_max = max(dep_coords[0], arr_coords[0]) + 2.0
        lon_min = min(dep_coords[1], arr_coords[1]) - 2.0
        lon_max = max(dep_coords[1], arr_coords[1]) + 2.0
        u, v, wind_speed, lat_mesh, lon_mesh = generate_synthetic_wind_field(
            lat_min, lat_max, lon_min, lon_max, grid_size
        )
    wind_speed_min, wind_speed_max = wind_speed.min(), wind_speed.max()
    if wind_speed_max <= wind_speed_min:
        wind_speed_max = wind_speed_min + 1.0

    # One colorscale for both heatmap and vectors: blue (calm) -> cyan -> green -> yellow -> red (strong)
    wind_colorscale = [
        [0.0, "rgb(26, 26, 102)"],
        [0.2, "rgb(0, 140, 180)"],
        [0.4, "rgb(0, 200, 100)"],
        [0.6, "rgb(220, 220, 0)"],
        [0.8, "rgb(255, 140, 0)"],
        [1.0, "rgb(200, 0, 0)"],
    ]

    # Wind-from direction for hover (meteorological: where wind comes FROM)
    wind_from_deg = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
    hover_labels = [
        f"{s:.1f} m/s ({s*3.6:.0f} km/h)<br>From {_wind_direction_label(d)} ({d:.0f}°)"
        for s, d in zip(wind_speed.flatten(), wind_from_deg.flatten())
    ]

    # Single combined wind trace: heatmap with moderate opacity so route shows through
    wind_trace = go.Scattergeo(
        lat=lat_mesh.flatten(),
        lon=lon_mesh.flatten(),
        mode="markers",
        marker=dict(
            size=12,
            color=wind_speed.flatten(),
            colorscale=wind_colorscale,
            cmin=wind_speed_min,
            cmax=wind_speed_max,
            colorbar=dict(
                title=dict(text="Wind (m/s)", side="right"),
                thickness=16,
                len=0.5,
                x=1.01,
                tickfont=dict(size=9),
            ),
            opacity=0.52,
            showscale=True,
            line=dict(width=0),
        ),
        showlegend=False,
        hovertext=hover_labels,
        hoverinfo="text",
        name="Wind",
    )
    fig.add_trace(wind_trace)
    # Draw wind layer first so route and waypoints stay on top
    fig.data = [fig.data[-1]] + list(fig.data[:-1])

    # Wind vectors: line + arrowhead (V) at tip so direction is clear; Scattergeo so zoom/pan with map
    step = max(1, grid_size // 10)
    arrow_scale = 0.2
    min_arrow_deg = 0.3
    head_ratio = 0.4   # arrowhead back from tip (fraction of shaft length)
    head_spread = 0.28  # arrowhead half-width (fraction of shaft length)
    seg_lats, seg_lons = [], []
    for i in range(0, u.shape[0], step):
        for j in range(0, u.shape[1], step):
            lat = float(lat_mesh[i, j])
            lon = float(lon_mesh[i, j])
            u_val = u[i, j]
            v_val = v[i, j]
            speed = float(wind_speed[i, j])
            if speed < 1.0:
                continue
            length_deg = min_arrow_deg + arrow_scale * np.sqrt(speed)
            mag = np.hypot(u_val, v_val)
            u_deg = (u_val / mag) * length_deg if mag > 0 else 0
            v_deg = (v_val / mag) * length_deg if mag > 0 else 0
            lat_tip = lat + v_deg
            lon_tip = lon + u_deg
            # Shaft: base -> tip
            seg_lats.extend([lat, lat_tip, None])
            seg_lons.extend([lon, lon_tip, None])
            # Arrowhead (V): direction (dx, dy); perpendicular left = (-dy, dx), right = (dy, -dx)
            dx = (lon_tip - lon) / length_deg if length_deg > 0 else 0
            dy = (lat_tip - lat) / length_deg if length_deg > 0 else 0
            back = head_ratio * length_deg
            side = head_spread * length_deg
            # Left wing: tip -> tip - back*(dx,dy) + side*(-dy, dx)
            lon_L = lon_tip - back * dx - side * dy
            lat_L = lat_tip - back * dy + side * dx
            seg_lats.extend([lat_tip, lat_L, None])
            seg_lons.extend([lon_tip, lon_L, None])
            # Right wing: tip -> tip - back*(dx,dy) - side*(-dy, dx)
            lon_R = lon_tip - back * dx + side * dy
            lat_R = lat_tip - back * dy - side * dx
            seg_lats.extend([lat_tip, lat_R, None])
            seg_lons.extend([lon_tip, lon_R, None])
    if seg_lats:
        fig.add_trace(
            go.Scattergeo(
                lat=seg_lats,
                lon=seg_lons,
                mode="lines",
                line=dict(width=1.4, color="rgba(0, 80, 160, 0.75)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # Insert vectors after heatmap (index 0) so order is: heatmap, vectors, route
        fig.data = [fig.data[0], fig.data[-1]] + list(fig.data[1:-1])

def plot_route_map(route_coords, waypoints, dep_name, arr_name):
    lats = [p[0] for p in route_coords]
    lons = [p[1] for p in route_coords]

    wp_lats = [wp["lat"] for wp in waypoints]
    wp_lons = [wp["lon"] for wp in waypoints]
    wp_text = [
        f"<b>{wp['name']}</b><br>Lat: {wp['lat']:.2f}<br>Lon: {wp['lon']:.2f}<br>Leg: {wp['distance_from_prev']:.1f} km"
        for wp in waypoints
    ]

    fig = go.Figure()

    # Gradient colored route
    n_points = len(lats)
    color_vals = np.linspace(0, 1, n_points)
    cmap = plotly.colors.sequential.Bluered
    route_colors = [plotly.colors.sample_colorscale(cmap, c)[0] for c in color_vals]

    for k in range(n_points-1):
        fig.add_trace(go.Scattergeo(
            lat=[lats[k], lats[k+1]],
            lon=[lons[k], lons[k+1]],
            mode="lines",
            line=dict(width=5, color=route_colors[k]),
            hoverinfo="skip",
            showlegend=(k==0),
            name="Optimized Route" if k==0 else None,
        ))

    # Straight line (gray dashed)
    fig.add_trace(go.Scattergeo(
        lat=[lats[0], lats[-1]],
        lon=[lons[0], lons[-1]],
        mode="lines",
        line=dict(width=2, color="gray", dash="dash"),
        name="Straight Line",
        hoverinfo="skip"
    ))

    # Waypoints markers
    fig.add_trace(go.Scattergeo(
        lat=wp_lats, lon=wp_lons,
        mode="markers+text",
        marker=dict(size=8, color="orange", symbol="diamond"),
        text=[wp["name"] for wp in waypoints],
        textposition="top center",
        textfont=dict(size=10, color="black"),
        hovertext=wp_text,
        hoverinfo="text+name",
        name="Waypoints"
    ))

    # Departure & Arrival
    fig.add_trace(go.Scattergeo(
        lat=[lats[0]], lon=[lons[0]],
        mode="markers+text",
        marker=dict(size=16, color="green", symbol="star"),
        text=f"<b>{dep_name}</b><br>({lats[0]:.2f}°, {lons[0]:.2f}°)",
        textposition="top center",
        textfont=dict(color="black"),
        hoverinfo="text", name="Departure"
    ))
    fig.add_trace(go.Scattergeo(
        lat=[lats[-1]], lon=[lons[-1]],
        mode="markers+text",
        marker=dict(size=16, color="red", symbol="star"),
        text=f"<b>{arr_name}</b><br>({lats[-1]:.2f}°, {lons[-1]:.2f}°)",
        textposition="bottom center",
        textfont=dict(color="black"),
        hoverinfo="text", name="Arrival"
    ))

    # Auto center & scale map to route
    center_lat = (max(lats)+min(lats))/2
    center_lon = (max(lons)+min(lons))/2
    geo_extent = max(max(lats)-min(lats), max(lons)-min(lons))
    proj_scale = 6.5-geo_extent if geo_extent < 6 else 2.5

    fig.update_layout(
        title=f"✈️ Flight Route: {dep_name} → {arr_name}",
        geo=dict(
            scope="asia",
            projection_type="mercator",
            showland=True,
            showsubunits=True,
            landcolor="rgb(236, 236, 236)",
            showocean=True,
            oceancolor="rgb(219, 242, 255)",
            showcountries=True,
            countrycolor="rgb(175, 175, 175)",
            coastlinecolor="rgb(90, 130, 160)",
            center=dict(lat=center_lat, lon=center_lon),
            projection_scale=proj_scale,
        ),
        height=580,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.015, bgcolor="rgba(247,247,247,0.9)", font=dict(color="black")),
        hovermode="closest",
        dragmode="zoom"
    )

    return fig

def generate_direct_route(start, end, num_points=100):
    lats = np.linspace(start[0], end[0], num_points)
    lons = np.linspace(start[1], end[1], num_points)
    return list(zip(lats, lons))
def generate_latlon_grid(lat1, lon1, lat2, lon2, size=50):
    lats = np.linspace(lat1, lat2, size)
    lons = np.linspace(lon1, lon2, size)
    return lats, lons
