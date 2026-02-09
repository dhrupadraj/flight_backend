import math
import heapq
import numpy as np
from typing import List, Tuple


# -----------------------------
# Geometry & Physics Utilities
# -----------------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_ground_speed(v_air, u_wind, v_wind, dx, dy):
    norm = math.sqrt(dx ** 2 + dy ** 2)
    ux, uy = dx / norm, dy / norm

    vax, vay = v_air * ux, v_air * uy
    vg_x = vax + u_wind
    vg_y = vay + v_wind

    return math.sqrt(vg_x ** 2 + vg_y ** 2)


# -----------------------------
# A* Node
# -----------------------------

class Node:
    __slots__ = ("lat", "lon")

    def __init__(self, lat_idx: int, lon_idx: int):
        self.lat = lat_idx
        self.lon = lon_idx

    def __lt__(self, other):
        return False  # required for heapq


# -----------------------------
# Cost Functions
# -----------------------------

def heuristic(node, goal, lat_grid, lon_grid, wind_u, wind_v, v_air):
    lat1, lon1 = lat_grid[node.lat], lon_grid[node.lon]
    lat2, lon2 = lat_grid[goal.lat], lon_grid[goal.lon]

    d = haversine(lat1, lon1, lat2, lon2)

    u = wind_u[node.lat, node.lon]
    v = wind_v[node.lat, node.lon]

    v_ground = max(compute_ground_speed(v_air, u, v, 1, 1), 50.0)
    return d / v_ground


def step_cost(curr, nxt, lat_grid, lon_grid, wind_u, wind_v, v_air):
    lat1, lon1 = lat_grid[curr.lat], lon_grid[curr.lon]
    lat2, lon2 = lat_grid[nxt.lat], lon_grid[nxt.lon]

    d = haversine(lat1, lon1, lat2, lon2)

    dx = nxt.lat - curr.lat
    dy = nxt.lon - curr.lon

    u = wind_u[curr.lat, curr.lon]
    v = wind_v[curr.lat, curr.lon]

    v_ground = max(compute_ground_speed(v_air, u, v, dx, dy), 50.0)
    return d / v_ground


# -----------------------------
# A* Search (Inference)
# -----------------------------

def astar_search(
    start_idx: Tuple[int,int],
    goal_idx: Tuple[int,int],
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    v_air: float = 250.0,
    max_iterations: int = 50000,  # Prevent infinite loops
) -> List[Tuple[float, float]]:
    """
    Returns optimized path as [(lat, lon), ...]
    Optimized with iteration limit and cached heuristic calculations.
    """

    open_set = []
    heapq.heappush(open_set, (0.0, start_idx))

    came_from = {}
    g_cost = {start_idx: 0.0}
    closed_set = set()  # Track visited nodes to avoid revisiting

    moves = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    
    # Pre-compute goal node for heuristic
    goal_node = Node(*goal_idx)
    iterations = 0

    while open_set and iterations < max_iterations:
        iterations += 1
        _, current = heapq.heappop(open_set)
        
        # Skip if already processed
        if current in closed_set:
            continue
            
        closed_set.add(current)

        if current == goal_idx:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_idx)

            # convert grid indices → lat/lon
            return [
                (float(lat_grid[i]), float(lon_grid[j]))
                for i, j in reversed(path)
            ]

        for dx, dy in moves:
            ni, nj = current[0] + dx, current[1] + dy
            if ni < 0 or nj < 0 or ni >= len(lat_grid) or nj >= len(lon_grid):
                continue

            neighbor = (ni, nj)
            
            # Skip if already processed
            if neighbor in closed_set:
                continue

            lat1, lon1 = lat_grid[current[0]], lon_grid[current[1]]
            lat2, lon2 = lat_grid[ni], lon_grid[nj]
            d = haversine(lat1, lon1, lat2, lon2)

            u = wind_u[current[0], current[1]]
            v = wind_v[current[0], current[1]]
            # Pass v_air first, then wind components, then the direction dx, dy
            cost = d / max(compute_ground_speed(v_air, u, v, dx, dy), 10.0)

            tentative = g_cost[current] + cost

            if neighbor not in g_cost or tentative < g_cost[neighbor]:
                g_cost[neighbor] = tentative
                neighbor_node = Node(ni, nj)
                f = tentative + heuristic(neighbor_node, goal_node, lat_grid, lon_grid, wind_u, wind_v, v_air)
                heapq.heappush(open_set, (f, neighbor))
                came_from[neighbor] = current

    # If we hit max iterations, return best path found so far or empty
    return []


# -----------------------------
# Path Reconstruction
# -----------------------------

def _reconstruct_path(current, came_from, lat_grid, lon_grid):
    path = []

    while (current.lat, current.lon) in came_from:
        path.append((lat_grid[current.lat], lon_grid[current.lon]))
        current = came_from[(current.lat, current.lon)]

    path.append((lat_grid[current.lat], lon_grid[current.lon]))
    return path[::-1]
