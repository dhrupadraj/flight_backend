
from model.Astarinference import astar_search


def find_optimized_path(start_idx, goal_idx, lat_grid, lon_grid, wind_u, wind_v, v_air=250.0):
    return astar_search(
        start_idx,
        goal_idx,
        lat_grid,
        lon_grid,
        wind_u,
        wind_v,
        v_air=v_air,
    )
