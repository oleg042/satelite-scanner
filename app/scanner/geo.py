"""Coordinate math utilities for tile mapping and bounding boxes."""

import math

TILE_SIZE = 256
M_PER_DEG_LAT = 111_320


def meters_per_deg_lng(lat: float) -> float:
    """Meters per degree of longitude at a given latitude."""
    return M_PER_DEG_LAT * math.cos(math.radians(lat))


def lat_lng_to_tile(lat: float, lng: float, zoom: int) -> tuple[int, int]:
    """Convert lat/lng to tile (x, y) at a given zoom level."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lng + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_to_lat_lng(x: int, y: int, zoom: int) -> tuple[float, float]:
    """Convert tile (x, y) back to lat/lng (NW corner of tile)."""
    n = 2 ** zoom
    lng = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lng


def bbox_add_buffer(
    min_lat: float, min_lng: float, max_lat: float, max_lng: float, buffer_m: float
) -> tuple[float, float, float, float]:
    """Expand a bounding box by buffer_m meters in all directions."""
    lat_buf = buffer_m / M_PER_DEG_LAT
    center_lat = (min_lat + max_lat) / 2
    lng_buf = buffer_m / meters_per_deg_lng(center_lat)
    return (min_lat - lat_buf, min_lng - lng_buf, max_lat + lat_buf, max_lng + lng_buf)


def point_in_polygon(lat: float, lng: float, polygon_coords: list[tuple[float, float]]) -> bool:
    """Ray-casting algorithm to check if a point is inside a polygon."""
    n = len(polygon_coords)
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = polygon_coords[i]
        yj, xj = polygon_coords[j]
        if ((yi > lng) != (yj > lng)) and (lat < (xj - xi) * (lng - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def pixel_to_latlng(
    px: int, py: int, grid_info: dict
) -> tuple[float, float]:
    """Convert pixel coordinates in a stitched image to lat/lng."""
    zoom = grid_info["zoom"]
    nw_lat, nw_lng = tile_to_lat_lng(grid_info["tile_min_x"], grid_info["tile_min_y"], zoom)
    se_lat, se_lng = tile_to_lat_lng(
        grid_info["tile_max_x"] + 1, grid_info["tile_max_y"] + 1, zoom
    )
    lng = nw_lng + (se_lng - nw_lng) * (px / grid_info["full_w"])
    lat = nw_lat + (se_lat - nw_lat) * (py / grid_info["full_h"])
    return lat, lng


def bbox_dimensions_m(
    min_lat: float, min_lng: float, max_lat: float, max_lng: float
) -> tuple[float, float]:
    """Return (width_m, height_m) of a bounding box."""
    center_lat = (min_lat + max_lat) / 2
    width_m = (max_lng - min_lng) * meters_per_deg_lng(center_lat)
    height_m = (max_lat - min_lat) * M_PER_DEG_LAT
    return width_m, height_m
