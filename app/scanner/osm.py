"""OSM Overpass queries for building/landuse detection."""

import logging
import math
import urllib.parse

import httpx

from app.scanner.geo import M_PER_DEG_LAT, meters_per_deg_lng, point_in_polygon

logger = logging.getLogger(__name__)

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


async def _query_overpass(query: str) -> dict | None:
    """Send a query to Overpass API with endpoint failover."""
    encoded_data = urllib.parse.urlencode({"data": query})
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    async with httpx.AsyncClient(timeout=30) as client:
        for endpoint in OVERPASS_ENDPOINTS:
            try:
                resp = await client.post(endpoint, content=encoded_data, headers=headers)
                resp.raise_for_status()
                result = resp.json()
                if result and result.get("elements"):
                    logger.info("Overpass query via %s", endpoint.split("//")[1].split("/")[0])
                    return result
            except Exception as e:
                logger.warning("Overpass %s failed: %s", endpoint.split("//")[1].split("/")[0], e)
    return None


async def query_osm_buildings(
    lat: float, lng: float, search_radius_m: int = 200
) -> list[dict]:
    """Query OSM for building footprints near coordinates."""
    query = (
        f'[out:json][timeout:25];'
        f'(way["building"](around:{search_radius_m},{lat},{lng});'
        f'relation["building"](around:{search_radius_m},{lat},{lng}););'
        f'out body;>;out skel qt;'
    )
    result = await _query_overpass(query)
    if result is None:
        logger.warning("OSM building query failed on all endpoints")
        return []

    # Two-pass parse: nodes first, then resolve ways
    nodes = {}
    ways = []
    for el in result["elements"]:
        if el["type"] == "node":
            nodes[el["id"]] = (el["lat"], el["lon"])
        elif el["type"] == "way" and "building" in el.get("tags", {}):
            ways.append(el)

    buildings = []
    for w in ways:
        coords = [nodes[nid] for nid in w.get("nodes", []) if nid in nodes]
        if coords:
            buildings.append({"id": w["id"], "tags": w.get("tags", {}), "coords": coords})

    logger.info("Found %d buildings (%d nodes, %d ways)", len(buildings), len(nodes), len(ways))
    return buildings


async def query_osm_landuse(
    lat: float, lng: float, search_radius_m: int = 1500
) -> list[dict]:
    """Query OSM for landuse=industrial polygons near coordinates."""
    query = (
        f'[out:json][timeout:25];'
        f'(way["landuse"="industrial"](around:{search_radius_m},{lat},{lng});'
        f'relation["landuse"="industrial"](around:{search_radius_m},{lat},{lng}););'
        f'out body;>;out skel qt;'
    )
    result = await _query_overpass(query)
    if result is None:
        logger.warning("OSM landuse query failed on all endpoints")
        return []

    nodes = {}
    polygons = []
    for el in result["elements"]:
        if el["type"] == "node":
            nodes[el["id"]] = (el["lat"], el["lon"])
        elif el["type"] == "way" and el.get("tags", {}).get("landuse") == "industrial":
            polygons.append(el)

    landuse_areas = []
    for w in polygons:
        coords = [nodes[nid] for nid in w.get("nodes", []) if nid in nodes]
        if coords:
            landuse_areas.append({"id": w["id"], "tags": w.get("tags", {}), "coords": coords})

    logger.info("Found %d landuse=industrial areas", len(landuse_areas))
    return landuse_areas


def _building_edge_distance(
    b: dict, target_lat: float, target_lng: float
) -> float:
    """Minimum distance (m) from target point to nearest edge of building polygon."""
    m_lng = meters_per_deg_lng(target_lat)
    min_dist = float("inf")
    for lat, lng in b["coords"]:
        dlat = (lat - target_lat) * M_PER_DEG_LAT
        dlng = (lng - target_lng) * m_lng
        dist = math.sqrt(dlat**2 + dlng**2)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def _building_area(b: dict, ref_lat: float) -> float:
    """Approximate area (m²) of a building's bounding box."""
    lats = [c[0] for c in b["coords"]]
    lngs = [c[1] for c in b["coords"]]
    return ((max(lats) - min(lats)) * M_PER_DEG_LAT) * (
        (max(lngs) - min(lngs)) * meters_per_deg_lng(ref_lat)
    )


# Max distance (m) from pin to nearest building edge before we give up
# and let AI vision handle it instead of guessing
_MAX_EDGE_DISTANCE_M = 100


def find_target_building(
    buildings: list[dict], target_lat: float, target_lng: float
) -> dict | None:
    """Find the building containing the target point, or the nearest one within range."""
    # Check if target point is inside any building polygon
    for b in buildings:
        if point_in_polygon(target_lat, target_lng, b["coords"]):
            area = _building_area(b, target_lat)
            logger.info("Target inside building %d (~%.0f m²)", b["id"], area)
            return b

    # Pick the nearest building by edge distance
    ranked = []
    for b in buildings:
        edge_dist = _building_edge_distance(b, target_lat, target_lng)
        area = _building_area(b, target_lat)
        ranked.append((b, edge_dist, area))

    if not ranked:
        return None

    ranked.sort(key=lambda x: x[1])
    best, best_dist, best_area = ranked[0]

    if best_dist > _MAX_EDGE_DISTANCE_M:
        logger.info(
            "Nearest building %d is %.0fm away (>%dm) — deferring to AI vision",
            best["id"], best_dist, _MAX_EDGE_DISTANCE_M,
        )
        return None

    logger.info("Selected nearest building %d (~%.0f m², %.0fm from pin edge)", best["id"], best_area, best_dist)
    return best


def find_target_landuse(
    landuse_areas: list[dict], target_lat: float, target_lng: float
) -> dict | None:
    """Find the landuse polygon containing the target point, or the closest one."""
    for area in landuse_areas:
        if point_in_polygon(target_lat, target_lng, area["coords"]):
            return area

    # Nearest by center distance
    best = None
    best_dist = float("inf")
    for area in landuse_areas:
        lats = [c[0] for c in area["coords"]]
        lngs = [c[1] for c in area["coords"]]
        clat = sum(lats) / len(lats)
        clng = sum(lngs) / len(lngs)
        dlat = (clat - target_lat) * M_PER_DEG_LAT
        dlng = (clng - target_lng) * meters_per_deg_lng(target_lat)
        dist = math.sqrt(dlat**2 + dlng**2)
        if dist < best_dist:
            best_dist = dist
            best = area

    return best
