"""Overture Maps Foundation buildings — 2.3B footprints, MSFT+Google+OSM merged."""

import asyncio
import json
import logging

import duckdb

from app.scanner.geo import M_PER_DEG_LAT, meters_per_deg_lng

logger = logging.getLogger(__name__)

OVERTURE_S3 = (
    "s3://overturemaps-us-west-2/release/{release}"
    "/theme=buildings/type=building/*.parquet"
)

_con = None


def _get_connection():
    """Lazily create a DuckDB connection with spatial + httpfs extensions."""
    global _con
    if _con is None:
        _con = duckdb.connect()
        _con.install_extension("spatial")
        _con.load_extension("spatial")
        _con.install_extension("httpfs")
        _con.load_extension("httpfs")
        _con.sql("SET s3_region='us-west-2'")
        _con.sql("SET enable_http_metadata_cache = true")
    return _con


def _query_sync(lat: float, lng: float, search_radius_m: int, release: str) -> list[dict]:
    """Run DuckDB query against Overture S3 (blocking). Called from thread pool."""
    dlat = search_radius_m / M_PER_DEG_LAT
    dlng = search_radius_m / meters_per_deg_lng(lat)
    min_lat = lat - dlat
    max_lat = lat + dlat
    min_lng = lng - dlng
    max_lng = lng + dlng

    con = _get_connection()
    s3_path = OVERTURE_S3.format(release=release)

    query = """
        SELECT
            id,
            ST_AsGeoJSON(geometry) AS geojson,
            height,
            num_floors
        FROM read_parquet(?, hive_partitioning=1)
        WHERE bbox.xmin <= ?
          AND bbox.xmax >= ?
          AND bbox.ymin <= ?
          AND bbox.ymax >= ?
    """

    try:
        result = con.execute(query, [s3_path, max_lng, min_lng, max_lat, min_lat]).fetchall()
    except Exception as e:
        logger.error("Overture DuckDB query failed for S3 path '%s': %s", s3_path, e)
        return []

    buildings = []
    idx = 0
    for row in result:
        overture_id, geojson_str, height, num_floors = row
        try:
            geom = json.loads(geojson_str)
        except (json.JSONDecodeError, TypeError):
            continue

        geom_type = geom.get("type")
        coords_raw = geom.get("coordinates", [])
        if not coords_raw:
            continue

        if geom_type == "Polygon":
            rings = [coords_raw[0]]
        elif geom_type == "MultiPolygon":
            rings = [poly[0] for poly in coords_raw]
        else:
            continue

        tags = {}
        if height and height > 0:
            tags["height"] = height
        if num_floors and num_floors > 0:
            tags["num_floors"] = num_floors

        for ring in rings:
            # Quick centroid proximity check
            sum_lat = sum(pt[1] for pt in ring)
            sum_lng = sum(pt[0] for pt in ring)
            n = len(ring)
            if n == 0:
                continue
            c_lat = sum_lat / n
            c_lng = sum_lng / n

            if abs(c_lat - lat) > dlat * 2 or abs(c_lng - lng) > dlng * 2:
                continue

            # Flip [lng, lat] → (lat, lng) for consistency with OSM/MSFT
            coords = [(pt[1], pt[0]) for pt in ring]

            buildings.append({
                "id": idx,
                "tags": tags,
                "coords": coords,
            })
            idx += 1

    return buildings


async def query_overture_buildings(
    lat: float, lng: float, search_radius_m: int = 200, release: str = "2026-02-18.0"
) -> list[dict]:
    """Query Overture Maps for buildings near coordinates.

    Runs the blocking DuckDB query in a thread pool to avoid blocking the event loop.
    Returns list of building dicts matching MSFT/OSM format:
    {"id": int, "tags": {...}, "coords": [(lat, lng), ...]}
    Returns [] on any failure.
    """
    try:
        buildings = await asyncio.to_thread(
            _query_sync, lat, lng, search_radius_m, release
        )
        logger.info(
            "Overture: %d buildings near (%.4f, %.4f) [release=%s]",
            len(buildings), lat, lng, release,
        )
        return buildings
    except Exception as e:
        logger.warning("Overture query failed: %s", e)
        return []
