"""Overture Maps Foundation buildings — 2.3B footprints, MSFT+Google+OSM merged.

Uses STAC catalog to pre-filter parquet files by bounding box, so DuckDB only
reads the 1-3 files covering the target area instead of all 237 (~230GB).
"""

import asyncio
import json
import logging

import duckdb
import httpx

from app.scanner.geo import M_PER_DEG_LAT, meters_per_deg_lng

logger = logging.getLogger(__name__)

# STAC index cache: release → [{bbox: [xmin,ymin,xmax,ymax], href: "s3://..."}, ...]
_stac_cache: dict[str, list[dict]] = {}

# Fallback glob when STAC filtering fails (e.g. catalog unreachable)
_OVERTURE_S3_GLOB = (
    "s3://overturemaps-us-west-2/release/{release}"
    "/theme=buildings/type=building/*.parquet"
)


async def _fetch_stac_index(release: str) -> list[dict]:
    """Fetch and cache the STAC item index for a given Overture release.

    Each item provides a bounding box and the S3 path for one parquet file.
    ~237 items, each ~1KB — fetched in parallel in ~2-3s.
    """
    if release in _stac_cache:
        return _stac_cache[release]

    base = f"https://stac.overturemaps.org/{release}/buildings/building"
    collection_url = f"{base}/collection.json"

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(collection_url)
        resp.raise_for_status()
        collection = resp.json()

        # Extract item hrefs from links with rel="item"
        item_hrefs = [
            link["href"] for link in collection.get("links", [])
            if link.get("rel") == "item"
        ]

        if not item_hrefs:
            raise ValueError(f"No item links in STAC collection for release {release}")

        # Fetch all item JSONs in parallel
        sem = asyncio.Semaphore(50)

        async def fetch_item(href: str) -> dict | None:
            # Resolve relative hrefs (e.g. "./00042/00042.json")
            if href.startswith("./"):
                url = f"{base}/{href[2:]}"
            elif href.startswith("http"):
                url = href
            else:
                url = f"{base}/{href}"
            try:
                async with sem:
                    r = await client.get(url)
                    r.raise_for_status()
                item = r.json()
                bbox = item.get("bbox")
                # Find the S3 parquet asset — nested under alternate.s3
                s3_href = None
                for asset in item.get("assets", {}).values():
                    alt_s3 = asset.get("alternate", {}).get("s3", {})
                    href_val = alt_s3.get("href", "")
                    if href_val.startswith("s3://"):
                        s3_href = href_val
                        break
                if bbox and s3_href:
                    return {"bbox": bbox, "href": s3_href}
            except Exception as e:
                logger.warning("Failed to fetch STAC item %s: %s", url, e)
            return None

        results = await asyncio.gather(*[fetch_item(h) for h in item_hrefs])
        items = [r for r in results if r is not None]

    logger.info("STAC: cached %d building file bboxes for release %s", len(items), release)
    _stac_cache[release] = items
    return items


def _find_matching_files(lat: float, lng: float, release: str) -> list[str]:
    """Find S3 parquet paths whose bounding box contains the given point."""
    items = _stac_cache.get(release, [])
    matches = []
    for item in items:
        b = item["bbox"]
        # STAC bbox: [west, south, east, north] or 6-element [w, s, zmin, e, n, zmax]
        if len(b) == 6:
            xmin, ymin, xmax, ymax = b[0], b[1], b[3], b[4]
        else:
            xmin, ymin, xmax, ymax = b
        if xmin <= lng <= xmax and ymin <= lat <= ymax:
            matches.append(item["href"])

    if not matches:
        logger.warning(
            "STAC: no file bbox contains (%.4f, %.4f), falling back to glob", lat, lng
        )
        return [_OVERTURE_S3_GLOB.format(release=release)]

    logger.info("STAC: %d file(s) match (%.4f, %.4f)", len(matches), lat, lng)
    return matches


def _new_connection():
    """Create a fresh DuckDB connection with spatial + httpfs extensions.

    Each query gets its own connection that is closed after use,
    so no memory accumulates between scans.
    """
    con = duckdb.connect()
    con.install_extension("spatial")
    con.load_extension("spatial")
    con.install_extension("httpfs")
    con.load_extension("httpfs")
    con.sql("SET s3_region='us-west-2'")
    con.sql("SET memory_limit='128MB'")
    con.sql("SET threads=2")
    return con


def _query_sync(lat: float, lng: float, search_radius_m: int, s3_paths: list[str]) -> list[dict]:
    """Run DuckDB query against specific Overture S3 files (blocking)."""
    dlat = search_radius_m / M_PER_DEG_LAT
    dlng = search_radius_m / meters_per_deg_lng(lat)
    min_lat = lat - dlat
    max_lat = lat + dlat
    min_lng = lng - dlng
    max_lng = lng + dlng

    con = _new_connection()

    # Build file list as a SQL literal — parameterized binding for read_parquet's
    # file list argument is unreliable across DuckDB versions.
    escaped = ", ".join(f"'{p}'" for p in s3_paths)
    path_list_sql = f"[{escaped}]"

    query = f"""
        SELECT
            id,
            ST_AsGeoJSON(geometry) AS geojson,
            height,
            num_floors
        FROM read_parquet({path_list_sql}, hive_partitioning=1)
        WHERE bbox.xmin <= ?
          AND bbox.xmax >= ?
          AND bbox.ymin <= ?
          AND bbox.ymax >= ?
    """

    try:
        result = con.execute(query, [max_lng, min_lng, max_lat, min_lat]).fetchall()
    except Exception as e:
        logger.error("Overture DuckDB query failed for %d file(s): %s", len(s3_paths), e)
        return []
    finally:
        con.close()

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

    Uses STAC catalog to pre-filter to only the parquet files covering the
    target area, then runs DuckDB against those 1-3 files instead of all 237.
    Returns list of building dicts matching MSFT/OSM format:
    {"id": int, "tags": {...}, "coords": [(lat, lng), ...]}
    Returns [] on any failure.
    """
    try:
        # Fetch STAC index (no-op if already cached for this release)
        try:
            await _fetch_stac_index(release)
            s3_paths = _find_matching_files(lat, lng, release)
        except Exception as e:
            logger.warning("STAC index fetch failed, falling back to glob: %s", e)
            s3_paths = [_OVERTURE_S3_GLOB.format(release=release)]

        buildings = await asyncio.to_thread(
            _query_sync, lat, lng, search_radius_m, s3_paths
        )
        logger.info(
            "Overture: %d buildings near (%.4f, %.4f) [release=%s, files=%d]",
            len(buildings), lat, lng, release, len(s3_paths),
        )
        return buildings
    except Exception as e:
        logger.warning("Overture query failed: %s", e)
        return []
