"""Microsoft Building Footprints fallback — 1.4B buildings, free, streaming."""

import gzip
import io
import json
import logging
import math

import httpx

from app.scanner.geo import M_PER_DEG_LAT, meters_per_deg_lng, point_in_polygon

logger = logging.getLogger(__name__)

# Module-level cache: quadkey → tile URL
_url_cache: dict[str, str] = {}

_CSV_URL = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
_MAX_EDGE_DISTANCE_M = 100


def _lat_lng_to_quadkey(lat: float, lng: float, level: int = 9) -> str:
    """Convert lat/lng to a Bing Maps quadkey at the given level."""
    lat_rad = math.radians(lat)
    x = (lng + 180.0) / 360.0
    sin_lat = math.sin(lat_rad)
    y = 0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)

    quadkey = []
    for i in range(level, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        ix = int(x * (1 << level))
        iy = int(y * (1 << level))
        if ix & mask:
            digit += 1
        if iy & mask:
            digit += 2
        quadkey.append(str(digit))
    return "".join(quadkey)


async def _get_tile_url(quadkey: str) -> str | None:
    """Stream the CSV index to find the tile URL for a quadkey. Caches results."""
    if quadkey in _url_cache:
        return _url_cache[quadkey]

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream("GET", _CSV_URL) as resp:
                resp.raise_for_status()
                buffer = b""
                async for chunk in resp.aiter_bytes(8192):
                    buffer += chunk
                    while b"\n" in buffer:
                        line_bytes, buffer = buffer.split(b"\n", 1)
                        line = line_bytes.decode("utf-8", errors="replace").strip()
                        if not line or line.startswith("Location"):
                            continue
                        # CSV format: Location,QuadKey,Url,Rows
                        parts = line.split(",", 3)
                        if len(parts) >= 3:
                            row_qk = parts[1].strip()
                            row_url = parts[2].strip()
                            _url_cache[row_qk] = row_url
                            if row_qk == quadkey:
                                return row_url
    except Exception as e:
        logger.warning("MSFT CSV index fetch failed: %s", e)
    return None


async def _stream_and_filter(
    url: str, lat: float, lng: float, radius_deg: float
) -> list[dict]:
    """Stream-decompress a gzipped GeoJSONL tile, filtering buildings by proximity."""
    buildings = []
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            raw_bytes = resp.content

        gz = gzip.GzipFile(fileobj=io.BytesIO(raw_bytes))
        reader = io.TextIOWrapper(gz, encoding="utf-8")

        idx = 0
        for line in reader:
            line = line.strip()
            if not line:
                continue
            try:
                feature = json.loads(line)
            except json.JSONDecodeError:
                continue

            geom = feature.get("geometry", {})
            coords_raw = geom.get("coordinates", [])
            if not coords_raw or geom.get("type") != "Polygon":
                continue

            # GeoJSON: [lng, lat] — take outer ring only
            ring = coords_raw[0]
            # Quick centroid check before expensive point-in-polygon
            sum_lat = 0.0
            sum_lng = 0.0
            for pt in ring:
                sum_lng += pt[0]
                sum_lat += pt[1]
            n = len(ring)
            c_lat = sum_lat / n
            c_lng = sum_lng / n

            if abs(c_lat - lat) > radius_deg * 2 or abs(c_lng - lng) > radius_deg * 2:
                continue

            # Flip [lng, lat] → (lat, lng) tuples for consistency with OSM
            coords = [(pt[1], pt[0]) for pt in ring]

            props = feature.get("properties", {})
            height = props.get("height", -1)

            buildings.append({
                "id": idx,
                "tags": {"height": height} if height > 0 else {},
                "coords": coords,
            })
            idx += 1

    except Exception as e:
        logger.warning("MSFT tile stream/filter failed: %s", e)

    return buildings


async def query_msft_buildings(
    lat: float, lng: float, search_radius_m: int = 200
) -> list[dict]:
    """Query Microsoft Building Footprints for buildings near coordinates.

    Returns list of building dicts matching OSM format:
    {"id": int, "tags": {"height": H}, "coords": [(lat, lng), ...]}
    Returns [] on any failure.
    """
    try:
        quadkey = _lat_lng_to_quadkey(lat, lng, level=9)
        url = await _get_tile_url(quadkey)
        if not url:
            logger.info("No MSFT tile for quadkey %s (region not covered)", quadkey)
            return []

        radius_deg = search_radius_m / M_PER_DEG_LAT
        buildings = await _stream_and_filter(url, lat, lng, radius_deg)
        logger.info("MSFT: %d buildings near (%.4f, %.4f) [quadkey=%s]", len(buildings), lat, lng, quadkey)
        return buildings

    except Exception as e:
        logger.warning("MSFT query failed: %s", e)
        return []


def _building_edge_distance(b: dict, target_lat: float, target_lng: float) -> float:
    """Minimum distance (m) from target point to nearest vertex of building polygon."""
    m_lng = meters_per_deg_lng(target_lat)
    min_dist = float("inf")
    for blat, blng in b["coords"]:
        dlat = (blat - target_lat) * M_PER_DEG_LAT
        dlng = (blng - target_lng) * m_lng
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


_MERGE_GAP_M = 15


def _bbox_gap_m(a: dict, b: dict, ref_lat: float) -> float:
    """Minimum gap (m) between two buildings' bounding boxes. 0 if they overlap."""
    a_lats = [c[0] for c in a["coords"]]
    a_lngs = [c[1] for c in a["coords"]]
    b_lats = [c[0] for c in b["coords"]]
    b_lngs = [c[1] for c in b["coords"]]

    lat_gap = max(0, max(min(a_lats), min(b_lats)) - min(max(a_lats), max(b_lats)))
    lng_gap = max(0, max(min(a_lngs), min(b_lngs)) - min(max(a_lngs), max(b_lngs)))

    return math.sqrt((lat_gap * M_PER_DEG_LAT) ** 2 + (lng_gap * meters_per_deg_lng(ref_lat)) ** 2)


def _merge_adjacent(seed: dict, all_buildings: list[dict], ref_lat: float) -> dict:
    """Flood-fill merge: starting from seed, absorb buildings within _MERGE_GAP_M."""
    merged_ids = {seed["id"]}
    merged_coords = list(seed["coords"])
    cluster = [seed]

    changed = True
    while changed:
        changed = False
        for b in all_buildings:
            if b["id"] in merged_ids:
                continue
            for m in cluster:
                if _bbox_gap_m(m, b, ref_lat) <= _MERGE_GAP_M:
                    merged_ids.add(b["id"])
                    merged_coords.extend(b["coords"])
                    cluster.append(b)
                    changed = True
                    break

    if len(merged_ids) > 1:
        logger.info("MSFT: merged %d adjacent polygons (ids: %s)", len(merged_ids), list(merged_ids))

    return {
        "id": seed["id"],
        "tags": seed.get("tags", {}),
        "coords": merged_coords,
    }


def find_target_building(
    buildings: list[dict], target_lat: float, target_lng: float
) -> dict | None:
    """Find the facility's building near the target point.

    Seed selection:
    - If pin is inside a building → use it
    - If not → use the LARGEST building within range (not nearest — the
      main building is what the user is pointing at)

    Then flood-fill merge adjacent polygons within 15m gap to capture
    connected sections (loading docks, walkways, split polygons).
    """
    seed = None

    # Check if pin is inside any building
    for b in buildings:
        if point_in_polygon(target_lat, target_lng, b["coords"]):
            area = _building_area(b, target_lat)
            logger.info("MSFT: target inside building %d (~%.0f m²)", b["id"], area)
            seed = b
            break

    # Pin not inside any building — pick the largest within range
    if seed is None:
        candidates = []
        for b in buildings:
            dist = _building_edge_distance(b, target_lat, target_lng)
            if dist <= _MAX_EDGE_DISTANCE_M:
                area = _building_area(b, target_lat)
                candidates.append((b, dist, area))

        if not candidates:
            return None

        # Sort by area descending — pick the biggest
        candidates.sort(key=lambda x: x[2], reverse=True)
        seed, seed_dist, seed_area = candidates[0]
        logger.info("MSFT: selected largest building %d (~%.0f m², %.0fm from pin)", seed["id"], seed_area, seed_dist)

    return _merge_adjacent(seed, buildings, target_lat)
