"""Tile download, stitching, and cropping — adapted from vision_scanner.py."""

import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import httpx
from PIL import Image

from app.config import settings
from app.scanner.cache import TileCache
from app.scanner.geo import TILE_SIZE, lat_lng_to_tile, tile_to_lat_lng

logger = logging.getLogger(__name__)

# Pillow pixel limit — we build large stitched images ourselves
Image.MAX_IMAGE_PIXELS = None

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    "Referer": "https://www.google.com/maps",
}


def download_tile(
    x: int, y: int, zoom: int, cache: TileCache | None = None, delay: float = 0.05,
    client: httpx.Client | None = None, max_retries: int = 2,
) -> bytes | None:
    """Download a single satellite tile, using cache if available.

    If *client* is provided it will be used instead of creating a fresh one
    per call (much better for concurrent workloads — shared connection pool).
    Retries on 429/503 with exponential backoff.
    """
    if cache is not None:
        cached = cache.get(x, y, zoom)
        if cached is not None:
            return cached

    url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}"
    own_client = client is None
    try:
        c = httpx.Client(timeout=15, headers=HEADERS) if own_client else client
        try:
            for attempt in range(1 + max_retries):
                resp = c.get(url)
                if resp.status_code in (429, 503) and attempt < max_retries:
                    wait = 2 ** attempt  # 1s, 2s
                    logger.warning(
                        "Tile (%d,%d) got %d, retry %d/%d in %ds",
                        x, y, resp.status_code, attempt + 1, max_retries, wait,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.content
                if cache is not None and data:
                    cache.put(x, y, zoom, data)
                if delay > 0:
                    time.sleep(delay)
                return data
        finally:
            if own_client:
                c.close()
    except Exception as e:
        logger.warning("Error downloading tile (%d,%d): %s", x, y, e)
        return None


def _stitch_chunked(
    tiles_data: dict, num_x: int, num_y: int,
    tile_min_x: int, tile_min_y: int, tile_size: int, temp_dir: str
) -> Image.Image:
    """Stitch tiles in row-bands to limit peak memory."""
    full_w = num_x * tile_size
    full_h = num_y * tile_size
    rows_per_band = 8
    band_files = []

    for band_start in range(0, num_y, rows_per_band):
        band_end = min(band_start + rows_per_band, num_y)
        band_h = (band_end - band_start) * tile_size
        band_img = Image.new("RGB", (full_w, band_h))

        for row in range(band_start, band_end):
            for col in range(num_x):
                tx = tile_min_x + col
                ty = tile_min_y + row
                key = f"{tx},{ty}"
                tile_bytes = tiles_data.get(key)
                if tile_bytes:
                    tile_img = Image.open(BytesIO(tile_bytes))
                    band_img.paste(tile_img, (col * tile_size, (row - band_start) * tile_size))
                    tile_img.close()

        band_path = os.path.join(temp_dir, f"band_{band_start:04d}.png")
        band_img.save(band_path)
        band_img.close()
        band_files.append((band_path, band_h))

    # Combine bands
    stitched = Image.new("RGB", (full_w, full_h))
    y_offset = 0
    for band_path, band_h in band_files:
        band_img = Image.open(band_path)
        stitched.paste(band_img, (0, y_offset))
        band_img.close()
        y_offset += band_h
        try:
            os.unlink(band_path)
        except OSError:
            pass

    return stitched


def capture_area(
    min_lat: float, min_lng: float, max_lat: float, max_lng: float,
    zoom: int, cache: TileCache | None = None, delay: float = 0.05,
    max_image_mb: int = 512, max_workers: int | None = None,
) -> tuple[Image.Image | None, dict | None]:
    """Download and stitch tiles covering a bounding box.

    Downloads tiles in parallel using a thread pool (*max_workers* defaults to
    ``settings.tile_concurrency``).  The download phase is fully parallel; the
    stitch phase runs sequentially afterwards to avoid PIL thread-safety issues.

    Returns (image, grid_info) or (None, None) on failure.
    """
    if max_workers is None:
        max_workers = settings.tile_concurrency

    tile_min_x, tile_max_y = lat_lng_to_tile(min_lat, min_lng, zoom)
    tile_max_x, tile_min_y = lat_lng_to_tile(max_lat, max_lng, zoom)

    if tile_min_x > tile_max_x:
        tile_min_x, tile_max_x = tile_max_x, tile_min_x
    if tile_min_y > tile_max_y:
        tile_min_y, tile_max_y = tile_max_y, tile_min_y

    num_x = tile_max_x - tile_min_x + 1
    num_y = tile_max_y - tile_min_y + 1
    total = num_x * num_y
    full_w = num_x * TILE_SIZE
    full_h = num_y * TILE_SIZE

    logger.info("Tile grid: %dx%d = %d tiles at zoom %d", num_x, num_y, total, zoom)

    # --- parallel download phase -------------------------------------------
    tile_coords = [
        (tx, ty)
        for ty in range(tile_min_y, tile_max_y + 1)
        for tx in range(tile_min_x, tile_max_x + 1)
    ]

    tiles_data: dict[str, bytes] = {}
    with httpx.Client(timeout=15, headers=HEADERS) as shared_client:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    download_tile, tx, ty, zoom,
                    cache=cache, delay=delay, client=shared_client,
                ): (tx, ty)
                for tx, ty in tile_coords
            }
            for future in as_completed(futures):
                tx, ty = futures[future]
                try:
                    data = future.result()
                    if data:
                        tiles_data[f"{tx},{ty}"] = data
                except Exception as e:
                    logger.warning("Tile (%d,%d) failed: %s", tx, ty, e)

    downloaded = len(tiles_data)
    logger.info("Downloaded %d/%d tiles", downloaded, total)

    if downloaded == 0:
        return None, None

    # --- sequential stitch phase -------------------------------------------
    stitched_mb = (full_w * full_h * 3) / (1024 * 1024)
    use_chunked = stitched_mb > max_image_mb

    if use_chunked:
        logger.info("Using chunked stitching (%.0f MB > %d MB limit)", stitched_mb, max_image_mb)
        temp_dir = tempfile.mkdtemp(prefix="sat_bands_")
        try:
            stitched = _stitch_chunked(
                tiles_data, num_x, num_y, tile_min_x, tile_min_y, TILE_SIZE, temp_dir
            )
        finally:
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass
        tiles_data.clear()
    else:
        stitched = Image.new("RGB", (full_w, full_h))
        for ty in range(tile_min_y, tile_max_y + 1):
            for tx in range(tile_min_x, tile_max_x + 1):
                key = f"{tx},{ty}"
                tile_bytes = tiles_data.get(key)
                if tile_bytes:
                    tile_img = Image.open(BytesIO(tile_bytes))
                    col = tx - tile_min_x
                    row = ty - tile_min_y
                    stitched.paste(tile_img, (col * TILE_SIZE, row * TILE_SIZE))
                    tile_img.close()

    grid_info = {
        "tile_min_x": tile_min_x, "tile_max_x": tile_max_x,
        "tile_min_y": tile_min_y, "tile_max_y": tile_max_y,
        "zoom": zoom, "full_w": full_w, "full_h": full_h,
        "downloaded": downloaded, "total": total,
    }
    return stitched, grid_info


def crop_to_bbox(
    image: Image.Image, min_lat: float, min_lng: float,
    max_lat: float, max_lng: float, grid_info: dict
) -> Image.Image:
    """Crop a stitched image to exact lat/lng bounds."""
    zoom = grid_info["zoom"]
    nw_lat, nw_lng = tile_to_lat_lng(grid_info["tile_min_x"], grid_info["tile_min_y"], zoom)
    se_lat, se_lng = tile_to_lat_lng(
        grid_info["tile_max_x"] + 1, grid_info["tile_max_y"] + 1, zoom
    )

    px_per_deg_lng = grid_info["full_w"] / (se_lng - nw_lng)
    px_per_deg_lat = grid_info["full_h"] / (nw_lat - se_lat)

    crop_left = max(0, int((min_lng - nw_lng) * px_per_deg_lng))
    crop_right = min(grid_info["full_w"], int((max_lng - nw_lng) * px_per_deg_lng))
    crop_top = max(0, int((nw_lat - max_lat) * px_per_deg_lat))
    crop_bottom = min(grid_info["full_h"], int((nw_lat - min_lat) * px_per_deg_lat))

    if crop_right > crop_left and crop_bottom > crop_top:
        return image.crop((crop_left, crop_top, crop_right, crop_bottom))
    return image
