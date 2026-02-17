"""Tile download, stitching, and cropping — adapted from vision_scanner.py."""

import logging
import os
import tempfile
import time
from io import BytesIO

import httpx
from PIL import Image

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
    x: int, y: int, zoom: int, cache: TileCache | None = None, delay: float = 0.05
) -> bytes | None:
    """Download a single satellite tile, using cache if available."""
    if cache is not None:
        cached = cache.get(x, y, zoom)
        if cached is not None:
            return cached

    url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}"
    try:
        with httpx.Client(timeout=15, headers=HEADERS) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.content
        if cache is not None and data:
            cache.put(x, y, zoom, data)
        if delay > 0:
            time.sleep(delay)
        return data
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
    max_image_mb: int = 512,
) -> tuple[Image.Image | None, dict | None]:
    """Download and stitch tiles covering a bounding box.

    Returns (image, grid_info) or (None, None) on failure.
    """
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

    # Estimate memory: 3 bytes/pixel for RGB
    stitched_mb = (full_w * full_h * 3) / (1024 * 1024)
    use_chunked = stitched_mb > max_image_mb

    if use_chunked:
        logger.info("Using chunked stitching (%.0f MB > %d MB limit)", stitched_mb, max_image_mb)
        tiles_data: dict[str, bytes] = {}
        stitched = None
    else:
        tiles_data = None
        stitched = Image.new("RGB", (full_w, full_h))

    downloaded = 0
    for ty in range(tile_min_y, tile_max_y + 1):
        for tx in range(tile_min_x, tile_max_x + 1):
            col = tx - tile_min_x
            row = ty - tile_min_y

            tile_data = download_tile(tx, ty, zoom, cache=cache, delay=delay)
            if tile_data:
                if use_chunked:
                    tiles_data[f"{tx},{ty}"] = tile_data
                else:
                    tile_img = Image.open(BytesIO(tile_data))
                    stitched.paste(tile_img, (col * TILE_SIZE, row * TILE_SIZE))
                    tile_img.close()
                downloaded += 1

    logger.info("Downloaded %d/%d tiles", downloaded, total)

    if downloaded == 0:
        return None, None

    # Chunked stitching if needed
    if use_chunked:
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
