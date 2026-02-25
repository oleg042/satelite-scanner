"""API router — aggregates all endpoint modules."""

import asyncio
import logging
import os
import re
import shutil
import urllib.parse

import httpx
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session, get_db
from app.models import Screenshot, Setting
from app.schemas import HealthResponse, SettingsResponse, SettingsUpdate
from app.config import settings as app_settings
from app.scanner.geocode import serper_maps_resolve
from app.worker import scan_queue

from app.api.scans import router as scans_router
from app.api.screenshots import router as screenshots_router

logger = logging.getLogger(__name__)

# --- Persistent headless browser (shared across requests) ---
_pw = None          # Playwright instance
_browser = None     # Chromium browser instance
_browser_lock = asyncio.Lock()
_browser_semaphore: asyncio.Semaphore | None = None


def init_browser_semaphore(concurrency: int | None = None):
    """Initialize the browser semaphore. Called once at startup from lifespan."""
    global _browser_semaphore
    n = concurrency if concurrency is not None else app_settings.browser_concurrency
    _browser_semaphore = asyncio.Semaphore(n)  # ~150MB each


async def _get_browser():
    """Return the persistent Chromium browser, launching it lazily on first use."""
    global _pw, _browser
    async with _browser_lock:
        if _browser and _browser.is_connected():
            return _browser
        from playwright.async_api import async_playwright
        if _pw is None:
            _pw = await async_playwright().start()
        _browser = await _pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        logger.info("Persistent Chromium browser launched")
    return _browser


async def shutdown_browser():
    """Close the persistent browser and stop Playwright. Called at app shutdown."""
    global _pw, _browser
    if _browser:
        try:
            await _browser.close()
        except Exception:
            pass
        _browser = None
    if _pw:
        try:
            await _pw.stop()
        except Exception:
            pass
        _pw = None
    logger.info("Browser shut down")


async def _get_setting(db: AsyncSession, key: str) -> str | None:
    """Read a single setting value from the DB."""
    result = await db.execute(select(Setting).where(Setting.key == key))
    setting = result.scalar_one_or_none()
    return setting.value if setting else None


api_router = APIRouter(prefix="/api")

# Include sub-routers
api_router.include_router(scans_router)
api_router.include_router(screenshots_router)


# --- Auth endpoints ---

class LoginRequest(BaseModel):
    password: str


@api_router.post("/login")
async def login(req: LoginRequest):
    from app.auth import verify_password, create_session
    if not await verify_password(req.password):
        return JSONResponse({"detail": "Invalid password"}, status_code=401)
    token = await create_session()
    resp = JSONResponse({"ok": True})
    resp.set_cookie("session", token, httponly=True, samesite="lax", max_age=86400)
    return resp


@api_router.get("/auth/check")
async def auth_check(request: Request):
    from app.auth import validate_session
    token = request.cookies.get("session")
    if validate_session(token):
        return {"authenticated": True}
    return JSONResponse({"detail": "Not authenticated"}, status_code=401)


# --- Settings endpoints (inline since they're small) ---

@api_router.get("/settings", response_model=SettingsResponse)
async def get_settings(db: AsyncSession = Depends(get_db)):
    """Get current app settings."""
    result = await db.execute(select(Setting))
    rows = result.scalars().all()
    settings_dict = {s.key: s.value for s in rows}
    settings_dict.pop("app_password", None)

    # Mask API keys for security
    api_key = settings_dict.get("openai_api_key", "")
    if api_key and len(api_key) > 8:
        api_key = api_key[:4] + "..." + api_key[-4:]

    serper_key = settings_dict.get("serper_api_key", "")
    if serper_key and len(serper_key) > 8:
        serper_key = serper_key[:4] + "..." + serper_key[-4:]

    return SettingsResponse(
        openai_api_key=api_key,
        serper_api_key=serper_key,
        validation_model=settings_dict.get("validation_model", ""),
        boundary_model=settings_dict.get("boundary_model", ""),
        default_zoom=settings_dict.get("default_zoom", ""),
        default_buffer_m=settings_dict.get("default_buffer_m", ""),
        overview_zoom=settings_dict.get("overview_zoom", ""),
        validation_prompt=settings_dict.get("validation_prompt", ""),
        boundary_prompt=settings_dict.get("boundary_prompt", ""),
        verification_prompt=settings_dict.get("verification_prompt", ""),
        verification_model=settings_dict.get("verification_model", ""),
        correction_mode=settings_dict.get("correction_mode", ""),
        verification_correction_prompt=settings_dict.get("verification_correction_prompt", ""),
        bbox_validation_enabled=settings_dict.get("bbox_validation_enabled", "true"),
        # Bin detection
        bin_detection_enabled=settings_dict.get("bin_detection_enabled", "false"),
        bin_detection_prompt=settings_dict.get("bin_detection_prompt", ""),
        bin_detection_model=settings_dict.get("bin_detection_model", ""),
        bin_detection_max_chunk_m=settings_dict.get("bin_detection_max_chunk_m", ""),
        bin_detection_min_confidence=settings_dict.get("bin_detection_min_confidence", ""),
        bin_detection_tentative_confidence=settings_dict.get("bin_detection_tentative_confidence", ""),
        bin_delete_final_image=settings_dict.get("bin_delete_final_image", "false"),
        bin_resize_final_image=settings_dict.get("bin_resize_final_image", "false"),
        bin_detection_reasoning=settings_dict.get("bin_detection_reasoning", "true"),
        # Performance (per-scan, dynamic)
        tile_concurrency=settings_dict.get("tile_concurrency", ""),
        tile_delay_s=settings_dict.get("tile_delay_s", ""),
        max_image_mb=settings_dict.get("max_image_mb", ""),
        duckdb_memory_limit=settings_dict.get("duckdb_memory_limit", ""),
        duckdb_threads=settings_dict.get("duckdb_threads", ""),
        # Infrastructure (require restart)
        worker_concurrency=settings_dict.get("worker_concurrency", ""),
        heavy_phase_concurrency=settings_dict.get("heavy_phase_concurrency", ""),
        browser_concurrency=settings_dict.get("browser_concurrency", ""),
        stale_scan_timeout_minutes=settings_dict.get("stale_scan_timeout_minutes", ""),
    )


@api_router.put("/settings", response_model=SettingsResponse)
async def update_settings(req: SettingsUpdate, db: AsyncSession = Depends(get_db)):
    """Update app settings (API key, models, prompts, etc.)."""
    updates = req.model_dump(exclude_none=True)

    for key, value in updates.items():
        result = await db.execute(select(Setting).where(Setting.key == key))
        setting = result.scalar_one_or_none()
        if setting:
            setting.value = value
        else:
            db.add(Setting(key=key, value=value))

    await db.commit()

    # Password change → invalidate all sessions so everyone re-authenticates
    if "app_password" in updates:
        from app.auth import clear_sessions
        await clear_sessions()

    return await get_settings(db)


# --- Storage diagnostics ---

def _dir_stats(path: str) -> tuple[float, int]:
    """Walk a directory and return (total_bytes, file_count)."""
    total = 0
    count = 0
    if not os.path.isdir(path):
        return total, count
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
                count += 1
            except OSError:
                pass
    return total, count


@api_router.get("/storage")
async def get_storage():
    """Return volume storage breakdown in MB (with debug)."""
    volume = app_settings.volume_path

    tile_bytes, tile_files = _dir_stats(os.path.join(volume, "tile_cache"))
    ss_bytes, ss_files = _dir_stats(os.path.join(volume, "screenshots"))

    # Total volume usage
    total_bytes, _ = _dir_stats(volume)
    other_bytes = total_bytes - tile_bytes - ss_bytes

    to_mb = lambda b: round(b / (1024 * 1024), 1)

    # --- Debug info (TEMPORARY - remove after investigating volume issue) ---
    import subprocess
    debug = {}
    debug["volume_path_setting"] = volume
    debug["volume_path_resolved"] = os.path.realpath(volume)
    debug["env_VOLUME_PATH"] = os.environ.get("VOLUME_PATH", "NOT SET")
    debug["env_RAILWAY_VOLUME_MOUNT_PATH"] = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "NOT SET")
    try:
        du = subprocess.run(["du", "-sh", volume], capture_output=True, text=True, timeout=30)
        debug["du_total"] = du.stdout.strip() if du.stdout else du.stderr.strip()
    except Exception as e:
        debug["du_total"] = str(e)
    # du per subdirectory
    try:
        du2 = subprocess.run(["du", "-sh"] + [
            os.path.join(volume, d) for d in os.listdir(volume) if os.path.isdir(os.path.join(volume, d))
        ], capture_output=True, text=True, timeout=30)
        debug["du_subdirs"] = du2.stdout.strip().split("\n") if du2.stdout else du2.stderr.strip()
    except Exception as e:
        debug["du_subdirs"] = str(e)
    # Count screenshot subdirectories and their sizes
    ss_path = os.path.join(volume, "screenshots")
    if os.path.isdir(ss_path):
        facility_dirs = []
        for d in sorted(os.listdir(ss_path)):
            dp = os.path.join(ss_path, d)
            if os.path.isdir(dp):
                files = os.listdir(dp)
                size = sum(os.path.getsize(os.path.join(dp, f)) for f in files if os.path.isfile(os.path.join(dp, f)))
                facility_dirs.append({"name": d, "files": len(files), "size_mb": round(size / 1048576, 1)})
        debug["screenshot_facilities"] = facility_dirs
        debug["screenshot_facility_count"] = len(facility_dirs)
    try:
        df = subprocess.run(["df", "-h"], capture_output=True, text=True, timeout=10)
        debug["df"] = df.stdout.strip().split("\n") if df.stdout else df.stderr.strip()
    except Exception as e:
        debug["df"] = str(e)
    # List top-level dirs in /data
    if os.path.isdir(volume):
        debug["ls_data"] = sorted(os.listdir(volume))
    # Check lost+found size
    lf = os.path.join(volume, "lost+found")
    if os.path.isdir(lf):
        try:
            lfdu = subprocess.run(["du", "-sh", lf], capture_output=True, text=True, timeout=10)
            debug["lost_found_size"] = lfdu.stdout.strip() if lfdu.stdout else "empty or error"
        except Exception:
            debug["lost_found_size"] = "error"

    return {
        "total_mb": to_mb(total_bytes),
        "tile_cache_mb": to_mb(tile_bytes),
        "tile_cache_files": tile_files,
        "screenshots_mb": to_mb(ss_bytes),
        "screenshots_files": ss_files,
        "other_mb": to_mb(other_bytes),
        "debug": debug,
    }


@api_router.post("/storage/cleanup")
async def cleanup_storage():
    """One-time cleanup: nuke tile_cache and delete orphaned screenshot files."""
    volume = app_settings.volume_path
    tile_cache_path = os.path.join(volume, "tile_cache")
    screenshots_path = os.path.join(volume, "screenshots")
    to_mb = lambda b: round(b / (1024 * 1024), 1)

    # --- Before stats ---
    tile_before_bytes, tile_before_files = _dir_stats(tile_cache_path)
    ss_before_bytes, ss_before_files = _dir_stats(screenshots_path)
    total_before_bytes, _ = _dir_stats(volume)

    # --- 1. Nuke entire tile_cache directory ---
    tile_deleted = tile_before_files
    shutil.rmtree(tile_cache_path, ignore_errors=True)

    # --- 2. Delete orphaned screenshot files ---
    async with async_session() as db:
        result = await db.execute(select(Screenshot.file_path))
        valid_paths = {row[0] for row in result.all() if row[0]}

    orphan_deleted = 0
    if os.path.isdir(screenshots_path):
        for dirpath, _, filenames in os.walk(screenshots_path, topdown=False):
            for fname in filenames:
                abs_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(abs_path, volume)
                if rel_path not in valid_paths:
                    try:
                        os.unlink(abs_path)
                        orphan_deleted += 1
                    except OSError:
                        pass
            # Remove empty directories
            try:
                os.rmdir(dirpath)
            except OSError:
                pass  # not empty or doesn't exist

    # --- After stats ---
    tile_after_bytes, _ = _dir_stats(tile_cache_path)
    ss_after_bytes, _ = _dir_stats(screenshots_path)
    total_after_bytes, _ = _dir_stats(volume)

    return {
        "before": {
            "tile_cache_mb": to_mb(tile_before_bytes),
            "screenshots_mb": to_mb(ss_before_bytes),
            "total_mb": to_mb(total_before_bytes),
        },
        "after": {
            "tile_cache_mb": to_mb(tile_after_bytes),
            "screenshots_mb": to_mb(ss_after_bytes),
            "total_mb": to_mb(total_after_bytes),
        },
        "deleted": {
            "tile_cache_files": tile_deleted,
            "screenshot_files": orphan_deleted,
        },
    }


# --- Debug: download volume as zip (TEMPORARY) ---

@api_router.get("/debug/download-volume")
async def download_volume():
    """Zip and download the entire /data volume. REMOVE after debugging."""
    import tempfile
    import zipfile
    from fastapi.responses import FileResponse

    volume = app_settings.volume_path
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp.close()

    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(volume):
            # Skip lost+found
            dirnames[:] = [d for d in dirnames if d != "lost+found"]
            for fname in filenames:
                abs_path = os.path.join(dirpath, fname)
                arc_name = os.path.relpath(abs_path, volume)
                try:
                    zf.write(abs_path, arc_name)
                except OSError:
                    pass

    return FileResponse(
        tmp.name,
        media_type="application/zip",
        filename="satellite-scanner-volume.zip",
    )


# --- Debug: volume inspection (TEMPORARY) ---

@api_router.get("/debug/volume")
async def debug_volume():
    """Temporary endpoint to inspect what's on the volume. REMOVE after debugging."""
    import subprocess
    volume = app_settings.volume_path
    result = {}

    # 1. What does volume_path resolve to?
    result["volume_path_setting"] = volume
    result["volume_path_resolved"] = os.path.realpath(volume)
    result["volume_path_exists"] = os.path.isdir(volume)

    # 2. List top-level contents of /data (the actual mount)
    for check_path in [volume, "/data"]:
        key = f"ls_{check_path.replace('/', '_').strip('_')}"
        if os.path.isdir(check_path):
            entries = []
            for entry in sorted(os.listdir(check_path)):
                full = os.path.join(check_path, entry)
                if os.path.isdir(full):
                    entries.append({"name": entry, "type": "dir"})
                else:
                    entries.append({"name": entry, "type": "file", "size_bytes": os.path.getsize(full)})
            result[key] = entries
        else:
            result[key] = "NOT FOUND"

    # 3. du -sh for each top-level dir under volume_path
    try:
        du = subprocess.run(
            ["du", "-sh", "--max-depth=1", volume],
            capture_output=True, text=True, timeout=30
        )
        result["du_volume_path"] = du.stdout.strip().split("\n") if du.stdout else du.stderr
    except Exception as e:
        result["du_volume_path"] = str(e)

    # 4. du -sh for /data specifically (in case it differs)
    if volume != "/data" and os.path.isdir("/data"):
        try:
            du = subprocess.run(
                ["du", "-sh", "--max-depth=1", "/data"],
                capture_output=True, text=True, timeout=30
            )
            result["du_data"] = du.stdout.strip().split("\n") if du.stdout else du.stderr
        except Exception as e:
            result["du_data"] = str(e)

    # 5. df for disk usage
    try:
        df = subprocess.run(
            ["df", "-h"],
            capture_output=True, text=True, timeout=10
        )
        result["df"] = df.stdout.strip().split("\n") if df.stdout else df.stderr
    except Exception as e:
        result["df"] = str(e)

    # 6. Check RAILWAY env vars
    result["env_VOLUME_PATH"] = os.environ.get("VOLUME_PATH", "NOT SET")
    result["env_RAILWAY_VOLUME_MOUNT_PATH"] = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "NOT SET")

    return result


# --- Railway restart ---

@api_router.post("/restart")
async def restart_service():
    """Trigger a redeploy of this Railway service instance."""
    token = os.environ.get("RAILWAY_API_TOKEN")
    service_id = os.environ.get("RAILWAY_SERVICE_ID")
    environment_id = os.environ.get("RAILWAY_ENVIRONMENT_ID")
    if not all([token, service_id, environment_id]):
        return JSONResponse({"error": "Railway env vars not configured"}, status_code=500)

    query = """
    mutation serviceInstanceRedeploy($serviceId: String!, $environmentId: String!) {
        serviceInstanceRedeploy(serviceId: $serviceId, environmentId: $environmentId)
    }
    """
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://backboard.railway.com/graphql/v2",
            headers={"Project-Access-Token": token, "Content-Type": "application/json"},
            json={"query": query, "variables": {"serviceId": service_id, "environmentId": environment_id}},
        )
    data = resp.json()
    if "errors" in data:
        logger.error("Railway redeploy failed: %s", data["errors"])
        return JSONResponse({"error": data["errors"][0].get("message", "Unknown error")}, status_code=502)
    return {"ok": True}


# --- Geocoder proxies (avoids CORS, sets proper User-Agent) ---

@api_router.get("/geocode/census")
async def geocode_census(address: str = Query(...)):
    """Proxy to US Census Geocoder to avoid browser CORS restrictions."""
    url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
    params = {"address": address, "benchmark": "Public_AR_Current", "format": "json"}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@api_router.get("/geocode/serper")
async def geocode_serper(
    name: str = Query(""),
    address: str = Query(""),
    db: AsyncSession = Depends(get_db),
):
    """Geocode via Serper Maps API with retry + fuzzy address matching."""
    key = await _get_setting(db, "serper_api_key")
    if not key:
        return JSONResponse(content={"results": []})

    result = await serper_maps_resolve(key, name, address)
    if result:
        return JSONResponse(content={"results": [result]})
    return JSONResponse(content={"results": [], "error": "Could not resolve address"})


@api_router.get("/geocode/google-search")
async def geocode_google_search(name: str = Query(""), address: str = Query("")):
    """Resolve facility coordinates via Google Maps headless browser.

    Opens a Google Maps search URL, waits for JS to resolve it to a place
    with coordinates, then parses the final URL. Returns empty results when
    Playwright is unavailable or Google can't resolve to a specific place.
    """
    query = " ".join(filter(None, [name.strip(), address.strip()]))
    if not query:
        return JSONResponse(content={"results": [], "error": "No query"})

    try:
        result = await _google_search_resolve(query)
        if result:
            return JSONResponse(content={"results": [result]})
        return JSONResponse(content={"results": []})
    except Exception as e:
        logger.warning("Google Maps geocode failed: %s", e)
        return JSONResponse(content={"results": [], "error": str(e)})


async def _google_search_resolve(query: str) -> dict | None:
    """Use persistent headless browser to resolve a Google Maps search to coordinates."""
    try:
        from playwright.async_api import async_playwright  # noqa: F401 — import check
    except ImportError:
        logger.warning("Playwright not installed — Google Maps geocoding unavailable")
        return None

    if _browser_semaphore is None:
        init_browser_semaphore()
    context = None
    async with _browser_semaphore:
        try:
            search_url = f"https://www.google.com/maps/search/{urllib.parse.quote(query)}"
            browser = await _get_browser()

            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                locale="en-US",
            )
            # Skip Google's cookie consent page
            await context.add_cookies([{
                "name": "SOCS",
                "value": "CAISNQgDEitib3FfaWRlbnRpdHlmcm9udGVuZHVpc2VydmVyXzIwMjMwODI5LjA3X3AxGgJlbiADGgYIgJa9pwY",
                "domain": ".google.com", "path": "/",
            }, {
                "name": "CONSENT",
                "value": "YES+cb.20231229-04-p0.en+FX+411",
                "domain": ".google.com", "path": "/",
            }])

            page = await context.new_page()
            await page.goto(search_url, wait_until="domcontentloaded", timeout=20000)

            # Wait for URL to contain @ (coordinates resolved by Google JS)
            try:
                await page.wait_for_url(re.compile(r"@-?\d+\.\d+,-?\d+\.\d+"), timeout=20000)
            except Exception:
                logger.info("Google Maps did not resolve coordinates for: %s", query)
                return None

            final_url = page.url
        except Exception as e:
            logger.warning("Google Maps geocode error for '%s': %s: %s", query, type(e).__name__, e)
            return None
        finally:
            if context:
                await context.close()

    # Parse result outside the semaphore (no browser needed)
    is_place = "/place/" in final_url

    # Extract zoom from viewport @ for precision check
    viewport_m = re.search(r"@(-?\d+\.\d+),(-?\d+\.\d+),(\d+(?:\.\d+)?)(z|m)", final_url)
    if not viewport_m:
        return None

    zoom_val = float(viewport_m.group(3))
    zoom_unit = viewport_m.group(4)

    # Prefer exact place coordinates from data section (!3d<lat>!4d<lng>)
    # The @ coordinates are the viewport center, which can drift ~200m from the pin
    place_m = re.search(r"!3d(-?\d+\.?\d*)!4d(-?\d+\.?\d*)", final_url)
    if place_m:
        lat = float(place_m.group(1))
        lng = float(place_m.group(2))
    else:
        lat = float(viewport_m.group(1))
        lng = float(viewport_m.group(2))

    # Determine if this is a precise match
    # z format: zoom level (17z = close, 13z = far) — 15+ is precise
    # m format: distance in meters (719m = close, 5745m = far) — <3000 is precise
    is_precise = (zoom_unit == "z" and zoom_val >= 15) or (zoom_unit == "m" and zoom_val < 3000)

    if is_place and is_precise:
        return {
            "lat": lat,
            "lng": lng,
            "display_name": f"Google Maps: {query}",
            "confidence": "high",
            "zoom": f"{zoom_val}{zoom_unit}",
            "source": "google_maps_search",
        }

    # Not precise enough — log and skip
    logger.info("Google Maps: %s match (%s%.0f%s) for: %s",
                "place" if is_place else "search", "" if is_place else "no /place/, ",
                zoom_val, zoom_unit, query)
    return None


# --- Health check (at root, not /api) ---

health_router = APIRouter()


@health_router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", queue_size=scan_queue.qsize(), workers=app_settings.worker_concurrency)
