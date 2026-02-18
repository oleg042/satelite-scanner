"""API router — aggregates all endpoint modules."""

import asyncio
import logging
import re
import urllib.parse

import httpx
from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Setting
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
_browser_semaphore = asyncio.Semaphore(2)  # max 2 concurrent contexts (~150MB each)


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


# --- Settings endpoints (inline since they're small) ---

@api_router.get("/settings", response_model=SettingsResponse)
async def get_settings(db: AsyncSession = Depends(get_db)):
    """Get current app settings."""
    result = await db.execute(select(Setting))
    rows = result.scalars().all()
    settings_dict = {s.key: s.value for s in rows}

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
    return await get_settings(db)


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
