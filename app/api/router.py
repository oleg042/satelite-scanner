"""API router — aggregates all endpoint modules."""

import logging

import httpx
from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Setting
from app.schemas import HealthResponse, SettingsResponse, SettingsUpdate
from app.worker import scan_queue

from app.api.facilities import router as facilities_router
from app.api.scans import router as scans_router
from app.api.screenshots import router as screenshots_router

logger = logging.getLogger(__name__)

api_router = APIRouter(prefix="/api")

# Include sub-routers
api_router.include_router(scans_router)
api_router.include_router(facilities_router)
api_router.include_router(screenshots_router)


# --- Settings endpoints (inline since they're small) ---

@api_router.get("/settings", response_model=SettingsResponse)
async def get_settings(db: AsyncSession = Depends(get_db)):
    """Get current app settings."""
    result = await db.execute(select(Setting))
    rows = result.scalars().all()
    settings_dict = {s.key: s.value for s in rows}

    # Mask API key for security
    api_key = settings_dict.get("openai_api_key", "")
    if api_key and len(api_key) > 8:
        api_key = api_key[:4] + "..." + api_key[-4:]

    return SettingsResponse(
        openai_api_key=api_key,
        validation_model=settings_dict.get("validation_model", ""),
        boundary_model=settings_dict.get("boundary_model", ""),
        default_zoom=settings_dict.get("default_zoom", ""),
        default_buffer_m=settings_dict.get("default_buffer_m", ""),
        overview_zoom=settings_dict.get("overview_zoom", ""),
        validation_prompt=settings_dict.get("validation_prompt", ""),
        boundary_prompt=settings_dict.get("boundary_prompt", ""),
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

@api_router.get("/geocode/nominatim")
async def geocode_nominatim(address: str = Query(...)):
    """Proxy to Nominatim to avoid CORS preflight and set User-Agent server-side."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": "3"}
    headers = {"User-Agent": "SatelliteScanner/1.0"}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params, headers=headers)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@api_router.get("/geocode/census")
async def geocode_census(address: str = Query(...)):
    """Proxy to US Census Geocoder to avoid browser CORS restrictions."""
    url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
    params = {"address": address, "benchmark": "Public_AR_Current", "format": "json"}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


# --- Health check (at root, not /api) ---

health_router = APIRouter()


@health_router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", queue_size=scan_queue.qsize())
