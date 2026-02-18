"""FastAPI application with lifespan for background worker and DB init."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app.api.router import api_router, health_router, shutdown_browser
from app.config import settings
from app.database import engine
from app.models import Base, Setting
from app.worker import worker_pool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default settings to seed on first run
DEFAULT_SETTINGS = {
    "openai_api_key": settings.openai_api_key,
    "serper_api_key": settings.serper_api_key,
    "validation_model": "gpt-5-mini",
    "boundary_model": "gpt-5.2",
    "default_zoom": str(settings.default_zoom),
    "default_buffer_m": str(settings.default_buffer_m),
    "overview_zoom": str(settings.overview_zoom),
    "validation_prompt": "",
    "boundary_prompt": "",
    "verification_prompt": "",
}


async def _init_db():
    """Create tables, run migrations, and seed default settings."""
    # 1. Enum migration — MUST be outside a transaction (PG restriction)
    try:
        conn = await engine.connect()
        await conn.execution_options(isolation_level="AUTOCOMMIT")
        await conn.execute(text(
            "ALTER TYPE scan_status ADD VALUE IF NOT EXISTS 'pending' BEFORE 'queued'"
        ))
        await conn.close()
    except Exception as e:
        logger.info("Enum migration skipped (fresh DB or already done): %s", e)

    # 2. Merge facilities → scans migration (idempotent, guarded)
    async with engine.begin() as conn:
        await conn.execute(text("""
            DO $$
            BEGIN
              IF EXISTS (SELECT 1 FROM information_schema.tables
                         WHERE table_name = 'facilities' AND table_schema = 'public') THEN

                -- Add new columns to scans
                ALTER TABLE scans ADD COLUMN IF NOT EXISTS facility_name TEXT;
                ALTER TABLE scans ADD COLUMN IF NOT EXISTS facility_address TEXT;
                ALTER TABLE scans ADD COLUMN IF NOT EXISTS lat FLOAT;
                ALTER TABLE scans ADD COLUMN IF NOT EXISTS lng FLOAT;

                -- Backfill from facilities
                UPDATE scans SET
                  facility_name = f.name,
                  facility_address = f.address,
                  lat = f.lat,
                  lng = f.lng
                FROM facilities f WHERE scans.facility_id = f.id;

                -- Drop FK and make facility_id nullable BEFORE orphan insert
                ALTER TABLE scans DROP CONSTRAINT IF EXISTS scans_facility_id_fkey;
                ALTER TABLE scans ALTER COLUMN facility_id DROP NOT NULL;

                -- Recover orphaned facilities as pending scans
                INSERT INTO scans (id, facility_name, facility_address, lat, lng, status)
                SELECT id, name, address, lat, lng, 'pending'::scan_status
                FROM facilities f WHERE NOT EXISTS (SELECT 1 FROM scans s WHERE s.facility_id = f.id);

                -- Handle NULLs, then set NOT NULL
                UPDATE scans SET facility_name = 'Unknown' WHERE facility_name IS NULL;
                ALTER TABLE scans ALTER COLUMN facility_name SET NOT NULL;

                -- Drop column and table
                ALTER TABLE scans DROP COLUMN IF EXISTS facility_id;
                DROP TABLE IF EXISTS facilities;

                RAISE NOTICE 'facilities table merged into scans';
              END IF;
            END $$;
        """))

        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ready")

    # Seed default settings (don't overwrite existing)
    from app.database import async_session
    from sqlalchemy import select

    async with async_session() as db:
        for key, value in DEFAULT_SETTINGS.items():
            result = await db.execute(select(Setting).where(Setting.key == key))
            if result.scalar_one_or_none() is None:
                db.add(Setting(key=key, value=value))
        await db.commit()
    logger.info("Default settings seeded")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB, ensure volume dir, start worker. Shutdown: cancel worker."""
    # Ensure volume directory exists
    os.makedirs(settings.volume_path, exist_ok=True)
    os.makedirs(os.path.join(settings.volume_path, "screenshots"), exist_ok=True)

    await _init_db()

    # Start background worker
    worker_task = asyncio.create_task(worker_pool())
    logger.info("Background worker started")

    yield

    # Shutdown
    await shutdown_browser()
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    logger.info("Background worker stopped")


app = FastAPI(
    title="Satellite Scanner API",
    description="Automated satellite imagery capture for industrial facilities",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_router)
app.include_router(health_router)

# Mount static files (after routers so API routes take priority)
_static_dir = Path(__file__).parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    index = _static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return RedirectResponse(url="/docs")
