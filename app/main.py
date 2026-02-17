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

from app.api.router import api_router, health_router
from app.config import settings
from app.database import engine
from app.models import Base, Setting
from app.worker import worker_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default settings to seed on first run
DEFAULT_SETTINGS = {
    "openai_api_key": settings.openai_api_key,
    "validation_model": "gpt-4o-mini",
    "boundary_model": "gpt-4o",
    "default_zoom": str(settings.default_zoom),
    "default_buffer_m": str(settings.default_buffer_m),
    "overview_zoom": str(settings.overview_zoom),
    "validation_prompt": "",
    "boundary_prompt": "",
}


async def _init_db():
    """Create tables and seed default settings."""
    async with engine.begin() as conn:
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
    worker_task = asyncio.create_task(worker_loop())
    logger.info("Background worker started")

    yield

    # Shutdown
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
