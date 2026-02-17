"""Serve screenshot images from the volume."""

import os
from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings
from app.database import async_session
from app.models import Screenshot

router = APIRouter()


@router.get("/screenshots/{screenshot_id}")
async def get_screenshot(screenshot_id: UUID):
    """Serve a screenshot PNG from the volume."""
    async with async_session() as db:
        ss = await db.get(Screenshot, screenshot_id)
        if not ss:
            raise HTTPException(status_code=404, detail="Screenshot not found")

        abs_path = os.path.join(settings.volume_path, ss.file_path)
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail="Screenshot file not found on disk")

        return FileResponse(
            abs_path,
            media_type="image/png",
            filename=ss.filename,
        )
