"""Serve screenshot images from the volume (full-res + thumbnails)."""

import os
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from PIL import Image as PILImage

from app.config import settings
from app.database import async_session
from app.models import Screenshot

router = APIRouter()

THUMB_MAX_HEIGHT = 256


def _thumb_path(abs_path: str) -> str:
    """Return the thumbnail path for a given screenshot path."""
    root, ext = os.path.splitext(abs_path)
    return f"{root}_thumb{ext}"


def _ensure_thumbnail(abs_path: str) -> str:
    """Generate a thumbnail if it doesn't exist yet. Returns the thumb path."""
    thumb = _thumb_path(abs_path)
    if os.path.isfile(thumb):
        return thumb
    img = PILImage.open(abs_path)
    if img.height <= THUMB_MAX_HEIGHT:
        img.close()
        return abs_path
    ratio = THUMB_MAX_HEIGHT / img.height
    new_width = int(img.width * ratio)
    resized = img.resize((new_width, THUMB_MAX_HEIGHT), PILImage.LANCZOS)
    img.close()
    resized.save(thumb)
    resized.close()
    return thumb


@router.get("/screenshots/{screenshot_id}")
async def get_screenshot(screenshot_id: UUID, thumb: bool = Query(False)):
    """Serve a screenshot PNG from the volume. Pass ?thumb=1 for a 256px thumbnail."""
    async with async_session() as db:
        ss = await db.get(Screenshot, screenshot_id)
        if not ss:
            raise HTTPException(status_code=404, detail="Screenshot not found")

        abs_path = os.path.join(settings.volume_path, ss.file_path)
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail="Screenshot file not found on disk")

        if thumb:
            serve_path = _ensure_thumbnail(abs_path)
        else:
            serve_path = abs_path

        return FileResponse(
            serve_path,
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename=\"{ss.filename}\""},
        )
