"""Shared helpers for saving images, recording screenshots, and recording pipeline steps."""

import os
import re

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ScanStep, Screenshot, ScreenshotType


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name).strip("_")


def save_image(image, name: str, suffix: str, zoom: int, scan_id, volume_path: str) -> tuple:
    """Save image to volume, return (relative_path, filename, abs_path)."""
    safe = _safe_name(name)
    rel_dir = os.path.join("screenshots", safe)
    abs_dir = os.path.join(volume_path, rel_dir)
    os.makedirs(abs_dir, exist_ok=True)
    id_tag = f"_{str(scan_id)[:8]}" if scan_id else ""
    filename = f"{safe}_{suffix}_z{zoom}{id_tag}.png"
    abs_path = os.path.join(abs_dir, filename)
    image.save(abs_path)
    rel_path = os.path.join(rel_dir, filename)
    return rel_path, filename, abs_path


async def record_screenshot(
    db: AsyncSession, scan_id, screenshot_type: ScreenshotType,
    filename: str, file_path: str, abs_path: str, zoom: int,
    width: int, height: int,
):
    """Create screenshot DB record."""
    file_size = os.path.getsize(abs_path) if os.path.exists(abs_path) else 0
    ss = Screenshot(
        scan_id=scan_id,
        type=screenshot_type,
        filename=filename,
        file_path=file_path,
        file_size_bytes=file_size,
        width=width,
        height=height,
        zoom=zoom,
    )
    db.add(ss)
    await db.commit()


async def record_step(db: AsyncSession, scan_id, step_number: int, step_name: str, **kwargs):
    """Create a pipeline step record."""
    step = ScanStep(
        scan_id=scan_id,
        step_number=step_number,
        step_name=step_name,
        **kwargs,
    )
    db.add(step)
    await db.commit()
    return step
