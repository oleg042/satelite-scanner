"""Waterfall pipeline orchestrator with full step-by-step tracing.

Flow:
  1. OSM CHECK → find building footprint
  2. AI VALIDATION → verify OSM bbox is complete
  3. AI VISION FALLBACK → detect boundary from scratch
  4. TILING → capture high-res imagery
  5. DONE
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Scan, ScanMethod, ScanStatus, ScanStep, Screenshot, ScreenshotType, Setting
from app.scanner.cache import TileCache
from app.scanner.geo import (
    M_PER_DEG_LAT,
    bbox_add_buffer,
    bbox_dimensions_m,
    meters_per_deg_lng,
    pixel_to_latlng,
)
from app.scanner.osm import (
    find_target_building,
    find_target_landuse,
    query_osm_buildings,
    query_osm_landuse,
)
from PIL import Image as PILImage

from app.scanner.tiles import capture_area, crop_to_bbox
from app.scanner.vision import detect_facility_boundary, validate_osm_bbox, verify_facility_boundary

logger = logging.getLogger(__name__)


async def _get_setting(db: AsyncSession, key: str, default: str = "") -> str:
    """Get a setting value from the database."""
    result = await db.execute(select(Setting.value).where(Setting.key == key))
    row = result.scalar_one_or_none()
    return row if row is not None else default


async def _get_scan_config(db: AsyncSession, scan: Scan) -> dict:
    """Resolve scan config: per-scan overrides > DB settings > env defaults."""
    api_key = await _get_setting(db, "openai_api_key", settings.openai_api_key)
    validation_model = await _get_setting(db, "validation_model", "gpt-4o-mini")
    boundary_model = await _get_setting(db, "boundary_model", "gpt-4o")
    default_zoom = int(await _get_setting(db, "default_zoom", str(settings.default_zoom)))
    default_buffer = int(await _get_setting(db, "default_buffer_m", str(settings.default_buffer_m)))
    overview_zoom = int(await _get_setting(db, "overview_zoom", str(settings.overview_zoom)))
    validation_prompt = await _get_setting(db, "validation_prompt", "")
    boundary_prompt = await _get_setting(db, "boundary_prompt", "")
    verification_prompt = await _get_setting(db, "verification_prompt", "")

    return {
        "api_key": api_key,
        "validation_model": validation_model,
        "boundary_model": boundary_model,
        "zoom": scan.zoom or default_zoom,
        "buffer_m": scan.buffer_m or default_buffer,
        "overview_zoom": overview_zoom,
        "validation_prompt": validation_prompt,
        "boundary_prompt": boundary_prompt,
        "verification_prompt": verification_prompt,
    }


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name).strip("_")


def _save_image(image, name: str, suffix: str, zoom: int, scan_id=None) -> str:
    """Save image to volume, return relative path."""
    safe = _safe_name(name)
    rel_dir = os.path.join("screenshots", safe)
    abs_dir = os.path.join(settings.volume_path, rel_dir)
    os.makedirs(abs_dir, exist_ok=True)

    id_tag = f"_{str(scan_id)[:8]}" if scan_id else ""
    filename = f"{safe}_{suffix}_z{zoom}{id_tag}.png"
    abs_path = os.path.join(abs_dir, filename)
    image.save(abs_path)

    rel_path = os.path.join(rel_dir, filename)
    return rel_path, filename, abs_path


async def _record_screenshot(
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


async def _record_step(db: AsyncSession, scan_id, step_number: int, step_name: str, **kwargs):
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


async def _downscale_overview(
    db: AsyncSession, scan_id, screenshot_type: ScreenshotType,
    abs_path: str, max_width: int = 1080,
):
    """Downscale an overview screenshot to save storage. Overwrites the file in place."""
    img = PILImage.open(abs_path)
    if img.width <= max_width:
        img.close()
        return
    ratio = max_width / img.width
    new_height = int(img.height * ratio)
    resized = img.resize((max_width, new_height), PILImage.LANCZOS)
    img.close()
    resized.save(abs_path)
    resized.close()

    new_size = os.path.getsize(abs_path)
    result = await db.execute(
        select(Screenshot).where(
            Screenshot.scan_id == scan_id,
            Screenshot.type == screenshot_type,
        )
    )
    ss = result.scalar_one_or_none()
    if ss:
        ss.width = max_width
        ss.height = new_height
        ss.file_size_bytes = new_size
        await db.commit()

    logger.info("Downscaled %s screenshot to %dx%d (%dKB)", screenshot_type.value, max_width, new_height, new_size // 1024)


async def run_pipeline(scan_id, db: AsyncSession):
    """Execute the full waterfall pipeline for a single scan."""
    scan = await db.get(Scan, scan_id)
    if not scan:
        logger.error("Scan %s not found", scan_id)
        return

    # Read facility info from the scan row
    facility_name = scan.facility_name
    lat = scan.lat
    lng = scan.lng

    if lat is None or lng is None:
        logger.warning("Skipping scan %s — no coordinates", scan_id)
        scan.status = ScanStatus.failed
        scan.error_message = "No coordinates available"
        scan.completed_at = datetime.now(timezone.utc)
        await db.commit()
        return

    # Clear any old steps/screenshots from a previous run of this scan
    await db.execute(delete(ScanStep).where(ScanStep.scan_id == scan_id))
    await db.execute(delete(Screenshot).where(Screenshot.scan_id == scan_id))
    scan.error_message = None
    scan.started_at = datetime.now(timezone.utc)
    await db.commit()

    config = await _get_scan_config(db, scan)
    cache = TileCache()
    zoom = config["zoom"]
    buffer_m = config["buffer_m"]
    overview_zoom = config["overview_zoom"]

    # Will be set by OSM or AI vision
    final_bbox = None
    method = None
    step_num = 0

    try:
        # ── STEP 1: OSM CHECK ──────────────────────────────────────
        step_num += 1
        scan.status = ScanStatus.running_osm
        await db.commit()

        osm_start = time.monotonic()
        buildings = await query_osm_buildings(lat, lng, settings.search_radius_m)
        scan.osm_building_count = len(buildings)

        osm_bbox = None
        osm_decision = None
        osm_building_info = None

        if buildings:
            building = find_target_building(buildings, lat, lng)
            if building:
                scan.osm_building_id = building["id"]
                lats = [c[0] for c in building["coords"]]
                lngs = [c[1] for c in building["coords"]]
                raw_bbox = (min(lats), min(lngs), max(lats), max(lngs))
                osm_bbox = bbox_add_buffer(*raw_bbox, buffer_m)
                method = ScanMethod.osm_building
                osm_building_info = {
                    "osm_id": building["id"],
                    "raw_bbox": list(raw_bbox),
                    "buffered_bbox": list(osm_bbox),
                    "coord_count": len(building["coords"]),
                }
                osm_decision = f"Found target building OSM #{building['id']} among {len(buildings)} buildings. Applied {buffer_m}m buffer."
            else:
                osm_decision = f"Found {len(buildings)} buildings but none matched target coordinates."
        else:
            osm_decision = "No buildings found within search radius."

        osm_elapsed = int((time.monotonic() - osm_start) * 1000)
        scan.osm_duration_ms = osm_elapsed
        await db.commit()

        await _record_step(
            db, scan_id, step_num, "osm_check",
            status="completed",
            started_at=scan.started_at,
            completed_at=datetime.now(timezone.utc),
            duration_ms=osm_elapsed,
            input_summary=json.dumps({
                "facility_name": facility_name,
                "lat": lat,
                "lng": lng,
                "search_radius_m": settings.search_radius_m,
            }),
            output_summary=json.dumps({
                "buildings_found": len(buildings),
                "target_building": osm_building_info,
                "bbox": list(osm_bbox) if osm_bbox else None,
            }),
            decision=osm_decision,
        )

        # ── STEP 2: AI VALIDATION (if OSM found something) ────────
        if osm_bbox and config["api_key"]:
            step_num += 1
            scan.status = ScanStatus.running_validate
            await db.commit()

            ai_start = time.monotonic()
            step_started = datetime.now(timezone.utc)

            # Capture overview at OSM bbox for validation
            ov_img, ov_grid = await asyncio.to_thread(
                capture_area,
                *osm_bbox, overview_zoom, cache, settings.tile_delay_s,
            )

            if ov_img:
                rel_path, filename, abs_path = _save_image(
                    ov_img, facility_name, "overview", overview_zoom, scan_id
                )
                await _record_screenshot(
                    db, scan_id, ScreenshotType.overview,
                    filename, rel_path, abs_path, overview_zoom,
                    ov_img.width, ov_img.height,
                )

                width_m, height_m = bbox_dimensions_m(*osm_bbox)

                validation = None
                validation_error = None
                try:
                    validation = await asyncio.to_thread(
                        validate_osm_bbox,
                        abs_path, facility_name, lat, lng, width_m, height_m,
                        config["api_key"], config["validation_model"],
                        config["validation_prompt"],
                    )
                except Exception as e:
                    validation_error = str(e)
                    logger.warning("Validation call failed: %s", e)

                ai_elapsed = int((time.monotonic() - ai_start) * 1000)
                val_decision = None

                if validation:
                    scan.ai_validated = validation.approved
                    scan.ai_facility_type = validation.facility_type
                    scan.ai_notes = validation.notes
                    if validation.approved:
                        final_bbox = osm_bbox
                        val_decision = f"APPROVED — AI confirmed OSM bbox covers the facility. Type: {validation.facility_type}"
                        logger.info("AI approved OSM bbox")
                    else:
                        val_decision = f"REJECTED — {validation.reason}. Falling through to AI vision."
                        logger.info("AI rejected OSM bbox: %s", validation.reason)
                        method = None
                else:
                    final_bbox = osm_bbox
                    val_decision = f"AI validation failed: {validation_error or 'unknown error'}. Using OSM bbox as fallback."
                    logger.warning("Validation call failed, using OSM bbox anyway")

                await _record_step(
                    db, scan_id, step_num, "ai_validation",
                    status="completed",
                    started_at=step_started,
                    completed_at=datetime.now(timezone.utc),
                    duration_ms=ai_elapsed,
                    input_summary=json.dumps({
                        "bbox": list(osm_bbox),
                        "bbox_width_m": round(width_m, 1),
                        "bbox_height_m": round(height_m, 1),
                        "overview_zoom": overview_zoom,
                        "overview_image": f"{ov_img.width}x{ov_img.height}px",
                        "model": config["validation_model"],
                    }),
                    output_summary=json.dumps({
                        "approved": validation.approved if validation else None,
                        "facility_type": validation.facility_type if validation else None,
                        "reason": validation.reason if validation else None,
                        "notes": validation.notes if validation else None,
                        "error": validation_error,
                    }),
                    decision=val_decision,
                    ai_model=config["validation_model"],
                    ai_prompt=validation.prompt_text if validation else None,
                    ai_response_raw=validation.raw_response if validation else None,
                    ai_tokens_prompt=validation.usage.prompt_tokens if validation and validation.usage else None,
                    ai_tokens_completion=validation.usage.completion_tokens if validation and validation.usage else None,
                    ai_tokens_reasoning=validation.usage.reasoning_tokens if validation and validation.usage else None,
                    ai_tokens_total=validation.usage.total_tokens if validation and validation.usage else None,
                )

                await _downscale_overview(db, scan_id, ScreenshotType.overview, abs_path)
                ov_img.close()

            scan.ai_duration_ms = int((time.monotonic() - ai_start) * 1000)
            await db.commit()
        elif osm_bbox:
            # No API key — just use OSM bbox
            step_num += 1
            final_bbox = osm_bbox
            await _record_step(
                db, scan_id, step_num, "ai_validation",
                status="skipped",
                decision="No OpenAI API key configured. Using OSM bbox without AI validation.",
                duration_ms=0,
            )

        # ── STEP 3: AI VISION FALLBACK ─────────────────────────────
        if final_bbox is None and config["api_key"]:
            step_num += 1
            scan.status = ScanStatus.running_vision
            await db.commit()

            ai_start = time.monotonic()
            step_started = datetime.now(timezone.utc)

            # Capture wide overview
            wide_bbox = bbox_add_buffer(lat, lng, lat, lng, settings.overview_radius_m)
            ov_img, ov_grid = await asyncio.to_thread(
                capture_area,
                *wide_bbox, overview_zoom, cache, settings.tile_delay_s,
            )

            if ov_img and ov_grid:
                rel_path, filename, abs_path = _save_image(
                    ov_img, facility_name, "ai_overview", overview_zoom, scan_id
                )
                await _record_screenshot(
                    db, scan_id, ScreenshotType.ai_overview,
                    filename, rel_path, abs_path, overview_zoom,
                    ov_img.width, ov_img.height,
                )

                width_m, height_m = bbox_dimensions_m(*wide_bbox)

                boundary = None
                boundary_error = None
                try:
                    boundary = await asyncio.to_thread(
                        detect_facility_boundary,
                        abs_path, facility_name, lat, lng, width_m, height_m,
                        config["api_key"], config["boundary_model"],
                        config["boundary_prompt"],
                    )
                except Exception as e:
                    boundary_error = str(e)
                    logger.warning("Boundary detection failed: %s", e)

                vision_decision = None
                verification = None
                verification_error = None
                original_boundary = None

                if boundary:
                    scan.ai_confidence = boundary.confidence
                    scan.ai_facility_type = boundary.facility_type
                    scan.ai_building_count = boundary.building_count
                    scan.ai_notes = boundary.notes

                    NON_INDUSTRIAL_TYPES = {
                        "office", "residential", "apartment", "house", "housing",
                        "neighborhood", "school", "church", "park", "farm",
                        "agricultural", "retail", "restaurant", "hotel", "motel",
                        "cemetery", "golf", "parking lot", "vacant",
                    }

                    ft_lower = (boundary.facility_type or "").lower()
                    is_non_industrial = any(kw in ft_lower for kw in NON_INDUSTRIAL_TYPES)

                    if boundary.confidence == "low" and is_non_industrial:
                        ai_elapsed = int((time.monotonic() - ai_start) * 1000)
                        vision_decision = f"SKIPPED — AI detected non-industrial facility: {boundary.facility_type}"
                        scan.status = ScanStatus.skipped
                        scan.method = ScanMethod.skipped
                        scan.skip_reason = f"AI: {boundary.facility_type} — {boundary.notes}"
                        scan.completed_at = datetime.now(timezone.utc)

                        await _record_step(
                            db, scan_id, step_num, "ai_vision",
                            status="completed",
                            started_at=step_started,
                            completed_at=datetime.now(timezone.utc),
                            duration_ms=ai_elapsed,
                            input_summary=json.dumps({
                                "wide_bbox": list(wide_bbox),
                                "overview_radius_m": settings.overview_radius_m,
                                "overview_image": f"{ov_img.width}x{ov_img.height}px",
                                "model": config["boundary_model"],
                            }),
                            output_summary=json.dumps({
                                "confidence": boundary.confidence,
                                "facility_type": boundary.facility_type,
                                "building_count": boundary.building_count,
                                "notes": boundary.notes,
                                "pixel_bbox": {"top_y": boundary.top_y, "bottom_y": boundary.bottom_y,
                                               "left_x": boundary.left_x, "right_x": boundary.right_x},
                            }),
                            decision=vision_decision,
                            ai_model=config["boundary_model"],
                            ai_prompt=boundary.prompt_text if boundary else None,
                            ai_response_raw=boundary.raw_response if boundary else None,
                            ai_tokens_prompt=boundary.usage.prompt_tokens if boundary.usage else None,
                            ai_tokens_completion=boundary.usage.completion_tokens if boundary.usage else None,
                            ai_tokens_reasoning=boundary.usage.reasoning_tokens if boundary.usage else None,
                            ai_tokens_total=boundary.usage.total_tokens if boundary.usage else None,
                        )

                        await db.commit()
                        await _downscale_overview(db, scan_id, ScreenshotType.ai_overview, abs_path)
                        ov_img.close()
                        return

                    # Verify boundary with a second AI call
                    try:
                        verification = await asyncio.to_thread(
                            verify_facility_boundary,
                            abs_path, facility_name, boundary,
                            config["api_key"], config["validation_model"],
                            config["verification_prompt"],
                        )
                    except Exception as e:
                        verification_error = str(e)
                        logger.warning("Verification call failed: %s", e)

                    if verification and not verification.approved:
                        logger.info("Boundary rejected: %s. Retrying with feedback...", verification.reason)

                        # Preserve original detection before retry overwrites it
                        original_boundary = {
                            "pixel_bbox": {"top_y": boundary.top_y, "bottom_y": boundary.bottom_y,
                                           "left_x": boundary.left_x, "right_x": boundary.right_x},
                            "confidence": boundary.confidence,
                            "facility_type": boundary.facility_type,
                            "building_count": boundary.building_count,
                            "notes": boundary.notes,
                            "raw_response": boundary.raw_response,
                            "prompt_text": boundary.prompt_text,
                            "tokens": {
                                "prompt": boundary.usage.prompt_tokens,
                                "completion": boundary.usage.completion_tokens,
                                "total": boundary.usage.total_tokens,
                            } if boundary.usage else None,
                        }

                        # Retry boundary detection with rejection feedback
                        retry_extra = (
                            f"\n\n## IMPORTANT CORRECTION\n"
                            f"A previous attempt was rejected: {verification.reason}\n"
                            f"Issues found: {verification.issues}\n"
                            f"Please correct these issues in your new boundary detection."
                        )
                        retry_prompt = (config["boundary_prompt"] or "") + retry_extra
                        try:
                            boundary = await asyncio.to_thread(
                                detect_facility_boundary,
                                abs_path, facility_name, lat, lng, width_m, height_m,
                                config["api_key"], config["boundary_model"],
                                retry_prompt,
                            )
                        except Exception as e:
                            boundary = None
                            logger.warning("Boundary retry failed: %s", e)

                        if boundary:
                            scan.ai_confidence = boundary.confidence
                            scan.ai_facility_type = boundary.facility_type
                            scan.ai_building_count = boundary.building_count
                            scan.ai_notes = boundary.notes

                    # Convert pixel bbox to lat/lng
                    if boundary:
                        lat1, lng1 = pixel_to_latlng(boundary.left_x, boundary.top_y, ov_grid)
                        lat2, lng2 = pixel_to_latlng(boundary.right_x, boundary.bottom_y, ov_grid)
                        raw_bbox = (min(lat1, lat2), min(lng1, lng2), max(lat1, lat2), max(lng1, lng2))
                        final_bbox = bbox_add_buffer(*raw_bbox, buffer_m)
                        method = ScanMethod.ai_vision
                        verified_str = "verified" if (not verification or verification.approved) else "retry after rejection"
                        vision_decision = (
                            f"AI detected facility boundary ({verified_str}) — confidence: {boundary.confidence}, "
                            f"type: {boundary.facility_type}, {boundary.building_count} buildings. "
                            f"Pixel bbox: ({boundary.left_x},{boundary.top_y})-({boundary.right_x},{boundary.bottom_y}) "
                            f"→ raw geo: {[round(x, 6) for x in raw_bbox]} "
                            f"→ buffered: {[round(x, 6) for x in final_bbox]}"
                        )
                    else:
                        vision_decision = f"AI boundary detection failed after retry. Will fall back to radius."
                else:
                    vision_decision = f"AI boundary detection failed: {boundary_error or 'unknown error'}. Will fall back to radius."

                ai_elapsed = int((time.monotonic() - ai_start) * 1000)

                await _record_step(
                    db, scan_id, step_num, "ai_vision",
                    status="completed",
                    started_at=step_started,
                    completed_at=datetime.now(timezone.utc),
                    duration_ms=ai_elapsed,
                    input_summary=json.dumps({
                        "wide_bbox": list(wide_bbox),
                        "overview_radius_m": settings.overview_radius_m,
                        "overview_image": f"{ov_img.width}x{ov_img.height}px" if ov_img else None,
                        "model": config["boundary_model"],
                    }),
                    output_summary=json.dumps({
                        "confidence": boundary.confidence if boundary else None,
                        "facility_type": boundary.facility_type if boundary else None,
                        "building_count": boundary.building_count if boundary else None,
                        "notes": boundary.notes if boundary else None,
                        "pixel_bbox": {"top_y": boundary.top_y, "bottom_y": boundary.bottom_y,
                                       "left_x": boundary.left_x, "right_x": boundary.right_x} if boundary else None,
                        "raw_geo_bbox": list(raw_bbox) if final_bbox and method == ScanMethod.ai_vision else None,
                        "detected_bbox": list(final_bbox) if final_bbox and method == ScanMethod.ai_vision else None,
                        "boundary_error": boundary_error,
                        "verification": {
                            "model": config["validation_model"],
                            "approved": verification.approved,
                            "reason": verification.reason,
                            "issues": verification.issues,
                            "prompt": verification.prompt_text,
                            "raw_response": verification.raw_response,
                            "tokens": {
                                "prompt": verification.usage.prompt_tokens,
                                "completion": verification.usage.completion_tokens,
                                "total": verification.usage.total_tokens,
                            } if verification.usage else None,
                        } if verification else {"error": verification_error} if verification_error else None,
                        "original_detection": original_boundary,
                    }),
                    decision=vision_decision,
                    ai_model=config["boundary_model"],
                    ai_prompt=boundary.prompt_text if boundary else None,
                    ai_response_raw=boundary.raw_response if boundary else None,
                    ai_tokens_prompt=boundary.usage.prompt_tokens if boundary and boundary.usage else None,
                    ai_tokens_completion=boundary.usage.completion_tokens if boundary and boundary.usage else None,
                    ai_tokens_reasoning=boundary.usage.reasoning_tokens if boundary and boundary.usage else None,
                    ai_tokens_total=boundary.usage.total_tokens if boundary and boundary.usage else None,
                )

                await _downscale_overview(db, scan_id, ScreenshotType.ai_overview, abs_path)
                ov_img.close()

            ai_elapsed = int((time.monotonic() - ai_start) * 1000)
            scan.ai_duration_ms = (scan.ai_duration_ms or 0) + ai_elapsed
            await db.commit()

        # ── FALLBACK: radius around point ──────────────────────────
        if final_bbox is None:
            step_num += 1
            final_bbox = bbox_add_buffer(lat, lng, lat, lng, settings.fallback_radius_m)
            method = ScanMethod.fallback_radius
            logger.info("Using fallback radius of %dm", settings.fallback_radius_m)

            await _record_step(
                db, scan_id, step_num, "fallback",
                status="completed",
                duration_ms=0,
                input_summary=json.dumps({
                    "lat": lat,
                    "lng": lng,
                    "fallback_radius_m": settings.fallback_radius_m,
                }),
                output_summary=json.dumps({
                    "bbox": list(final_bbox),
                }),
                decision=f"No boundary found by OSM or AI. Using {settings.fallback_radius_m}m radius fallback.",
            )

        # Record bbox
        scan.method = method
        scan.bbox_min_lat, scan.bbox_min_lng = final_bbox[0], final_bbox[1]
        scan.bbox_max_lat, scan.bbox_max_lng = final_bbox[2], final_bbox[3]
        width_m, height_m = bbox_dimensions_m(*final_bbox)
        scan.bbox_width_m = width_m
        scan.bbox_height_m = height_m
        await db.commit()

        # ── STEP 4: TILING ─────────────────────────────────────────
        step_num += 1
        scan.status = ScanStatus.running_tiling
        await db.commit()

        tile_start = time.monotonic()
        step_started = datetime.now(timezone.utc)

        detail_img, grid_info = await asyncio.to_thread(
            capture_area,
            *final_bbox, zoom, cache, settings.tile_delay_s,
        )

        tile_decision = None
        grid_cols = None
        grid_rows = None
        if detail_img and grid_info:
            scan.tile_count = grid_info["total"]
            scan.tiles_downloaded = grid_info["downloaded"]

            # Crop to exact bounds
            cropped = crop_to_bbox(detail_img, *final_bbox, grid_info)
            detail_img.close()

            rel_path, filename, abs_path = _save_image(
                cropped, facility_name, "final", zoom, scan_id
            )
            scan.image_width = cropped.width
            scan.image_height = cropped.height

            await _record_screenshot(
                db, scan_id, ScreenshotType.final,
                filename, rel_path, abs_path, zoom,
                cropped.width, cropped.height,
            )

            file_size = os.path.getsize(abs_path) if os.path.exists(abs_path) else 0
            grid_cols = grid_info["tile_max_x"] - grid_info["tile_min_x"] + 1
            grid_rows = grid_info["tile_max_y"] - grid_info["tile_min_y"] + 1
            tile_decision = (
                f"Captured {grid_info['total']} tiles ({grid_cols}x{grid_rows} grid) "
                f"at zoom {zoom}. Downloaded {grid_info['downloaded']} from server, "
                f"{grid_info['total'] - grid_info['downloaded']} from cache. "
                f"Final image: {cropped.width}x{cropped.height}px ({file_size // 1024}KB)"
            )
            cropped.close()
        else:
            tile_decision = "Tile capture failed — no image produced."

        tile_elapsed = int((time.monotonic() - tile_start) * 1000)
        scan.tile_duration_ms = tile_elapsed

        await _record_step(
            db, scan_id, step_num, "tiling",
            status="completed",
            started_at=step_started,
            completed_at=datetime.now(timezone.utc),
            duration_ms=tile_elapsed,
            input_summary=json.dumps({
                "bbox": list(final_bbox),
                "bbox_width_m": round(width_m, 1),
                "bbox_height_m": round(height_m, 1),
                "zoom": zoom,
                "tile_delay_s": settings.tile_delay_s,
            }),
            output_summary=json.dumps({
                "tile_grid": f"{grid_cols}x{grid_rows}" if grid_info else None,
                "total_tiles": grid_info["total"] if grid_info else None,
                "tiles_downloaded": grid_info["downloaded"] if grid_info else None,
                "tiles_from_cache": grid_info["total"] - grid_info["downloaded"] if grid_info else None,
                "final_image_px": f"{scan.image_width}x{scan.image_height}" if scan.image_width else None,
            }),
            decision=tile_decision,
            tile_grid_cols=grid_cols if grid_info else None,
            tile_grid_rows=grid_rows if grid_info else None,
        )

        # ── STEP 5: DONE ──────────────────────────────────────────
        step_num += 1
        scan.status = ScanStatus.completed
        scan.completed_at = datetime.now(timezone.utc)

        total_ms = int((scan.completed_at - scan.started_at).total_seconds() * 1000)

        await _record_step(
            db, scan_id, step_num, "done",
            status="completed",
            completed_at=scan.completed_at,
            duration_ms=0,
            decision=(
                f"Pipeline complete — method: {method.value if method else 'unknown'}, "
                f"total time: {total_ms}ms, "
                f"bbox: {round(width_m, 1)}m x {round(height_m, 1)}m, "
                f"final image: {scan.image_width}x{scan.image_height}px"
            ),
        )

        await db.commit()
        logger.info("Scan %s completed — method=%s", scan_id, method)

    except Exception as e:
        logger.exception("Pipeline failed for scan %s", scan_id)
        scan.status = ScanStatus.failed
        scan.error_message = str(e)[:1000]
        scan.completed_at = datetime.now(timezone.utc)

        step_num += 1
        await _record_step(
            db, scan_id, step_num, "error",
            status="failed",
            completed_at=datetime.now(timezone.utc),
            decision=f"Pipeline failed: {str(e)[:500]}",
        )

        await db.commit()
