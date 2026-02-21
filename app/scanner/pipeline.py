"""Waterfall pipeline orchestrator with full step-by-step tracing.

Flow:
  1. OSM CHECK → query OpenStreetMap (fast, server-side spatial index)
  2. MSFT/OVERTURE FALLBACK → query Microsoft/Overture if OSM missed
  3. AI VALIDATION → verify bbox is complete
  4. AI VISION FALLBACK → detect boundary from scratch
  5. TILING → capture high-res imagery
  6. DONE
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
from app.scanner.geocode import serper_maps_resolve
from app.scanner.geo import (
    M_PER_DEG_LAT,
    bbox_add_buffer,
    bbox_dimensions_m,
    latlng_to_pixel,
    meters_per_deg_lng,
    pixel_to_latlng,
)
from app.scanner.msft import (
    find_target_building as find_target_building_msft,
    query_msft_buildings,
)
from app.scanner.overture import query_overture_buildings
from app.scanner.osm import (
    find_target_building,
    find_target_landuse,
    query_osm_buildings,
    query_osm_landuse,
)
from PIL import Image as PILImage

from app.scanner.tiles import capture_area, crop_to_bbox
from app.scanner.vision import correct_facility_boundary, detect_facility_boundary, validate_osm_bbox, verify_and_correct_boundary, verify_facility_boundary

logger = logging.getLogger(__name__)

_heavy_semaphore = asyncio.Semaphore(settings.heavy_phase_concurrency)


def _is_fatal_ai_error(exc: Exception) -> bool:
    """Check if an OpenAI error is fatal (billing/auth) vs transient (network/rate-limit).

    Fatal errors should abort the scan rather than silently falling back.
    """
    try:
        import openai
        if isinstance(exc, openai.AuthenticationError):
            return True
        if isinstance(exc, openai.RateLimitError):
            # insufficient_quota = billing issue (fatal), rate_limit_exceeded = transient
            code = getattr(exc, "code", None) or ""
            return "insufficient_quota" in str(code)
        if isinstance(exc, openai.PermissionDeniedError):
            return True
    except ImportError:
        pass
    return False


async def _get_setting(db: AsyncSession, key: str, default: str = "") -> str:
    """Get a setting value from the database."""
    result = await db.execute(select(Setting.value).where(Setting.key == key))
    row = result.scalar_one_or_none()
    return row if row is not None else default


async def _get_scan_config(db: AsyncSession, scan: Scan) -> dict:
    """Resolve scan config: per-scan overrides > DB settings > env defaults."""
    api_key = await _get_setting(db, "openai_api_key", settings.openai_api_key)
    validation_model = await _get_setting(db, "validation_model", "gpt-5-mini")
    boundary_model = await _get_setting(db, "boundary_model", "gpt-5.2")
    default_zoom = int(await _get_setting(db, "default_zoom", str(settings.default_zoom)))
    default_buffer = int(await _get_setting(db, "default_buffer_m", str(settings.default_buffer_m)))
    overview_zoom = int(await _get_setting(db, "overview_zoom", str(settings.overview_zoom)))
    validation_prompt = await _get_setting(db, "validation_prompt", "")
    boundary_prompt = await _get_setting(db, "boundary_prompt", "")
    verification_prompt = await _get_setting(db, "verification_prompt", "")
    verification_model = await _get_setting(db, "verification_model", "gpt-5.2")
    correction_mode = await _get_setting(db, "correction_mode", "v1")
    verification_correction_prompt = await _get_setting(db, "verification_correction_prompt", "")
    bbox_validation_enabled = (await _get_setting(db, "bbox_validation_enabled", "true")).lower() == "true"
    building_provider = await _get_setting(db, "building_footprint_provider", "msft")
    overture_release = await _get_setting(db, "overture_release", "2026-02-18.0")
    return {
        "api_key": api_key,
        "validation_model": validation_model,
        "boundary_model": boundary_model,
        "verification_model": verification_model,
        "zoom": scan.zoom or default_zoom,
        "buffer_m": scan.buffer_m or default_buffer,
        "overview_zoom": overview_zoom,
        "validation_prompt": validation_prompt,
        "boundary_prompt": boundary_prompt,
        "verification_prompt": verification_prompt,
        "correction_mode": correction_mode,
        "verification_correction_prompt": verification_correction_prompt,
        "bbox_validation_enabled": bbox_validation_enabled,
        "building_provider": building_provider,
        "overture_release": overture_release,
    }


async def _re_geocode(db: AsyncSession, facility_name: str, address: str) -> tuple[float, float] | None:
    """Try to re-geocode via Serper Maps. Returns (lat, lng) or None on failure."""
    key = await _get_setting(db, "serper_api_key")
    if not key:
        return None
    if not facility_name and not address:
        return None
    try:
        result = await serper_maps_resolve(key, facility_name, address)
        if result:
            return (result["lat"], result["lng"])
    except Exception as e:
        logger.warning("Serper re-geocode error: %s", e)
    return None


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

    # Re-geocode via Serper if address is available (fixes stale coords from older geocoders)
    if scan.facility_address:
        new_coords = await _re_geocode(db, facility_name, scan.facility_address)
        if new_coords:
            new_lat, new_lng = new_coords
            if abs(new_lat - (lat or 0)) > 0.0005 or abs(new_lng - (lng or 0)) > 0.0005:
                logger.info(
                    "Re-geocode updated coords for scan %s: (%.6f, %.6f) → (%.6f, %.6f)",
                    scan_id, lat or 0, lng or 0, new_lat, new_lng,
                )
            lat, lng = new_lat, new_lng
            scan.lat = lat
            scan.lng = lng
            await db.commit()

    if lat is None or lng is None:
        logger.warning("Skipping scan %s — no coordinates", scan_id)
        scan.status = ScanStatus.failed
        scan.error_message = "No coordinates available"
        scan.completed_at = datetime.now(timezone.utc)
        await db.commit()
        return

    # Clear any old steps/screenshots from a previous run of this scan
    # Delete orphaned screenshot files from disk before removing DB records
    old_screenshots = (await db.execute(
        select(Screenshot.file_path).where(Screenshot.scan_id == scan_id)
    )).scalars().all()
    for rel_path in old_screenshots:
        abs_path = os.path.join(settings.volume_path, rel_path)
        try:
            os.unlink(abs_path)
        except OSError:
            pass

    await db.execute(delete(ScanStep).where(ScanStep.scan_id == scan_id))
    await db.execute(delete(Screenshot).where(Screenshot.scan_id == scan_id))
    scan.error_message = None
    scan.method = None
    scan.osm_building_count = None
    scan.osm_building_id = None
    scan.bbox_min_lat = scan.bbox_min_lng = None
    scan.bbox_max_lat = scan.bbox_max_lng = None
    scan.bbox_width_m = scan.bbox_height_m = None
    scan.ai_confidence = scan.ai_facility_type = None
    scan.ai_building_count = scan.ai_notes = None
    scan.ai_validated = None
    scan.tile_count = scan.tiles_downloaded = None
    scan.image_width = scan.image_height = None
    scan.skip_reason = None
    scan.osm_duration_ms = scan.ai_duration_ms = scan.tile_duration_ms = None
    scan.completed_at = None
    scan.started_at = datetime.now(timezone.utc)
    await db.commit()

    config = await _get_scan_config(db, scan)
    cache = TileCache(os.path.join(settings.volume_path, "tile_cache", str(scan_id)))
    zoom = config["zoom"]
    buffer_m = config["buffer_m"]
    overview_zoom = config["overview_zoom"]

    # Will be set by MSFT, OSM, or AI vision
    final_bbox = None
    method = None
    step_num = 0
    osm_bbox = None  # shared variable — set by MSFT or OSM, consumed by AI validation
    msft_buildings = []  # hoisted — populated by step 1, used by step 3 for overlay
    msft_target_ids = set()  # tracks which building(s) are the selected target
    _heavy_acquired = False

    try:
        # ── STEP 1: OSM BUILDINGS CHECK ──────────────────────────
        step_num += 1
        scan.status = ScanStatus.running_osm
        await db.commit()

        osm_start = time.monotonic()
        buildings = await query_osm_buildings(lat, lng, settings.search_radius_m)
        scan.osm_building_count = len(buildings)

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

        # ── STEP 2: MSFT/OVERTURE FALLBACK (if OSM missed) ───────
        if osm_bbox is None:
            step_num += 1
            building_provider = config["building_provider"]
            scan.status = ScanStatus.running_overture if building_provider == "overture" else ScanStatus.running_msft
            await db.commit()
            msft_start = time.monotonic()

            if building_provider == "overture":
                msft_buildings = await query_overture_buildings(
                    lat, lng, settings.search_radius_m, release=config["overture_release"]
                )
            else:
                msft_buildings = await query_msft_buildings(lat, lng, settings.search_radius_m)

            msft_decision = None
            msft_building_info = None

            if msft_buildings:
                building = find_target_building_msft(msft_buildings, lat, lng)
                if building:
                    msft_target_ids = set(building.get("ids", [building["id"]]))
                    lats = [c[0] for c in building["coords"]]
                    lngs = [c[1] for c in building["coords"]]
                    raw_bbox = (min(lats), min(lngs), max(lats), max(lngs))
                    osm_bbox = bbox_add_buffer(*raw_bbox, buffer_m)
                    method = ScanMethod.overture_buildings if building_provider == "overture" else ScanMethod.msft_buildings
                    msft_building_info = {
                        "coord_count": len(building["coords"]),
                        "raw_bbox": list(raw_bbox),
                        "buffered_bbox": list(osm_bbox),
                    }
                    msft_decision = f"Found target building among {len(msft_buildings)} {building_provider.upper()} buildings. Applied {buffer_m}m buffer."
                else:
                    msft_decision = f"Found {len(msft_buildings)} {building_provider.upper()} buildings but none matched target coordinates."
            else:
                msft_decision = f"No {building_provider.upper()} buildings found (no data for region or query failed)."

            msft_elapsed = int((time.monotonic() - msft_start) * 1000)
            step_name = "overture_buildings_check" if building_provider == "overture" else "msft_buildings_check"
            await _record_step(
                db, scan_id, step_num, step_name,
                status="completed",
                duration_ms=msft_elapsed,
                input_summary=json.dumps({
                    "lat": lat,
                    "lng": lng,
                    "search_radius_m": settings.search_radius_m,
                    "provider": building_provider,
                }),
                output_summary=json.dumps({
                    "buildings_found": len(msft_buildings),
                    "target": msft_building_info,
                    "bbox": list(osm_bbox) if osm_bbox else None,
                }),
                decision=msft_decision,
            )

        # ── HEAVY PHASE GATE (steps 3-5 under memory semaphore) ──
        await _heavy_semaphore.acquire()
        _heavy_acquired = True

        # ── STEP 3: AI VALIDATION (if MSFT or OSM found something) ─
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

                # Compute MSFT polygon pixel coords for frontend overlay
                msft_polygons_px = None
                msft_target_mask = None
                if msft_buildings and ov_grid:
                    msft_polygons_px = []
                    msft_target_mask = []
                    for b in msft_buildings:
                        pixels = [list(latlng_to_pixel(blat, blng, ov_grid)) for blat, blng in b["coords"]]
                        if len(pixels) >= 3:
                            msft_polygons_px.append(pixels)
                            msft_target_mask.append(b["id"] in msft_target_ids)

                width_m, height_m = bbox_dimensions_m(*osm_bbox)

                validation = None
                validation_error = None
                val_decision = None

                if config["bbox_validation_enabled"]:
                    try:
                        validation = await asyncio.to_thread(
                            validate_osm_bbox,
                            abs_path, facility_name, lat, lng, width_m, height_m,
                            config["api_key"], config["validation_model"],
                            config["validation_prompt"],
                        )
                    except Exception as e:
                        if _is_fatal_ai_error(e):
                            raise RuntimeError(f"OpenAI API account error: {e}") from e
                        validation_error = str(e)
                        logger.warning("Validation call failed: %s", e)

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
                else:
                    final_bbox = osm_bbox
                    val_decision = "BBox AI validation disabled in settings. Using OSM/MSFT bbox directly."
                    logger.info("BBox AI validation disabled, using bbox directly")

                ai_elapsed = int((time.monotonic() - ai_start) * 1000)

                await _record_step(
                    db, scan_id, step_num, "ai_validation",
                    status="skipped" if not config["bbox_validation_enabled"] else "completed",
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
                        "msft_polygons_px": msft_polygons_px,
                        "msft_target_mask": msft_target_mask if msft_polygons_px else None,
                    }),
                    output_summary=json.dumps({
                        "approved": validation.approved if validation else None,
                        "facility_type": validation.facility_type if validation else None,
                        "reason": validation.reason if validation else None,
                        "notes": validation.notes if validation else None,
                        "error": validation_error,
                    }),
                    decision=val_decision,
                    ai_model=config["validation_model"] if config["bbox_validation_enabled"] else None,
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

        # ── STEP 4: AI VISION FALLBACK ─────────────────────────────
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
                    if _is_fatal_ai_error(e):
                        raise RuntimeError(f"OpenAI API account error: {e}") from e
                    boundary_error = str(e)
                    logger.warning("Boundary detection failed: %s", e)

                vision_decision = None
                verification = None
                verification_error = None
                original_boundary = None
                retries = []  # list of rejected attempt dicts
                retry_count = 0
                max_retries = 2

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
                                "reasoning": boundary.reasoning,
                                "self_check": boundary.self_check,
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

                    if config["correction_mode"] == "v2":
                        # V2: Single-agent — verify_and_correct_boundary()
                        last_evaluated = boundary  # track what was passed to V2 (for tracing)
                        try:
                            vc_result = await asyncio.to_thread(
                                verify_and_correct_boundary,
                                abs_path, facility_name, lat, lng, width_m, height_m,
                                boundary, config["api_key"], config["boundary_model"],
                                config["boundary_prompt"],
                                config["verification_correction_prompt"],
                            )
                            if vc_result:
                                verification = vc_result.verification
                                if vc_result.correction:
                                    boundary = vc_result.correction
                                    # Bug 1 fix: update scan metadata after initial correction
                                    scan.ai_confidence = boundary.confidence
                                    scan.ai_facility_type = boundary.facility_type
                                    scan.ai_building_count = boundary.building_count
                                    scan.ai_notes = boundary.notes
                        except Exception as e:
                            if _is_fatal_ai_error(e):
                                raise RuntimeError(f"OpenAI API account error: {e}") from e
                            verification_error = str(e)
                            logger.warning("V2 verify-and-correct failed: %s", e)

                        # V2 retry loop
                        while verification and not verification.approved and retry_count < max_retries:
                            retry_count += 1
                            logger.info(
                                "V2 boundary rejected (attempt %d/%d): %s. Retrying...",
                                retry_count, max_retries, verification.reason,
                            )

                            # Record rejected attempt — use last_evaluated (the boundary
                            # that was actually passed to V2), not boundary (which may
                            # already be the correction)
                            rejected_attempt = {
                                "attempt": retry_count,
                                "correction_mode": "v2",
                                "boundary": {
                                    "pixel_bbox": {"top_y": last_evaluated.top_y, "bottom_y": last_evaluated.bottom_y,
                                                   "left_x": last_evaluated.left_x, "right_x": last_evaluated.right_x},
                                    "confidence": last_evaluated.confidence,
                                    "facility_type": last_evaluated.facility_type,
                                    "building_count": last_evaluated.building_count,
                                    "notes": last_evaluated.notes,
                                    "reasoning": last_evaluated.reasoning,
                                    "self_check": last_evaluated.self_check,
                                    "raw_response": last_evaluated.raw_response,
                                    "tokens": {
                                        "prompt": last_evaluated.usage.prompt_tokens,
                                        "completion": last_evaluated.usage.completion_tokens,
                                        "total": last_evaluated.usage.total_tokens,
                                    } if last_evaluated.usage else None,
                                },
                                "verification": {
                                    "approved": verification.approved,
                                    "reason": verification.reason,
                                    "issues": verification.issues,
                                    "edge_feedback": verification.edge_feedback,
                                    "raw_response": verification.raw_response,
                                    "tokens": {
                                        "prompt": verification.usage.prompt_tokens,
                                        "completion": verification.usage.completion_tokens,
                                        "total": verification.usage.total_tokens,
                                    } if verification.usage else None,
                                },
                            }

                            if retry_count == 1:
                                original_boundary = rejected_attempt["boundary"].copy()
                                original_boundary["prompt_text"] = last_evaluated.prompt_text

                            retries.append(rejected_attempt)

                            # Update tracker: boundary is what we're about to evaluate next
                            last_evaluated = boundary

                            # Retry with previous_verification for correction context
                            try:
                                vc_result = await asyncio.to_thread(
                                    verify_and_correct_boundary,
                                    abs_path, facility_name, lat, lng, width_m, height_m,
                                    boundary, config["api_key"], config["boundary_model"],
                                    config["boundary_prompt"],
                                    config["verification_correction_prompt"],
                                    verification,  # previous_verification
                                )
                                if vc_result:
                                    verification = vc_result.verification
                                    if vc_result.correction:
                                        boundary = vc_result.correction
                                    elif not verification.approved:
                                        # Model rejected but didn't provide coords — treat as stuck
                                        logger.warning("V2 retry %d: rejected without corrected coords", retry_count)
                                        break
                                else:
                                    break
                            except Exception as e:
                                if _is_fatal_ai_error(e):
                                    raise RuntimeError(f"OpenAI API account error: {e}") from e
                                logger.warning("V2 retry %d failed: %s", retry_count, e)
                                break

                            # Update scan with latest boundary
                            scan.ai_confidence = boundary.confidence
                            scan.ai_facility_type = boundary.facility_type
                            scan.ai_building_count = boundary.building_count
                            scan.ai_notes = boundary.notes

                    else:
                        # V1: Multi-agent — verify → correct → verify loop (existing)
                        try:
                            verification = await asyncio.to_thread(
                                verify_facility_boundary,
                                abs_path, facility_name, boundary,
                                config["api_key"], config["verification_model"],
                                config["verification_prompt"],
                            )
                        except Exception as e:
                            if _is_fatal_ai_error(e):
                                raise RuntimeError(f"OpenAI API account error: {e}") from e
                            verification_error = str(e)
                            logger.warning("Verification call failed: %s", e)

                        while verification and not verification.approved and retry_count < max_retries:
                            retry_count += 1
                            logger.info(
                                "Boundary rejected (attempt %d/%d): %s. Retrying with feedback...",
                                retry_count, max_retries, verification.reason,
                            )

                            rejected_attempt = {
                                "attempt": retry_count,
                                "boundary": {
                                    "pixel_bbox": {"top_y": boundary.top_y, "bottom_y": boundary.bottom_y,
                                                   "left_x": boundary.left_x, "right_x": boundary.right_x},
                                    "confidence": boundary.confidence,
                                    "facility_type": boundary.facility_type,
                                    "building_count": boundary.building_count,
                                    "notes": boundary.notes,
                                    "reasoning": boundary.reasoning,
                                    "self_check": boundary.self_check,
                                    "raw_response": boundary.raw_response,
                                    "tokens": {
                                        "prompt": boundary.usage.prompt_tokens,
                                        "completion": boundary.usage.completion_tokens,
                                        "total": boundary.usage.total_tokens,
                                    } if boundary.usage else None,
                                },
                                "verification": {
                                    "approved": verification.approved,
                                    "reason": verification.reason,
                                    "issues": verification.issues,
                                    "edge_feedback": verification.edge_feedback,
                                    "raw_response": verification.raw_response,
                                    "tokens": {
                                        "prompt": verification.usage.prompt_tokens,
                                        "completion": verification.usage.completion_tokens,
                                        "total": verification.usage.total_tokens,
                                    } if verification.usage else None,
                                },
                            }

                            if retry_count == 1:
                                original_boundary = rejected_attempt["boundary"].copy()
                                original_boundary["prompt_text"] = boundary.prompt_text

                            retries.append(rejected_attempt)

                            try:
                                boundary = await asyncio.to_thread(
                                    correct_facility_boundary,
                                    abs_path, facility_name, lat, lng, width_m, height_m,
                                    boundary, verification,
                                    config["api_key"], config["boundary_model"],
                                    config["boundary_prompt"],
                                )
                            except Exception as e:
                                if _is_fatal_ai_error(e):
                                    raise RuntimeError(f"OpenAI API account error: {e}") from e
                                boundary = None
                                logger.warning("Boundary correction %d failed: %s", retry_count, e)
                                break

                            if not boundary:
                                break

                            scan.ai_confidence = boundary.confidence
                            scan.ai_facility_type = boundary.facility_type
                            scan.ai_building_count = boundary.building_count
                            scan.ai_notes = boundary.notes

                            verification = None
                            try:
                                verification = await asyncio.to_thread(
                                    verify_facility_boundary,
                                    abs_path, facility_name, boundary,
                                    config["api_key"], config["verification_model"],
                                    config["verification_prompt"],
                                )
                            except Exception as e:
                                if _is_fatal_ai_error(e):
                                    raise RuntimeError(f"OpenAI API account error: {e}") from e
                                logger.warning("Retry %d verification failed: %s", retry_count, e)
                                break

                    # Convert pixel bbox to lat/lng
                    if boundary:
                        lat1, lng1 = pixel_to_latlng(boundary.left_x, boundary.top_y, ov_grid)
                        lat2, lng2 = pixel_to_latlng(boundary.right_x, boundary.bottom_y, ov_grid)
                        raw_bbox = (min(lat1, lat2), min(lng1, lng2), max(lat1, lat2), max(lng1, lng2))
                        final_bbox = bbox_add_buffer(*raw_bbox, buffer_m)
                        method = ScanMethod.ai_vision

                        # Determine verification status string
                        if not verification or verification.approved:
                            if retry_count == 0:
                                verified_str = "verified"
                            else:
                                verified_str = f"verified after {retry_count} {'retry' if retry_count == 1 else 'retries'}"
                        else:
                            verified_str = f"unverified (rejected after {retry_count + 1} attempts)"
                        vision_decision = (
                            f"AI detected facility boundary ({verified_str}) — confidence: {boundary.confidence}, "
                            f"type: {boundary.facility_type}, {boundary.building_count} buildings. "
                            f"Pixel bbox: ({boundary.left_x},{boundary.top_y})-({boundary.right_x},{boundary.bottom_y}) "
                            f"→ raw geo: {[round(x, 6) for x in raw_bbox]} "
                            f"→ buffered: {[round(x, 6) for x in final_bbox]}"
                        )
                    else:
                        if retry_count > 0:
                            vision_decision = f"AI boundary detection failed after {retry_count} {'retry' if retry_count == 1 else 'retries'}. Will fall back to radius."
                        else:
                            vision_decision = f"AI boundary detection failed: {boundary_error or 'unknown error'}. Will fall back to radius."
                else:
                    vision_decision = f"AI boundary detection failed: {boundary_error or 'unknown error'}. Will fall back to radius."

                ai_elapsed = int((time.monotonic() - ai_start) * 1000)

                # Build final verification dict for output_summary
                final_verification = None
                if verification:
                    v_model = config["boundary_model"] if config["correction_mode"] == "v2" else config["verification_model"]
                    final_verification = {
                        "model": v_model,
                        "approved": verification.approved,
                        "reason": verification.reason,
                        "issues": verification.issues,
                        "raw_response": verification.raw_response,
                        "tokens": {
                            "prompt": verification.usage.prompt_tokens,
                            "completion": verification.usage.completion_tokens,
                            "total": verification.usage.total_tokens,
                        } if verification.usage else None,
                    }
                elif verification_error:
                    final_verification = {"error": verification_error}

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
                        "correction_mode": config["correction_mode"],
                    }),
                    output_summary=json.dumps({
                        "confidence": boundary.confidence if boundary else None,
                        "facility_type": boundary.facility_type if boundary else None,
                        "building_count": boundary.building_count if boundary else None,
                        "notes": boundary.notes if boundary else None,
                        "reasoning": boundary.reasoning if boundary else None,
                        "self_check": boundary.self_check if boundary else None,
                        "pixel_bbox": {"top_y": boundary.top_y, "bottom_y": boundary.bottom_y,
                                       "left_x": boundary.left_x, "right_x": boundary.right_x} if boundary else None,
                        "raw_geo_bbox": list(raw_bbox) if final_bbox and method == ScanMethod.ai_vision else None,
                        "detected_bbox": list(final_bbox) if final_bbox and method == ScanMethod.ai_vision else None,
                        "boundary_error": boundary_error,
                        "verification": final_verification,
                        "retries": retries if retries else None,
                        "retry_count": retry_count,
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

        # ── FAIL if no boundary found ─────────────────────────────
        if final_bbox is None:
            raise RuntimeError("No building boundary found by OSM, MSFT, or AI Vision.")

        # Record bbox
        scan.method = method
        scan.bbox_min_lat, scan.bbox_min_lng = final_bbox[0], final_bbox[1]
        scan.bbox_max_lat, scan.bbox_max_lng = final_bbox[2], final_bbox[3]
        width_m, height_m = bbox_dimensions_m(*final_bbox)
        scan.bbox_width_m = width_m
        scan.bbox_height_m = height_m
        await db.commit()

        # ── STEP 5: TILING ─────────────────────────────────────────
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

        # ── Release heavy-phase semaphore before lightweight finalization ──
        _heavy_semaphore.release()
        _heavy_acquired = False

        # ── STEP 6: DONE ──────────────────────────────────────────
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
    finally:
        if _heavy_acquired:
            _heavy_semaphore.release()
        cache.clear()
