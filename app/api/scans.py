"""Scan endpoints — submit, status, list, delete, bin detection."""

import json
import logging
import os
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings as app_settings
from app.database import get_db
from app.models import Scan, ScanStatus, ScanStep, Screenshot, Setting
from app.schemas import (
    BatchScanRequest,
    BulkImportRequest,
    ScanRequest,
    ScanResponse,
    ScanStepResponse,
    ScanSubmitted,
    ScanTraceResponse,
    ScreenshotResponse,
)
from app.worker import enqueue_scan

logger = logging.getLogger(__name__)
router = APIRouter()


def _scan_to_response(scan: Scan, base_url: str = "") -> ScanResponse:
    """Convert Scan ORM to ScanResponse with screenshot URLs."""
    screenshots = []
    for ss in scan.screenshots:
        screenshots.append(ScreenshotResponse(
            id=ss.id,
            type=ss.type.value if hasattr(ss.type, "value") else ss.type,
            filename=ss.filename,
            url=f"/api/screenshots/{ss.id}",
            thumb_url=f"/api/screenshots/{ss.id}?thumb=1",
            width=ss.width,
            height=ss.height,
            zoom=ss.zoom,
            file_size_bytes=ss.file_size_bytes,
        ))

    return ScanResponse(
        id=scan.id,
        facility_name=scan.facility_name,
        facility_address=scan.facility_address,
        domain=scan.domain,
        facility_lat=scan.lat,
        facility_lng=scan.lng,
        status=scan.status.value if hasattr(scan.status, "value") else scan.status,
        method=scan.method.value if scan.method and hasattr(scan.method, "value") else scan.method,
        zoom=scan.zoom,
        buffer_m=scan.buffer_m,
        osm_building_count=scan.osm_building_count,
        bbox_min_lat=scan.bbox_min_lat,
        bbox_min_lng=scan.bbox_min_lng,
        bbox_max_lat=scan.bbox_max_lat,
        bbox_max_lng=scan.bbox_max_lng,
        bbox_width_m=scan.bbox_width_m,
        bbox_height_m=scan.bbox_height_m,
        ai_confidence=scan.ai_confidence,
        ai_facility_type=scan.ai_facility_type,
        ai_building_count=scan.ai_building_count,
        ai_notes=scan.ai_notes,
        ai_validated=scan.ai_validated,
        tile_count=scan.tile_count,
        tiles_downloaded=scan.tiles_downloaded,
        image_width=scan.image_width,
        image_height=scan.image_height,
        bin_present=scan.bin_present,
        bin_count=scan.bin_count,
        bin_filled_count=scan.bin_filled_count,
        bin_empty_count=scan.bin_empty_count,
        bin_confidence=scan.bin_confidence,
        bin_detection_status=scan.bin_detection_status,
        error_message=scan.error_message,
        skip_reason=scan.skip_reason,
        started_at=scan.started_at,
        completed_at=scan.completed_at,
        osm_duration_ms=scan.osm_duration_ms,
        ai_duration_ms=scan.ai_duration_ms,
        tile_duration_ms=scan.tile_duration_ms,
        screenshots=screenshots,
    )


@router.post("/scan", response_model=ScanSubmitted, status_code=202)
async def submit_scan(req: ScanRequest, db: AsyncSession = Depends(get_db)):
    """Submit a single facility for scanning."""
    scan = Scan(
        facility_name=req.name,
        facility_address=req.address,
        lat=req.lat,
        lng=req.lng,
        zoom=req.zoom,
        buffer_m=req.buffer_m,
    )
    db.add(scan)
    await db.commit()
    await db.refresh(scan)

    await enqueue_scan(scan.id)

    return ScanSubmitted(scan_id=scan.id)


@router.post("/scan/batch", response_model=list[ScanSubmitted], status_code=202)
async def submit_batch(req: BatchScanRequest, db: AsyncSession = Depends(get_db)):
    """Submit up to 50 facilities for scanning."""
    results = []
    for item in req.facilities:
        scan = Scan(
            facility_name=item.name,
            facility_address=item.address,
            lat=item.lat,
            lng=item.lng,
            zoom=item.zoom,
            buffer_m=item.buffer_m,
        )
        db.add(scan)
        await db.commit()
        await db.refresh(scan)

        await enqueue_scan(scan.id)
        results.append(ScanSubmitted(scan_id=scan.id))

    return results


@router.post("/scan/import", response_model=list[ScanSubmitted], status_code=202)
async def import_facilities(req: BulkImportRequest, db: AsyncSession = Depends(get_db)):
    """Import facilities by name/address — no coordinates, no worker queue.

    Duplicates are detected by facility_name + facility_address (case-insensitive).
    Existing rows are left untouched except domain is backfilled if missing.
    """
    results = []
    for item in req.facilities:
        # Check for existing facility by name + address (case-insensitive)
        q = select(Scan).where(
            func.lower(Scan.facility_name) == (item.name or "").strip().lower(),
            func.lower(func.coalesce(Scan.facility_address, ""))
            == (item.address or "").strip().lower(),
        )
        existing = (await db.execute(q)).scalar_one_or_none()

        if existing:
            # Backfill domain only if the existing row has none
            if not existing.domain and item.domain:
                existing.domain = item.domain
            await db.flush()
            results.append(ScanSubmitted(scan_id=existing.id, status=existing.status))
        else:
            scan = Scan(
                facility_name=item.name,
                facility_address=item.address,
                domain=item.domain,
                lat=None,
                lng=None,
                status=ScanStatus.pending,
            )
            db.add(scan)
            await db.flush()
            results.append(ScanSubmitted(scan_id=scan.id, status="pending"))

    await db.commit()
    return results


@router.get("/scan/{scan_id}", response_model=ScanResponse)
async def get_scan(scan_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get scan status and results."""
    scan = await db.get(Scan, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return _scan_to_response(scan)


@router.get("/scans", response_model=list[ScanResponse])
async def list_scans(
    status: str | None = Query(None),
    exclude_status: str | None = Query(None),
    method: str | None = Query(None),
    bins: str | None = Query(None),
    search: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List scans with optional filtering and facility name/address search."""
    q = select(Scan).order_by(Scan.started_at.desc().nullslast())

    if status:
        q = q.where(Scan.status == status)
    if exclude_status:
        q = q.where(Scan.status != exclude_status)
    if method:
        q = q.where(Scan.method == method)
    if bins == "true":
        q = q.where(Scan.bin_present == True)
    elif bins == "false":
        q = q.where(Scan.bin_present == False)
    if search:
        q = q.where(Scan.facility_name.ilike(f"%{search}%") | Scan.facility_address.ilike(f"%{search}%"))

    q = q.offset(offset).limit(limit)
    result = await db.execute(q)
    scans = result.scalars().all()
    return [_scan_to_response(scan) for scan in scans]


@router.delete("/scans", status_code=200)
async def delete_scans(
    scan_ids: list[UUID] = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
):
    """Delete scans and all related records (screenshots, steps)."""
    if not scan_ids:
        raise HTTPException(status_code=400, detail="No scan IDs provided")
    if len(scan_ids) > 200:
        raise HTTPException(status_code=400, detail="Max 200 scans per delete")

    # Delete child records first (no cascade on FK)
    await db.execute(delete(Screenshot).where(Screenshot.scan_id.in_(scan_ids)))
    await db.execute(delete(ScanStep).where(ScanStep.scan_id.in_(scan_ids)))
    result = await db.execute(delete(Scan).where(Scan.id.in_(scan_ids)))
    await db.commit()

    return {"deleted": result.rowcount}


@router.post("/scans/enqueue", status_code=200)
async def enqueue_scans(
    scan_ids: list[UUID] = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
):
    """Enqueue specific scans for processing."""
    if not scan_ids:
        raise HTTPException(status_code=400, detail="No scan IDs provided")
    if len(scan_ids) > 200:
        raise HTTPException(status_code=400, detail="Max 200 scans per enqueue")

    result = await db.execute(select(Scan).where(Scan.id.in_(scan_ids)))
    scans = result.scalars().all()

    queued = 0
    for scan in scans:
        scan.status = ScanStatus.queued
        scan.error_message = None
        queued += 1

    await db.commit()

    for scan in scans:
        await enqueue_scan(scan.id)

    return {"queued": queued}


@router.post("/scans/enqueue-pending", status_code=200)
async def enqueue_all_pending(db: AsyncSession = Depends(get_db)):
    """Enqueue all pending scans — pipeline handles geocoding from address."""
    result = await db.execute(
        select(Scan).where(Scan.status == ScanStatus.pending)
    )
    scans = result.scalars().all()

    queued = 0
    for scan in scans:
        scan.status = ScanStatus.queued
        scan.error_message = None
        queued += 1

    await db.commit()

    for scan in scans:
        await enqueue_scan(scan.id)

    return {"queued": queued}


async def _get_setting(db: AsyncSession, key: str, default: str = "") -> str:
    """Get a setting value from the database."""
    result = await db.execute(select(Setting.value).where(Setting.key == key))
    row = result.scalar_one_or_none()
    return row if row is not None else default


async def _run_bin_detection_for_scan(scan: Scan, db: AsyncSession) -> dict:
    """Run bin detection on a completed scan. Used by both single and bulk endpoints."""
    from app.scanner.bin_detection import execute_bin_detection

    if scan.status != ScanStatus.completed:
        return {"scan_id": str(scan.id), "status": "skipped", "reason": "not completed"}

    # Find final screenshot
    final_ss = None
    for ss in scan.screenshots:
        ss_type = ss.type.value if hasattr(ss.type, "value") else ss.type
        if ss_type == "final":
            final_ss = ss
            break
    if not final_ss:
        return {"scan_id": str(scan.id), "status": "skipped", "reason": "no final image"}

    final_path = os.path.join(app_settings.volume_path, final_ss.file_path)
    if not os.path.exists(final_path):
        return {"scan_id": str(scan.id), "status": "skipped", "reason": "final image file not found"}

    if not scan.bbox_width_m or not scan.bbox_height_m:
        return {"scan_id": str(scan.id), "status": "skipped", "reason": "no bbox dimensions"}

    # Load settings
    api_key = await _get_setting(db, "openai_api_key", app_settings.openai_api_key)
    if not api_key:
        return {"scan_id": str(scan.id), "status": "skipped", "reason": "no API key"}

    prompt = await _get_setting(db, "bin_detection_prompt", "")
    if not prompt:
        return {"scan_id": str(scan.id), "status": "skipped", "reason": "no bin detection prompt"}

    model = await _get_setting(db, "bin_detection_model", "gpt-5.2")
    try:
        max_chunk_m = max(50, int(await _get_setting(db, "bin_detection_max_chunk_m", "100")))
    except (ValueError, TypeError):
        max_chunk_m = 100
    try:
        min_confidence = max(0, min(100, int(await _get_setting(db, "bin_detection_min_confidence", "50"))))
    except (ValueError, TypeError):
        min_confidence = 50
    delete_final = (await _get_setting(db, "bin_delete_final_image", "false")).lower() == "true"
    resize_final = (await _get_setting(db, "bin_resize_final_image", "false")).lower() == "true"
    include_reasoning = (await _get_setting(db, "bin_detection_reasoning", "true")).lower() == "true"

    # Determine step number (after existing steps)
    max_step_result = await db.execute(
        select(func.max(ScanStep.step_number)).where(ScanStep.scan_id == scan.id)
    )
    max_step = max_step_result.scalar() or 0

    return await execute_bin_detection(
        scan, db, final_path,
        api_key=api_key,
        model=model,
        prompt=prompt,
        max_chunk_m=max_chunk_m,
        min_confidence=min_confidence,
        step_num_start=max_step + 1,
        volume_path=app_settings.volume_path,
        clean_old=True,
        delete_final_image=delete_final,
        resize_final_image=resize_final,
        include_reasoning=include_reasoning,
    )


@router.post("/scan/{scan_id}/detect-bins", status_code=200)
async def detect_bins_single(scan_id: UUID, db: AsyncSession = Depends(get_db)):
    """Run bin detection on a single completed scan."""
    scan = await db.get(Scan, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    try:
        result = await _run_bin_detection_for_scan(scan, db)
    except Exception as e:
        logger.error("Bin detection failed for scan %s: %s", scan_id, e)
        raise HTTPException(status_code=500, detail=str(e))

    return result


@router.post("/scans/detect-bins", status_code=200)
async def detect_bins_bulk(
    scan_ids: list[UUID] = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
):
    """Run bin detection on multiple completed scans."""
    if not scan_ids:
        raise HTTPException(status_code=400, detail="No scan IDs provided")
    if len(scan_ids) > 50:
        raise HTTPException(status_code=400, detail="Max 50 scans per batch")

    result_q = await db.execute(select(Scan).where(Scan.id.in_(scan_ids)))
    scans = result_q.scalars().all()

    results = []
    processed = 0
    skipped = 0

    for scan in scans:
        try:
            result = await _run_bin_detection_for_scan(scan, db)
            if result.get("status") == "skipped":
                skipped += 1
            else:
                processed += 1
            results.append(result)
        except Exception as e:
            logger.error("Bin detection failed for scan %s: %s", scan.id, e)
            results.append({
                "scan_id": str(scan.id),
                "status": "failed",
                "error": str(e),
            })

    return {
        "processed": processed,
        "skipped": skipped,
        "results": results,
    }


def _parse_json_field(text: str | None) -> dict | None:
    """Safely parse a JSON string field, return dict or None."""
    if not text:
        return None
    try:
        import json
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


@router.get("/scan/{scan_id}/steps", response_model=ScanTraceResponse)
async def get_scan_steps(scan_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get full pipeline trace with step-by-step decisions, AI prompts, and token usage."""
    scan = await db.get(Scan, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    steps = []
    total_ai_tokens = 0
    for step in scan.steps:
        steps.append(ScanStepResponse(
            id=step.id,
            step_number=step.step_number,
            step_name=step.step_name,
            status=step.status,
            started_at=step.started_at,
            completed_at=step.completed_at,
            duration_ms=step.duration_ms,
            input_summary=_parse_json_field(step.input_summary),
            output_summary=_parse_json_field(step.output_summary),
            decision=step.decision,
            ai_model=step.ai_model,
            ai_prompt=step.ai_prompt,
            ai_response_raw=step.ai_response_raw,
            ai_tokens_prompt=step.ai_tokens_prompt,
            ai_tokens_completion=step.ai_tokens_completion,
            ai_tokens_reasoning=step.ai_tokens_reasoning,
            ai_tokens_total=step.ai_tokens_total,
            tile_grid_cols=step.tile_grid_cols,
            tile_grid_rows=step.tile_grid_rows,
        ))
        if step.ai_tokens_total:
            total_ai_tokens += step.ai_tokens_total

    total_ms = None
    if scan.started_at and scan.completed_at:
        total_ms = int((scan.completed_at - scan.started_at).total_seconds() * 1000)

    return ScanTraceResponse(
        scan_id=scan.id,
        facility_name=scan.facility_name,
        lat=scan.lat,
        lng=scan.lng,
        status=scan.status.value if hasattr(scan.status, "value") else scan.status,
        method=scan.method.value if scan.method and hasattr(scan.method, "value") else scan.method,
        started_at=scan.started_at,
        completed_at=scan.completed_at,
        total_duration_ms=total_ms,
        total_ai_tokens=total_ai_tokens if total_ai_tokens > 0 else None,
        steps=steps,
    )
