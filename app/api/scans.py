"""Scan endpoints — submit, status, list."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Facility, Scan, ScanStatus, ScanStep
from app.schemas import (
    BatchScanRequest,
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


def _scan_to_response(scan: Scan, facility_name: str = "", base_url: str = "") -> ScanResponse:
    """Convert Scan ORM to ScanResponse with screenshot URLs."""
    screenshots = []
    for ss in scan.screenshots:
        screenshots.append(ScreenshotResponse(
            id=ss.id,
            type=ss.type.value if hasattr(ss.type, "value") else ss.type,
            filename=ss.filename,
            url=f"/api/screenshots/{ss.id}",
            width=ss.width,
            height=ss.height,
            zoom=ss.zoom,
        ))

    return ScanResponse(
        id=scan.id,
        facility_id=scan.facility_id,
        facility_name=facility_name,
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
        error_message=scan.error_message,
        skip_reason=scan.skip_reason,
        started_at=scan.started_at,
        completed_at=scan.completed_at,
        osm_duration_ms=scan.osm_duration_ms,
        ai_duration_ms=scan.ai_duration_ms,
        tile_duration_ms=scan.tile_duration_ms,
        screenshots=screenshots,
    )


async def _get_or_create_facility(
    db: AsyncSession, name: str, lat: float, lng: float, address: str | None = None
) -> Facility:
    """Find existing facility by coords or create a new one."""
    result = await db.execute(
        select(Facility).where(Facility.lat == lat, Facility.lng == lng)
    )
    facility = result.scalar_one_or_none()
    if facility:
        return facility

    facility = Facility(name=name, lat=lat, lng=lng, address=address)
    db.add(facility)
    await db.commit()
    await db.refresh(facility)
    return facility


@router.post("/scan", response_model=ScanSubmitted, status_code=202)
async def submit_scan(req: ScanRequest, db: AsyncSession = Depends(get_db)):
    """Submit a single facility for scanning."""
    facility = await _get_or_create_facility(db, req.name, req.lat, req.lng, req.address)

    scan = Scan(
        facility_id=facility.id,
        zoom=req.zoom,
        buffer_m=req.buffer_m,
    )
    db.add(scan)
    await db.commit()
    await db.refresh(scan)

    await enqueue_scan(scan.id, facility.name, facility.lat, facility.lng)

    return ScanSubmitted(scan_id=scan.id, facility_id=facility.id)


@router.post("/scan/batch", response_model=list[ScanSubmitted], status_code=202)
async def submit_batch(req: BatchScanRequest, db: AsyncSession = Depends(get_db)):
    """Submit up to 50 facilities for scanning."""
    results = []
    for item in req.facilities:
        facility = await _get_or_create_facility(db, item.name, item.lat, item.lng, item.address)
        scan = Scan(
            facility_id=facility.id,
            zoom=item.zoom,
            buffer_m=item.buffer_m,
        )
        db.add(scan)
        await db.commit()
        await db.refresh(scan)

        await enqueue_scan(scan.id, facility.name, facility.lat, facility.lng)
        results.append(ScanSubmitted(scan_id=scan.id, facility_id=facility.id))

    return results


@router.get("/scan/{scan_id}", response_model=ScanResponse)
async def get_scan(scan_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get scan status and results."""
    scan = await db.get(Scan, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    facility = await db.get(Facility, scan.facility_id)
    return _scan_to_response(scan, facility_name=facility.name if facility else "")


@router.get("/scans", response_model=list[ScanResponse])
async def list_scans(
    status: str | None = Query(None),
    method: str | None = Query(None),
    search: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List scans with optional filtering and facility name search."""
    q = select(Scan, Facility.name.label("facility_name")).join(
        Facility, Scan.facility_id == Facility.id, isouter=True
    ).order_by(Scan.started_at.desc().nullslast())

    if status:
        q = q.where(Scan.status == status)
    if method:
        q = q.where(Scan.method == method)
    if search:
        q = q.where(Facility.name.ilike(f"%{search}%"))

    q = q.offset(offset).limit(limit)
    result = await db.execute(q)
    rows = result.all()
    return [_scan_to_response(row[0], facility_name=row[1] or "") for row in rows]


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

    facility = await db.get(Facility, scan.facility_id)

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
        facility_name=facility.name if facility else "unknown",
        lat=facility.lat if facility else 0,
        lng=facility.lng if facility else 0,
        status=scan.status.value if hasattr(scan.status, "value") else scan.status,
        method=scan.method.value if scan.method and hasattr(scan.method, "value") else scan.method,
        started_at=scan.started_at,
        completed_at=scan.completed_at,
        total_duration_ms=total_ms,
        total_ai_tokens=total_ai_tokens if total_ai_tokens > 0 else None,
        steps=steps,
    )
