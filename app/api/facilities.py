"""Facility endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Facility, Scan
from app.schemas import FacilityResponse

router = APIRouter()


@router.get("/facilities", response_model=list[FacilityResponse])
async def list_facilities(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List all facilities with scan counts."""
    # Subquery for scan count
    scan_count_sq = (
        select(Scan.facility_id, func.count(Scan.id).label("scan_count"))
        .group_by(Scan.facility_id)
        .subquery()
    )

    q = (
        select(Facility, func.coalesce(scan_count_sq.c.scan_count, 0).label("scan_count"))
        .outerjoin(scan_count_sq, Facility.id == scan_count_sq.c.facility_id)
        .order_by(Facility.created_at.desc())
        .offset(offset)
        .limit(limit)
    )

    result = await db.execute(q)
    rows = result.all()

    return [
        FacilityResponse(
            id=facility.id,
            name=facility.name,
            lat=facility.lat,
            lng=facility.lng,
            address=facility.address,
            created_at=facility.created_at,
            scan_count=scan_count,
        )
        for facility, scan_count in rows
    ]
