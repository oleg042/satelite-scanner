"""Pydantic request/response schemas."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# --- Requests ---

class ScanRequest(BaseModel):
    name: str
    lat: float
    lng: float
    address: Optional[str] = None
    zoom: Optional[int] = Field(None, ge=16, le=22)
    buffer_m: Optional[int] = Field(None, ge=10, le=500)
    overview_zoom: Optional[int] = Field(None, ge=16, le=21)
    validation_model: Optional[str] = None
    boundary_model: Optional[str] = None


class BatchScanRequest(BaseModel):
    facilities: list[ScanRequest] = Field(..., max_length=50)


class SettingsUpdate(BaseModel):
    openai_api_key: Optional[str] = None
    validation_model: Optional[str] = None
    boundary_model: Optional[str] = None
    default_zoom: Optional[str] = None
    default_buffer_m: Optional[str] = None
    overview_zoom: Optional[str] = None
    validation_prompt: Optional[str] = None
    boundary_prompt: Optional[str] = None


# --- Responses ---

class ScreenshotResponse(BaseModel):
    id: UUID
    type: str
    filename: str
    url: str
    width: Optional[int] = None
    height: Optional[int] = None
    zoom: Optional[int] = None

    model_config = {"from_attributes": True}


class ScanResponse(BaseModel):
    id: UUID
    facility_id: UUID
    status: str
    method: Optional[str] = None
    zoom: Optional[int] = None
    buffer_m: Optional[int] = None
    osm_building_count: Optional[int] = None
    bbox_min_lat: Optional[float] = None
    bbox_min_lng: Optional[float] = None
    bbox_max_lat: Optional[float] = None
    bbox_max_lng: Optional[float] = None
    bbox_width_m: Optional[float] = None
    bbox_height_m: Optional[float] = None
    ai_confidence: Optional[str] = None
    ai_facility_type: Optional[str] = None
    ai_building_count: Optional[int] = None
    ai_notes: Optional[str] = None
    ai_validated: Optional[bool] = None
    tile_count: Optional[int] = None
    tiles_downloaded: Optional[int] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    error_message: Optional[str] = None
    skip_reason: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    osm_duration_ms: Optional[int] = None
    ai_duration_ms: Optional[int] = None
    tile_duration_ms: Optional[int] = None
    screenshots: list[ScreenshotResponse] = []

    model_config = {"from_attributes": True}


class ScanSubmitted(BaseModel):
    scan_id: UUID
    facility_id: UUID
    status: str = "queued"


class FacilityResponse(BaseModel):
    id: UUID
    name: str
    lat: float
    lng: float
    address: Optional[str] = None
    created_at: datetime
    scan_count: int = 0

    model_config = {"from_attributes": True}


class SettingsResponse(BaseModel):
    openai_api_key: str = ""
    validation_model: str = ""
    boundary_model: str = ""
    default_zoom: str = ""
    default_buffer_m: str = ""
    overview_zoom: str = ""
    validation_prompt: str = ""
    boundary_prompt: str = ""


class HealthResponse(BaseModel):
    status: str = "ok"
    queue_size: int = 0
