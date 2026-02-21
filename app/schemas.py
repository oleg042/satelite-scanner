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


class BulkImportItem(BaseModel):
    name: str
    address: Optional[str] = None


class BulkImportRequest(BaseModel):
    facilities: list[BulkImportItem] = Field(..., min_length=1, max_length=200)


class BatchScanRequest(BaseModel):
    facilities: list[ScanRequest] = Field(..., max_length=50)


class SettingsUpdate(BaseModel):
    openai_api_key: Optional[str] = None
    serper_api_key: Optional[str] = None
    validation_model: Optional[str] = None
    boundary_model: Optional[str] = None
    default_zoom: Optional[str] = None
    default_buffer_m: Optional[str] = None
    overview_zoom: Optional[str] = None
    validation_prompt: Optional[str] = None
    boundary_prompt: Optional[str] = None
    verification_prompt: Optional[str] = None
    verification_model: Optional[str] = None
    correction_mode: Optional[str] = None
    verification_correction_prompt: Optional[str] = None
    bbox_validation_enabled: Optional[str] = None
    # Performance (per-scan, dynamic)
    tile_concurrency: Optional[str] = None
    tile_delay_s: Optional[str] = None
    max_image_mb: Optional[str] = None
    duckdb_memory_limit: Optional[str] = None
    duckdb_threads: Optional[str] = None
    # Infrastructure (require restart)
    worker_concurrency: Optional[str] = None
    heavy_phase_concurrency: Optional[str] = None
    browser_concurrency: Optional[str] = None
    stale_scan_timeout_minutes: Optional[str] = None


# --- Responses ---

class ScreenshotResponse(BaseModel):
    id: UUID
    type: str
    filename: str
    url: str
    thumb_url: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    zoom: Optional[int] = None
    file_size_bytes: Optional[int] = None

    model_config = {"from_attributes": True}


class ScanResponse(BaseModel):
    id: UUID
    facility_name: str = ""
    facility_address: Optional[str] = None
    facility_lat: Optional[float] = None
    facility_lng: Optional[float] = None
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
    status: str = "queued"


class SettingsResponse(BaseModel):
    openai_api_key: str = ""
    serper_api_key: str = ""
    validation_model: str = ""
    boundary_model: str = ""
    default_zoom: str = ""
    default_buffer_m: str = ""
    overview_zoom: str = ""
    validation_prompt: str = ""
    boundary_prompt: str = ""
    verification_prompt: str = ""
    verification_model: str = ""
    correction_mode: str = ""
    verification_correction_prompt: str = ""
    bbox_validation_enabled: str = ""
    # Performance (per-scan, dynamic)
    tile_concurrency: str = ""
    tile_delay_s: str = ""
    max_image_mb: str = ""
    duckdb_memory_limit: str = ""
    duckdb_threads: str = ""
    # Infrastructure (require restart)
    worker_concurrency: str = ""
    heavy_phase_concurrency: str = ""
    browser_concurrency: str = ""
    stale_scan_timeout_minutes: str = ""


class ScanStepResponse(BaseModel):
    id: UUID
    step_number: int
    step_name: str
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    input_summary: Optional[dict] = None
    output_summary: Optional[dict] = None
    decision: Optional[str] = None
    ai_model: Optional[str] = None
    ai_prompt: Optional[str] = None
    ai_response_raw: Optional[str] = None
    ai_tokens_prompt: Optional[int] = None
    ai_tokens_completion: Optional[int] = None
    ai_tokens_reasoning: Optional[int] = None
    ai_tokens_total: Optional[int] = None
    tile_grid_cols: Optional[int] = None
    tile_grid_rows: Optional[int] = None

    model_config = {"from_attributes": True}


class ScanTraceResponse(BaseModel):
    scan_id: UUID
    facility_name: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    status: str
    method: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration_ms: Optional[int] = None
    total_ai_tokens: Optional[int] = None
    steps: list[ScanStepResponse] = []


class HealthResponse(BaseModel):
    status: str = "ok"
    queue_size: int = 0
    workers: int = 1
