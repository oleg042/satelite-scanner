"""SQLAlchemy ORM models."""

import enum
import uuid

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# --- Enums ---

class ScanStatus(str, enum.Enum):
    pending = "pending"
    queued = "queued"
    running_osm = "running_osm"
    running_validate = "running_validate"
    running_vision = "running_vision"
    running_msft = "running_msft"
    running_tiling = "running_tiling"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"


class ScanMethod(str, enum.Enum):
    osm_building = "osm_building"
    osm_landuse = "osm_landuse"
    ai_vision = "ai_vision"
    msft_buildings = "msft_buildings"
    skipped = "skipped"


class ScreenshotType(str, enum.Enum):
    overview = "overview"
    ai_overview = "ai_overview"
    msft_overlay = "msft_overlay"
    final = "final"


# --- Models ---

class Scan(Base):
    __tablename__ = "scans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    facility_name = Column(Text, nullable=False, default="Unknown")
    facility_address = Column(Text, nullable=True)
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    status = Column(Enum(ScanStatus, name="scan_status"), default=ScanStatus.queued)
    method = Column(Enum(ScanMethod, name="scan_method"), nullable=True)

    # Config inputs
    zoom = Column(Integer, nullable=True)
    buffer_m = Column(Integer, nullable=True)

    # OSM results
    osm_building_count = Column(Integer, nullable=True)
    osm_building_id = Column(BigInteger, nullable=True)

    # Bounding box
    bbox_min_lat = Column(Float, nullable=True)
    bbox_min_lng = Column(Float, nullable=True)
    bbox_max_lat = Column(Float, nullable=True)
    bbox_max_lng = Column(Float, nullable=True)
    bbox_width_m = Column(Float, nullable=True)
    bbox_height_m = Column(Float, nullable=True)

    # AI results
    ai_confidence = Column(Text, nullable=True)
    ai_facility_type = Column(Text, nullable=True)
    ai_building_count = Column(Integer, nullable=True)
    ai_notes = Column(Text, nullable=True)
    ai_validated = Column(Boolean, nullable=True)

    # Tile results
    tile_count = Column(Integer, nullable=True)
    tiles_downloaded = Column(Integer, nullable=True)
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)

    # Error / skip
    error_message = Column(Text, nullable=True)
    skip_reason = Column(Text, nullable=True)

    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    osm_duration_ms = Column(Integer, nullable=True)
    ai_duration_ms = Column(Integer, nullable=True)
    tile_duration_ms = Column(Integer, nullable=True)

    screenshots = relationship("Screenshot", back_populates="scan", lazy="selectin")
    steps = relationship("ScanStep", back_populates="scan", lazy="selectin", order_by="ScanStep.step_number")


class Screenshot(Base):
    __tablename__ = "screenshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id = Column(UUID(as_uuid=True), ForeignKey("scans.id"), nullable=False)
    type = Column(Enum(ScreenshotType, name="screenshot_type"), nullable=False)
    filename = Column(Text, nullable=False)
    file_path = Column(Text, nullable=False)
    file_size_bytes = Column(BigInteger, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    zoom = Column(Integer, nullable=True)

    scan = relationship("Scan", back_populates="screenshots")


class ScanStep(Base):
    __tablename__ = "scan_steps"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id = Column(UUID(as_uuid=True), ForeignKey("scans.id"), nullable=False)
    step_number = Column(Integer, nullable=False)
    step_name = Column(Text, nullable=False)
    status = Column(Text, nullable=False, default="started")
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Integer, nullable=True)

    # What went in / came out
    input_summary = Column(Text, nullable=True)
    output_summary = Column(Text, nullable=True)
    decision = Column(Text, nullable=True)

    # AI-specific
    ai_model = Column(Text, nullable=True)
    ai_prompt = Column(Text, nullable=True)
    ai_response_raw = Column(Text, nullable=True)
    ai_tokens_prompt = Column(Integer, nullable=True)
    ai_tokens_completion = Column(Integer, nullable=True)
    ai_tokens_reasoning = Column(Integer, nullable=True)
    ai_tokens_total = Column(Integer, nullable=True)

    # Tile-specific
    tile_grid_cols = Column(Integer, nullable=True)
    tile_grid_rows = Column(Integer, nullable=True)

    scan = relationship("Scan", back_populates="steps")


class Setting(Base):
    __tablename__ = "settings"

    key = Column(Text, primary_key=True)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
