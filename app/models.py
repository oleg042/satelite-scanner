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
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# --- Enums ---

class ScanStatus(str, enum.Enum):
    queued = "queued"
    running_osm = "running_osm"
    running_validate = "running_validate"
    running_vision = "running_vision"
    running_tiling = "running_tiling"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"


class ScanMethod(str, enum.Enum):
    osm_building = "osm_building"
    osm_landuse = "osm_landuse"
    ai_vision = "ai_vision"
    fallback_radius = "fallback_radius"
    skipped = "skipped"


class ScreenshotType(str, enum.Enum):
    overview = "overview"
    ai_overview = "ai_overview"
    final = "final"


# --- Models ---

class Facility(Base):
    __tablename__ = "facilities"
    __table_args__ = (UniqueConstraint("lat", "lng", name="uq_facility_coords"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    lat = Column(Float, nullable=False)
    lng = Column(Float, nullable=False)
    address = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    scans = relationship("Scan", back_populates="facility", lazy="selectin")


class Scan(Base):
    __tablename__ = "scans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    facility_id = Column(UUID(as_uuid=True), ForeignKey("facilities.id"), nullable=False)
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

    facility = relationship("Facility", back_populates="scans")
    screenshots = relationship("Screenshot", back_populates="scan", lazy="selectin")


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


class Setting(Base):
    __tablename__ = "settings"

    key = Column(Text, primary_key=True)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
