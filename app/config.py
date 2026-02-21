"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://localhost/satellite_scanner"
    volume_path: str = "./data"
    openai_api_key: str = ""
    serper_api_key: str = ""
    default_zoom: int = 20
    default_buffer_m: int = 50
    overview_zoom: int = 19
    tile_delay_s: float = 0.05
    search_radius_m: int = 200
    fallback_radius_m: int = 75
    overview_radius_m: int = 200

    # ── Concurrency & Memory Limits ──────────────────────────
    tile_concurrency: int = 10
    worker_concurrency: int = 3
    stale_scan_timeout_minutes: int = 30
    heavy_phase_concurrency: int = 2    # max scans in tile+vision phase simultaneously
    max_image_mb: int = 128             # PIL image size before chunked stitching kicks in
    browser_concurrency: int = 2        # max concurrent Playwright browser contexts

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
