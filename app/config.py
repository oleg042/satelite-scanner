"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://localhost/satellite_scanner"
    volume_path: str = "./data"
    openai_api_key: str = ""
    default_zoom: int = 20
    default_buffer_m: int = 50
    overview_zoom: int = 19
    tile_delay_s: float = 0.05
    search_radius_m: int = 200
    fallback_radius_m: int = 75
    overview_radius_m: int = 400

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
