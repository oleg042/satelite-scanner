"""Disk-based tile cache for Railway Volume storage."""

import os
import shutil
from typing import Optional

from app.config import settings


class TileCache:
    """Disk-based tile cache stored in {volume_path}/tile_cache/{zoom}/{x}_{y}.png."""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            base_dir = os.path.join(settings.volume_path, "tile_cache")
        self.base_dir = base_dir

    def _path(self, x: int, y: int, zoom: int) -> str:
        return os.path.join(self.base_dir, str(zoom), f"{x}_{y}.png")

    def get(self, x: int, y: int, zoom: int) -> Optional[bytes]:
        path = self._path(x, y, zoom)
        if os.path.isfile(path):
            try:
                with open(path, "rb") as f:
                    return f.read()
            except OSError:
                return None
        return None

    def put(self, x: int, y: int, zoom: int, data: bytes) -> None:
        path = self._path(x, y, zoom)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "wb") as f:
                f.write(data)
        except OSError:
            pass

    def clear(self) -> None:
        """Remove the entire cache directory tree."""
        shutil.rmtree(self.base_dir, ignore_errors=True)
