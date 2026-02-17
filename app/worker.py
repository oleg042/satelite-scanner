"""Background worker pool using asyncio.Queue for concurrent scan processing."""

import asyncio
import logging
from uuid import UUID

from app.config import settings
from app.database import async_session
from app.scanner.pipeline import run_pipeline

logger = logging.getLogger(__name__)

# Global scan queue
scan_queue: asyncio.Queue = asyncio.Queue()


async def enqueue_scan(scan_id: UUID, facility_name: str, lat: float, lng: float):
    """Add a scan job to the queue."""
    await scan_queue.put((scan_id, facility_name, lat, lng))
    logger.info("Enqueued scan %s (%s) — queue size: %d", scan_id, facility_name, scan_queue.qsize())


async def _worker(worker_id: int):
    """Single worker coroutine — pulls jobs from the shared queue."""
    logger.info("Worker %d ready", worker_id)
    while True:
        scan_id, facility_name, lat, lng = await scan_queue.get()
        if lat is None or lng is None:
            logger.warning("[W%d] Skipping scan %s — no coordinates", worker_id, scan_id)
            scan_queue.task_done()
            continue
        logger.info("[W%d] Processing scan %s (%s)...", worker_id, scan_id, facility_name)
        try:
            async with async_session() as db:
                await run_pipeline(scan_id, facility_name, lat, lng, db)
        except Exception:
            logger.exception("[W%d] Worker error for scan %s", worker_id, scan_id)
        finally:
            scan_queue.task_done()


async def worker_pool():
    """Spawn N concurrent workers pulling from the same queue."""
    n = settings.worker_concurrency
    logger.info("Starting worker pool with %d concurrent workers", n)
    tasks = [asyncio.create_task(_worker(i)) for i in range(n)]
    await asyncio.gather(*tasks)
