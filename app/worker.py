"""Background worker using asyncio.Queue for sequential scan processing."""

import asyncio
import logging
from uuid import UUID

from app.database import async_session
from app.scanner.pipeline import run_pipeline

logger = logging.getLogger(__name__)

# Global scan queue
scan_queue: asyncio.Queue = asyncio.Queue()


async def enqueue_scan(scan_id: UUID, facility_name: str, lat: float, lng: float):
    """Add a scan job to the queue."""
    await scan_queue.put((scan_id, facility_name, lat, lng))
    logger.info("Enqueued scan %s (%s) — queue size: %d", scan_id, facility_name, scan_queue.qsize())


async def worker_loop():
    """Process scans sequentially from the queue (rate limits require this)."""
    logger.info("Background worker started")
    while True:
        scan_id, facility_name, lat, lng = await scan_queue.get()
        logger.info("Processing scan %s (%s)...", scan_id, facility_name)
        try:
            async with async_session() as db:
                # Run pipeline in thread for CPU-bound Pillow operations
                await asyncio.to_thread(
                    _run_sync_wrapper, scan_id, facility_name, lat, lng, db
                )
        except Exception:
            logger.exception("Worker error for scan %s", scan_id)
        finally:
            scan_queue.task_done()


def _run_sync_wrapper(scan_id, facility_name, lat, lng, db):
    """Sync wrapper to run async pipeline from thread."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(run_pipeline(scan_id, facility_name, lat, lng, db))
    finally:
        loop.close()
