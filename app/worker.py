"""Background worker pool using asyncio.Queue for concurrent scan processing."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import select, update

from app.config import settings
from app.database import async_session
from app.models import Scan, ScanStatus
from app.scanner.pipeline import run_pipeline

logger = logging.getLogger(__name__)

# Global scan queue
scan_queue: asyncio.Queue = asyncio.Queue()

# Track running pipeline tasks: scan_id → asyncio.Task
running_tasks: dict[UUID, asyncio.Task] = {}

RUNNING_STATUSES = [
    ScanStatus.running_osm,
    ScanStatus.running_validate,
    ScanStatus.running_vision,
    ScanStatus.running_msft,
    ScanStatus.running_overture,
    ScanStatus.running_tiling,
    ScanStatus.running_bin_detection,
]


async def enqueue_scan(scan_id: UUID):
    """Add a scan job to the queue."""
    await scan_queue.put(scan_id)
    logger.info("Enqueued scan %s — queue size: %d", scan_id, scan_queue.qsize())


async def recover_stuck_scans():
    """On startup, mark scans stuck in running_* or queued states as failed.

    These were interrupted by a crash/restart. Instead of re-queuing them
    (which can cause OOM crash loops), mark them failed so the user can
    manually retry from the UI when ready.
    """
    async with async_session() as db:
        result = await db.execute(
            select(Scan).where(
                Scan.status.in_(RUNNING_STATUSES + [ScanStatus.queued])
            )
        )
        stuck_scans = result.scalars().all()

        if not stuck_scans:
            logger.info("No stuck scans found on startup")
            return 0

        for scan in stuck_scans:
            old_status = scan.status
            scan.status = ScanStatus.failed
            scan.error_message = f"Interrupted by server restart (was {old_status.value})"
            logger.info(
                "Marking stuck scan %s as failed (was %s)",
                scan.id, old_status.value,
            )

        await db.commit()

        logger.info("Marked %d stuck scans as failed on startup", len(stuck_scans))
        return len(stuck_scans)


async def stale_scan_watchdog(timeout_minutes: int | None = None):
    """Periodically check for scans stuck in running_* states too long.

    Catches workers hung on external API calls (OpenAI, OSM, tile servers).
    """
    if timeout_minutes is None:
        timeout_minutes = settings.stale_scan_timeout_minutes
    check_interval = max(timeout_minutes * 60 // 2, 60)

    logger.info(
        "Stale scan watchdog started (timeout=%dm, check every %ds)",
        timeout_minutes, check_interval,
    )

    while True:
        await asyncio.sleep(check_interval)

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)

            async with async_session() as db:
                result = await db.execute(
                    select(Scan).where(
                        Scan.status.in_(RUNNING_STATUSES),
                        Scan.started_at < cutoff,
                    )
                )
                stale_scans = result.scalars().all()

                if not stale_scans:
                    continue

                for scan in stale_scans:
                    old_status = scan.status
                    scan.status = ScanStatus.queued
                    scan.error_message = None
                    logger.warning(
                        "Stale scan detected: %s (was %s, started %s) → re-queued",
                        scan.id, old_status.value, scan.started_at,
                    )

                await db.commit()

                for scan in stale_scans:
                    await enqueue_scan(scan.id)

                logger.info("Watchdog recovered %d stale scans", len(stale_scans))

        except Exception:
            logger.exception("Stale scan watchdog error")


async def _worker(worker_id: int):
    """Single worker coroutine — pulls jobs from the shared queue."""
    logger.info("Worker %d ready", worker_id)
    while True:
        scan_id = await scan_queue.get()
        logger.info("[W%d] Processing scan %s...", worker_id, scan_id)
        try:
            running_tasks[scan_id] = asyncio.current_task()
            async with async_session() as db:
                await run_pipeline(scan_id, db)
        except asyncio.CancelledError:
            logger.info("[W%d] Scan %s was cancelled", worker_id, scan_id)
            try:
                async with async_session() as db:
                    scan = await db.get(Scan, scan_id)
                    if scan and scan.status not in (ScanStatus.completed, ScanStatus.failed):
                        scan.status = ScanStatus.failed
                        scan.error_message = "Cancelled by user"
                        scan.completed_at = datetime.now(timezone.utc)
                        await db.commit()
            except Exception:
                logger.exception("[W%d] Failed to mark cancelled scan %s", worker_id, scan_id)
        except Exception:
            logger.exception("[W%d] Worker error for scan %s", worker_id, scan_id)
        finally:
            running_tasks.pop(scan_id, None)
            scan_queue.task_done()


async def cancel_scan(scan_id: UUID) -> bool:
    """Cancel a single running scan. Returns True if it was running."""
    task = running_tasks.get(scan_id)
    if task and not task.done():
        task.cancel()
        return True
    return False


async def cancel_all() -> dict:
    """Cancel all running scans and drain the queue back to pending."""
    cancelled_ids = []

    # 1. Cancel all running tasks
    for scan_id, task in list(running_tasks.items()):
        if not task.done():
            task.cancel()
            cancelled_ids.append(scan_id)

    # 2. Drain the queue and collect queued scan IDs
    drained_ids = []
    while not scan_queue.empty():
        try:
            scan_id = scan_queue.get_nowait()
            drained_ids.append(scan_id)
            scan_queue.task_done()
        except asyncio.QueueEmpty:
            break

    # 3. Reset drained scans to pending in DB
    if drained_ids:
        async with async_session() as db:
            await db.execute(
                update(Scan)
                .where(Scan.id.in_(drained_ids))
                .values(status=ScanStatus.pending, error_message=None)
            )
            await db.commit()
        logger.info("Reset %d queued scans to pending", len(drained_ids))

    return {"cancelled": len(cancelled_ids), "requeued_to_pending": len(drained_ids)}


async def worker_pool(n: int | None = None):
    """Spawn N concurrent workers pulling from the same queue."""
    if n is None:
        n = settings.worker_concurrency
    logger.info("Starting worker pool with %d concurrent workers", n)
    tasks = [asyncio.create_task(_worker(i)) for i in range(n)]
    await asyncio.gather(*tasks)
