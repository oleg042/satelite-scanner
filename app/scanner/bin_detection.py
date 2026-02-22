"""Scrap bin detection via image chunking + parallel AI analysis.

Splits a final satellite image into <=100m grid chunks, runs AI detection
on each chunk in parallel, then aggregates results with coordinate conversion
from chunk-relative to full-image-relative space.
"""

import asyncio
import base64
import io
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from openai import OpenAI
from PIL import Image
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Scan, Screenshot, ScreenshotType, ScanStep
from app.scanner.utils import save_image, record_screenshot, record_step
from app.scanner.vision import _parse_json_response

logger = logging.getLogger(__name__)


@dataclass
class BinDetectionResult:
    bin_present: bool
    total_bins: int
    filled_or_partial_count: int
    empty_or_unclear_count: int
    overall_confidence: int  # 0-100
    bins: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    chunks_total: int = 0
    chunks_with_bins: int = 0
    chunks_failed: int = 0
    chunk_results: list[dict] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    raw_responses: list[str] = field(default_factory=list)


def calculate_chunk_grid(
    image_width_px: int,
    image_height_px: int,
    bbox_width_m: float,
    bbox_height_m: float,
    max_chunk_m: float = 100.0,
) -> list[dict]:
    """Calculate grid of chunks for an image based on ground coverage.

    Returns list of chunk descriptors with pixel offsets and ground dimensions.
    """
    cols = max(1, math.ceil(bbox_width_m / max_chunk_m))
    rows = max(1, math.ceil(bbox_height_m / max_chunk_m))

    px_per_m_x = image_width_px / bbox_width_m
    px_per_m_y = image_height_px / bbox_height_m

    chunks = []
    for row in range(rows):
        for col in range(cols):
            # Ground coverage for this chunk
            chunk_width_m = min(max_chunk_m, bbox_width_m - col * max_chunk_m)
            chunk_height_m = min(max_chunk_m, bbox_height_m - row * max_chunk_m)

            # Pixel coordinates
            px_x = round(col * max_chunk_m * px_per_m_x)
            px_y = round(row * max_chunk_m * px_per_m_y)
            px_w = round(chunk_width_m * px_per_m_x)
            px_h = round(chunk_height_m * px_per_m_y)

            # Clamp to image bounds
            px_w = min(px_w, image_width_px - px_x)
            px_h = min(px_h, image_height_px - px_y)

            if px_w <= 0 or px_h <= 0:
                continue

            chunks.append({
                "col": col,
                "row": row,
                "px_x": px_x,
                "px_y": px_y,
                "px_w": px_w,
                "px_h": px_h,
                "width_m": round(chunk_width_m, 2),
                "height_m": round(chunk_height_m, 2),
            })

    return chunks


def _crop_to_base64(image: Image.Image, chunk: dict) -> str:
    """Crop a chunk from the image and return as base64 PNG string."""
    box = (chunk["px_x"], chunk["px_y"],
           chunk["px_x"] + chunk["px_w"],
           chunk["px_y"] + chunk["px_h"])
    cropped = image.crop(box)
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    cropped.close()
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    return b64


async def detect_bins_in_chunk(
    chunk_image_b64: str,
    width_m: float,
    height_m: float,
    width_px: int,
    height_px: int,
    api_key: str,
    model: str,
    prompt_template: str,
    chunk_label: str = "",
    include_reasoning: bool = True,
) -> dict:
    """Run AI bin detection on a single image chunk.

    Returns dict with parsed AI response + metadata, or error info on failure.
    """
    # Pre-calculate scale and expected bin dimensions in pixels
    meters_per_pixel = width_m / width_px if width_px > 0 else 0
    bin_width_px_est = round(2.44 / meters_per_pixel) if meters_per_pixel > 0 else 0
    bin_len_min_px = round(4.0 / meters_per_pixel) if meters_per_pixel > 0 else 0
    bin_len_max_px = round(8.0 / meters_per_pixel) if meters_per_pixel > 0 else 0

    prompt = prompt_template.replace("{image_width_m}", str(round(width_m, 1)))
    prompt = prompt.replace("{image_height_m}", str(round(height_m, 1)))
    prompt = prompt.replace("{image_width_px}", str(width_px))
    prompt = prompt.replace("{image_height_px}", str(height_px))
    prompt = prompt.replace("{meters_per_pixel}", str(round(meters_per_pixel, 4)))
    prompt = prompt.replace("{bin_width_px}", str(bin_width_px_est))
    prompt = prompt.replace("{bin_length_min_px}", str(bin_len_min_px))
    prompt = prompt.replace("{bin_length_max_px}", str(bin_len_max_px))

    if not include_reasoning:
        import re as _re
        prompt = _re.sub(r',?\s*"evidence"\s*:\s*"[^"]*"', '', prompt)

    max_retries = 3
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            result = await asyncio.to_thread(
                _call_openai_vision,
                chunk_image_b64, prompt, api_key, model,
            )
            if not include_reasoning:
                for bin_item in result["parsed"].get("bins", []):
                    bin_item.pop("evidence", None)
            logger.info("Chunk %s: bin_present=%s, bins=%d",
                        chunk_label, result["parsed"].get("bin_present"),
                        result["parsed"].get("total_bins", 0))
            return {
                "status": "success",
                "parsed": result["parsed"],
                "raw_response": result["raw"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
            }
        except Exception as e:
            last_error = e
            # Check for rate limit (429)
            is_rate_limit = False
            retry_after = None
            try:
                import openai
                if isinstance(e, openai.RateLimitError):
                    is_rate_limit = True
                    # Try to extract Retry-After from headers
                    if hasattr(e, "response") and e.response:
                        retry_after = e.response.headers.get("Retry-After")
            except ImportError:
                pass

            if is_rate_limit and attempt < max_retries:
                if retry_after:
                    wait = min(float(retry_after), 30.0)
                else:
                    wait = (2 ** attempt)  # 1, 2, 4
                logger.warning("Chunk %s rate limited (attempt %d/%d), waiting %.1fs",
                               chunk_label, attempt + 1, max_retries, wait)
                await asyncio.sleep(wait)
                continue

            if attempt < max_retries and not _is_fatal_error(e):
                wait = (2 ** attempt)
                logger.warning("Chunk %s error (attempt %d/%d): %s, retrying in %.1fs",
                               chunk_label, attempt + 1, max_retries, e, wait)
                await asyncio.sleep(wait)
                continue

            break

    logger.error("Chunk %s failed after %d attempts: %s", chunk_label, max_retries + 1, last_error)
    return {
        "status": "failed",
        "error": str(last_error),
        "parsed": None,
        "raw_response": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _is_fatal_error(exc: Exception) -> bool:
    """Check if an error is fatal and shouldn't be retried."""
    try:
        import openai
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return True
        if isinstance(exc, openai.RateLimitError):
            code = getattr(exc, "code", None) or ""
            return "insufficient_quota" in str(code)
    except ImportError:
        pass
    return False


def _call_openai_vision(image_b64: str, prompt: str, api_key: str, model: str) -> dict:
    """Synchronous OpenAI vision call for a single chunk."""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "high"},
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )
    text = response.choices[0].message.content.strip()
    parsed = _parse_json_response(text)

    return {
        "raw": text,
        "parsed": parsed,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }


async def run_bin_detection(
    image_path: str,
    bbox_width_m: float,
    bbox_height_m: float,
    api_key: str,
    model: str,
    prompt_template: str,
    max_chunk_m: float = 100.0,
    min_confidence: int = 50,
    include_reasoning: bool = True,
) -> BinDetectionResult:
    """Run bin detection on a final satellite image.

    1. Calculate chunk grid
    2. Crop + base64 encode each chunk
    3. Run AI on all chunks in parallel
    4. Aggregate results with coordinate conversion
    """
    img = Image.open(image_path)
    img_w, img_h = img.size

    # Calculate grid
    chunks = calculate_chunk_grid(img_w, img_h, bbox_width_m, bbox_height_m, max_chunk_m)
    logger.info("Bin detection: %dx%dpx image, %.0fx%.0fm, %d chunks (max %.0fm)",
                img_w, img_h, bbox_width_m, bbox_height_m, len(chunks), max_chunk_m)

    if not chunks:
        img.close()
        return BinDetectionResult(
            bin_present=False, total_bins=0,
            filled_or_partial_count=0, empty_or_unclear_count=0,
            overall_confidence=0, chunks_total=0,
        )

    # Prepare all chunks: crop + base64 (release PIL crops immediately)
    chunk_data = []
    for chunk in chunks:
        b64 = _crop_to_base64(img, chunk)
        chunk_data.append((chunk, b64))

    img.close()

    # Run all chunks in parallel
    tasks = []
    for chunk, b64 in chunk_data:
        label = f"c{chunk['col']}r{chunk['row']}"
        tasks.append(detect_bins_in_chunk(
            b64, chunk["width_m"], chunk["height_m"],
            chunk["px_w"], chunk["px_h"],
            api_key, model, prompt_template, label,
            include_reasoning=include_reasoning,
        ))

    results = await asyncio.gather(*tasks)

    # Aggregate
    all_bins = []
    all_notes = []
    total_bins = 0
    filled_count = 0
    empty_count = 0
    chunks_with_bins = 0
    chunks_failed = 0
    confidences = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    raw_responses = []
    chunk_results = []

    for (chunk, _b64), result in zip(chunk_data, results):
        total_prompt_tokens += result.get("prompt_tokens", 0)
        total_completion_tokens += result.get("completion_tokens", 0)
        total_tokens += result.get("total_tokens", 0)

        if result.get("raw_response"):
            raw_responses.append(result["raw_response"])

        if result["status"] == "failed":
            chunks_failed += 1
            chunk_results.append({
                "col": chunk["col"], "row": chunk["row"],
                "status": "failed", "error": result.get("error"),
                "raw_response": None,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            })
            continue

        parsed = result["parsed"]
        chunk_bin_present = parsed.get("bin_present", False)
        chunk_bins = parsed.get("bins", [])
        # Derive chunk confidence from per-bin confidences (model has no top-level field)
        bin_confs = [b.get("confidence", 0) for b in chunk_bins if b.get("confidence")]
        chunk_confidence = max(bin_confs) if bin_confs else 0

        chunk_result = {
            "col": chunk["col"], "row": chunk["row"],
            "status": "success",
            "bin_present": chunk_bin_present,
            "total_bins": parsed.get("total_bins", 0),
            "filled_or_partial_count": parsed.get("filled_or_partial_count", 0),
            "empty_or_unclear_count": parsed.get("empty_or_unclear_count", 0),
            "overall_confidence": chunk_confidence,
            "bins": chunk_bins,
            "reasoning": parsed.get("reasoning", ""),
            "raw_response": result.get("raw_response", ""),
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
        }
        chunk_results.append(chunk_result)

        if chunk_bin_present and chunk_bins:
            chunks_with_bins += 1

            for bin_item in chunk_bins:
                bin_conf = bin_item.get("confidence", 0)
                # Only include bins above the confidence threshold in aggregates
                if bin_conf < min_confidence:
                    continue

                confidences.append(bin_conf)

                converted_bin = {**bin_item}
                converted_bin["source_chunk"] = {"col": chunk["col"], "row": chunk["row"]}
                all_bins.append(converted_bin)

                if bin_item.get("fill_status") == "filled_or_partial":
                    filled_count += 1
                else:
                    empty_count += 1

                total_bins += 1

        # Collect reasoning
        reasoning = parsed.get("reasoning", "")
        if isinstance(reasoning, list):
            reasoning = " ".join(r for r in reasoning if r and r.strip())
        if isinstance(reasoning, str) and reasoning.strip():
            all_notes.append(f"[c{chunk['col']}r{chunk['row']}] {reasoning}")

    # Overall confidence: average of per-bin confidences across all chunks
    if confidences:
        overall_confidence = round(sum(confidences) / len(confidences))
    else:
        overall_confidence = 0

    # Add failure notes
    if chunks_failed > 0:
        all_notes.append(f"{chunks_failed} chunk(s) failed — results may be partial")

    return BinDetectionResult(
        bin_present=total_bins > 0,
        total_bins=total_bins,
        filled_or_partial_count=filled_count,
        empty_or_unclear_count=empty_count,
        overall_confidence=overall_confidence,
        bins=all_bins,
        notes=all_notes,
        chunks_total=len(chunks),
        chunks_with_bins=chunks_with_bins,
        chunks_failed=chunks_failed,
        chunk_results=chunk_results,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
        raw_responses=raw_responses,
    )


async def execute_bin_detection(
    scan: Scan,
    db: AsyncSession,
    final_image_path: str,
    api_key: str,
    model: str,
    prompt: str,
    max_chunk_m: float,
    min_confidence: int,
    step_num_start: int,
    volume_path: str,
    clean_old: bool = True,
    delete_final_image: bool = False,
    resize_final_image: bool = False,
    include_reasoning: bool = True,
) -> dict:
    """Full bin detection workflow: chunk, detect, save screenshots, record steps.

    Used by both the API "Detect Bins" button and the full pipeline scan.

    Returns a summary dict with detection results.
    """
    step_num = step_num_start

    # Optionally clean old bin detection data (API re-run path)
    if clean_old:
        old_bin_ss = await db.execute(
            select(Screenshot).where(
                Screenshot.scan_id == scan.id,
                Screenshot.type == ScreenshotType.bin_chunk,
            )
        )
        for old_ss in old_bin_ss.scalars().all():
            old_path = os.path.join(volume_path, old_ss.file_path)
            try:
                os.unlink(old_path)
            except OSError:
                pass
        await db.execute(
            delete(Screenshot).where(
                Screenshot.scan_id == scan.id,
                Screenshot.type == ScreenshotType.bin_chunk,
            )
        )
        await db.execute(
            delete(ScanStep).where(
                ScanStep.scan_id == scan.id,
                ScanStep.step_name.in_(["bin_detection", "image_chunking"]),
            )
        )
        await db.commit()

    # Step A: Image Chunking
    chunk_grid = calculate_chunk_grid(
        scan.image_width, scan.image_height,
        scan.bbox_width_m, scan.bbox_height_m,
        max_chunk_m,
    )
    cols = max(c["col"] for c in chunk_grid) + 1 if chunk_grid else 0
    rows = max(c["row"] for c in chunk_grid) + 1 if chunk_grid else 0

    await record_step(
        db, scan.id, step_num, "image_chunking",
        status="completed",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        duration_ms=0,
        input_summary=json.dumps({
            "image_px": f"{scan.image_width}x{scan.image_height}",
            "bbox_m": f"{round(scan.bbox_width_m, 1)}x{round(scan.bbox_height_m, 1)}",
            "max_chunk_m": max_chunk_m,
        }),
        output_summary=json.dumps({
            "grid": f"{cols}x{rows}",
            "chunk_count": len(chunk_grid),
            "chunks": chunk_grid,
        }),
        decision=(
            f"Split {round(scan.bbox_width_m, 1)}m x {round(scan.bbox_height_m, 1)}m image into "
            f"{len(chunk_grid)} chunks ({cols}x{rows} grid, max {max_chunk_m}m per chunk)"
        ),
    )

    # Step B: Bin Detection
    step_num += 1
    bin_step_started = datetime.now(timezone.utc)

    bin_result = await run_bin_detection(
        final_image_path,
        scan.bbox_width_m, scan.bbox_height_m,
        api_key, model, prompt, max_chunk_m,
        min_confidence=min_confidence,
        include_reasoning=include_reasoning,
    )

    # Update scan fields
    scan.bin_present = bin_result.bin_present
    scan.bin_count = bin_result.total_bins
    scan.bin_filled_count = bin_result.filled_or_partial_count
    scan.bin_empty_count = bin_result.empty_or_unclear_count
    scan.bin_confidence = bin_result.overall_confidence

    if bin_result.chunks_failed > 0 and bin_result.chunks_failed < bin_result.chunks_total:
        scan.bin_detection_status = "partial"
    elif bin_result.chunks_failed >= bin_result.chunks_total:
        scan.bin_detection_status = "failed"
    else:
        scan.bin_detection_status = "completed"

    # Save chunk images where bins were found (only if confidence meets threshold)
    if bin_result.bin_present:
        final_img = Image.open(final_image_path)
        for cr in bin_result.chunk_results:
            if cr.get("status") == "success" and cr.get("bin_present"):
                chunk_conf = cr.get("overall_confidence", 0)
                if chunk_conf < min_confidence:
                    continue
                matching = [c for c in chunk_grid
                            if c["col"] == cr["col"] and c["row"] == cr["row"]]
                if not matching:
                    continue
                chunk_desc = matching[0]
                box = (chunk_desc["px_x"], chunk_desc["px_y"],
                       chunk_desc["px_x"] + chunk_desc["px_w"],
                       chunk_desc["px_y"] + chunk_desc["px_h"])
                chunk_crop = final_img.crop(box)
                suffix = f"bin_chunk_{cr['col']}_{cr['row']}"
                rel_path_c, filename_c, abs_path_c = save_image(
                    chunk_crop, scan.facility_name, suffix,
                    scan.zoom or 20, scan.id, volume_path,
                )
                await record_screenshot(
                    db, scan.id, ScreenshotType.bin_chunk,
                    filename_c, rel_path_c, abs_path_c,
                    scan.zoom or 20,
                    chunk_crop.width, chunk_crop.height,
                )
                chunk_crop.close()
        final_img.close()

    bin_completed = datetime.now(timezone.utc)
    bin_elapsed = int((bin_completed - bin_step_started).total_seconds() * 1000)

    bin_decision = (
        f"Detected {bin_result.total_bins} bins "
        f"({bin_result.filled_or_partial_count} filled, "
        f"{bin_result.empty_or_unclear_count} empty) "
        f"across {bin_result.chunks_total} chunks. "
        f"Confidence: {bin_result.overall_confidence}%"
    )
    if bin_result.chunks_failed > 0:
        bin_decision += f" ({bin_result.chunks_failed} chunks failed)"

    await record_step(
        db, scan.id, step_num, "bin_detection",
        status="completed",
        started_at=bin_step_started,
        completed_at=bin_completed,
        duration_ms=bin_elapsed,
        input_summary=json.dumps({
            "chunk_count": bin_result.chunks_total,
            "model": model,
            "min_confidence": min_confidence,
            "include_reasoning": include_reasoning,
        }),
        output_summary=json.dumps({
            "bin_present": bin_result.bin_present,
            "total_bins": bin_result.total_bins,
            "filled_or_partial_count": bin_result.filled_or_partial_count,
            "empty_or_unclear_count": bin_result.empty_or_unclear_count,
            "overall_confidence": bin_result.overall_confidence,
            "bins": bin_result.bins,
            "chunks_with_bins": bin_result.chunks_with_bins,
            "chunks_failed": bin_result.chunks_failed,
            "notes": bin_result.notes,
            "chunk_results": bin_result.chunk_results,
        }),
        decision=bin_decision,
        ai_model=model,
        ai_tokens_prompt=bin_result.total_prompt_tokens,
        ai_tokens_completion=bin_result.total_completion_tokens,
        ai_tokens_total=bin_result.total_tokens,
    )
    await db.commit()

    # Optionally delete or resize the final image to save storage
    if delete_final_image:
        try:
            os.unlink(final_image_path)
            logger.info("Deleted final image for scan %s: %s", scan.id, final_image_path)
        except OSError:
            pass
        # Remove the screenshot DB record so the UI doesn't show a broken image
        await db.execute(
            delete(Screenshot).where(
                Screenshot.scan_id == scan.id,
                Screenshot.type == ScreenshotType.final,
            )
        )
        await db.commit()
    elif resize_final_image:
        try:
            img = Image.open(final_image_path)
            new_w = img.width // 2
            new_h = img.height // 2
            resized = img.resize((new_w, new_h), Image.LANCZOS)
            img.close()
            resized.save(final_image_path)
            resized.close()
            # Update the screenshot DB record with new dimensions and file size
            new_size = os.path.getsize(final_image_path)
            result = await db.execute(
                select(Screenshot).where(
                    Screenshot.scan_id == scan.id,
                    Screenshot.type == ScreenshotType.final,
                )
            )
            ss = result.scalar_one_or_none()
            if ss:
                ss.width = new_w
                ss.height = new_h
                ss.file_size_bytes = new_size
                await db.commit()
            logger.info("Resized final image for scan %s to %dx%d (%dKB)",
                        scan.id, new_w, new_h, new_size // 1024)
        except Exception as e:
            logger.warning("Failed to resize final image for scan %s: %s", scan.id, e)

    return {
        "scan_id": str(scan.id),
        "status": scan.bin_detection_status,
        "bin_present": bin_result.bin_present,
        "total_bins": bin_result.total_bins,
        "filled_or_partial_count": bin_result.filled_or_partial_count,
        "empty_or_unclear_count": bin_result.empty_or_unclear_count,
        "overall_confidence": bin_result.overall_confidence,
        "chunks_total": bin_result.chunks_total,
        "chunks_with_bins": bin_result.chunks_with_bins,
        "chunks_failed": bin_result.chunks_failed,
    }
