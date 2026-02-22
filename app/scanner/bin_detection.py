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
import time
from dataclasses import dataclass, field

from openai import OpenAI
from PIL import Image

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
    api_key: str,
    model: str,
    prompt_template: str,
    chunk_label: str = "",
) -> dict:
    """Run AI bin detection on a single image chunk.

    Returns dict with parsed AI response + metadata, or error info on failure.
    """
    prompt = prompt_template.replace("{image_width_m}", str(round(width_m, 1)))
    prompt = prompt.replace("{image_height_m}", str(round(height_m, 1)))

    max_retries = 3
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            result = await asyncio.to_thread(
                _call_openai_vision,
                chunk_image_b64, prompt, api_key, model,
            )
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
            api_key, model, prompt_template, label,
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
        chunk_confidence = min(bin_confs) if bin_confs else 0

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

                # Convert chunk-relative bbox to full-image-relative
                bbox_rel = bin_item.get("bbox_relative", {})
                full_x = (chunk["px_x"] + bbox_rel.get("x", 0) * chunk["px_w"]) / img_w
                full_y = (chunk["px_y"] + bbox_rel.get("y", 0) * chunk["px_h"]) / img_h
                full_w = (bbox_rel.get("w", 0) * chunk["px_w"]) / img_w
                full_h = (bbox_rel.get("h", 0) * chunk["px_h"]) / img_h

                converted_bin = {**bin_item}
                converted_bin["bbox_relative"] = {
                    "x": round(full_x, 6),
                    "y": round(full_y, 6),
                    "w": round(full_w, 6),
                    "h": round(full_h, 6),
                }
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
