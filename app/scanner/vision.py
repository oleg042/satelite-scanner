"""OpenAI GPT vision API calls for validation and boundary detection."""

import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI
from PIL import Image

logger = logging.getLogger(__name__)

# Default prompts — stored in DB settings and overridable via API
DEFAULT_VALIDATION_PROMPT = """You are analyzing a satellite image of an industrial facility.

The facility "{name}" is located at ({lat}, {lng}). An OSM bounding box was used to capture
this overview image. The image covers approximately {width_m:.0f}m x {height_m:.0f}m.

Look at the image and determine:
1. Is the entire facility visible and not cut off at any edge?
2. Does the bounding box appear to fully contain the facility's operational footprint?

Respond with ONLY a JSON object:
{{
  "approved": true/false,
  "reason": "brief explanation",
  "facility_type": "what kind of facility this appears to be",
  "notes": "any observations"
}}"""

DEFAULT_BOUNDARY_PROMPT = ""  # Loaded from prompts/facility_boundary_identification.txt


def _load_default_boundary_prompt() -> str:
    """Load the boundary prompt from file."""
    prompt_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "prompts",
        "facility_boundary_identification.txt",
    )
    try:
        with open(prompt_path, "r") as f:
            return f.read()
    except OSError:
        logger.warning("Could not load boundary prompt from %s", prompt_path)
        return ""


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ValidationResult:
    approved: bool
    reason: str
    facility_type: str
    notes: str
    raw_response: str = ""
    prompt_text: str = ""
    usage: Optional[TokenUsage] = None


@dataclass
class BoundaryResult:
    top_y: int
    bottom_y: int
    left_x: int
    right_x: int
    confidence: str
    facility_type: str
    building_count: int
    notes: str
    reasoning: str = ""
    self_check: str = ""
    raw_response: str = ""
    prompt_text: str = ""
    usage: Optional[TokenUsage] = None


def _image_to_base64(image_path: str) -> str:
    """Read image file and return base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _prepare_image_for_ai(image_path: str, max_long_side: int = 2048) -> tuple:
    """Downscale image for AI input to avoid sending pixels OpenAI will discard anyway.

    Returns (base64_str, width, height, scale_factor).
    scale_factor = original / downscaled (1.0 if no scaling needed).
    Multiply AI-returned pixel coords by scale_factor to map back to original image space.
    """
    import io

    img = Image.open(image_path)
    orig_w, orig_h = img.size
    long_side = max(orig_w, orig_h)

    if long_side <= max_long_side:
        img.close()
        return _image_to_base64(image_path), orig_w, orig_h, 1.0

    ratio = max_long_side / long_side
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    img.close()

    buf = io.BytesIO()
    resized.save(buf, format="PNG")
    resized.close()
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    scale = orig_w / new_w
    logger.info("Pre-scaled image for AI: %dx%d → %dx%d (scale=%.3f)", orig_w, orig_h, new_w, new_h, scale)
    return b64, new_w, new_h, scale


def _parse_json_response(text: str) -> dict:
    """Parse JSON from an AI response, handling markdown code blocks."""
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def validate_osm_bbox(
    image_path: str,
    name: str,
    lat: float,
    lng: float,
    width_m: float,
    height_m: float,
    api_key: str,
    model: str,
    prompt_template: str = "",
) -> Optional[ValidationResult]:
    """Send overview image to GPT for OSM bbox validation.

    Returns ValidationResult or None on error.
    """
    if not api_key:
        logger.warning("No API key — skipping validation")
        return None

    if not prompt_template:
        prompt_template = DEFAULT_VALIDATION_PROMPT

    prompt = prompt_template.format(
        name=name, lat=lat, lng=lng, width_m=width_m, height_m=height_m
    )
    img_b64, _, _, _ = _prepare_image_for_ai(image_path)

    logger.info("Calling %s for OSM validation...", model)

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text = response.choices[0].message.content.strip()
        logger.info("Validation response: %s", text)

        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            reasoning_tokens=getattr(response.usage.completion_tokens_details, "reasoning_tokens", 0) or 0,
            total_tokens=response.usage.total_tokens,
        )

        data = _parse_json_response(text)
        return ValidationResult(
            approved=data.get("approved", False),
            reason=data.get("reason", ""),
            facility_type=data.get("facility_type", ""),
            notes=data.get("notes", ""),
            raw_response=text,
            prompt_text=prompt,
            usage=usage,
        )
    except Exception as e:
        logger.error("Validation API error: %s", e)
        raise


def detect_facility_boundary(
    image_path: str,
    name: str,
    lat: float,
    lng: float,
    width_m: float,
    height_m: float,
    api_key: str,
    model: str,
    prompt_template: str = "",
) -> Optional[BoundaryResult]:
    """Send overview image to GPT for facility boundary detection.

    Returns BoundaryResult or None on error.
    """
    if not api_key:
        logger.warning("No API key — skipping boundary detection")
        return None

    # Load boundary prompt
    if not prompt_template:
        prompt_template = _load_default_boundary_prompt()
    if not prompt_template:
        logger.error("No boundary prompt available")
        return None

    img_b64, img_w, img_h, scale = _prepare_image_for_ai(image_path)

    # Add image context to the prompt (dimensions reflect what the AI actually sees)
    context = (
        f"\n\n## IMAGE CONTEXT\n"
        f"- Facility name: {name}\n"
        f"- Center coordinates: ({lat}, {lng})\n"
        f"- Image dimensions: {img_w}px x {img_h}px\n"
        f"- Coverage: approximately {width_m:.0f}m x {height_m:.0f}m\n\n"
        f"Respond with ONLY a JSON object:\n"
        f'{{"top_y": <int>, "bottom_y": <int>, "left_x": <int>, "right_x": <int>, '
        f'"reasoning": "<how you identified the boundary>", '
        f'"self_check": "<edge-by-edge verification>", '
        f'"confidence": "high/medium/low", "facility_type": "<type>", '
        f'"building_count": <int>, "notes": "<observations>"}}'
    )
    full_prompt = prompt_template + context

    logger.info("Calling %s for boundary detection...", model)

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=8192,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                    {"type": "text", "text": full_prompt},
                ],
            }],
        )
        text = response.choices[0].message.content.strip()
        logger.info("Boundary response: %s", text)

        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            reasoning_tokens=getattr(response.usage.completion_tokens_details, "reasoning_tokens", 0) or 0,
            total_tokens=response.usage.total_tokens,
        )

        data = _parse_json_response(text)

        # Scale pixel coords back to original image space
        if scale != 1.0:
            data["top_y"] = round(data["top_y"] * scale)
            data["bottom_y"] = round(data["bottom_y"] * scale)
            data["left_x"] = round(data["left_x"] * scale)
            data["right_x"] = round(data["right_x"] * scale)

        return BoundaryResult(
            top_y=data["top_y"],
            bottom_y=data["bottom_y"],
            left_x=data["left_x"],
            right_x=data["right_x"],
            confidence=data.get("confidence", "medium"),
            facility_type=data.get("facility_type", "unknown"),
            building_count=data.get("building_count", 0),
            notes=data.get("notes", ""),
            reasoning=data.get("reasoning", ""),
            self_check=data.get("self_check", ""),
            raw_response=text,
            prompt_text=full_prompt,
            usage=usage,
        )
    except Exception as e:
        logger.error("Boundary detection API error: %s", e)
        raise


DEFAULT_VERIFICATION_PROMPT = """You are verifying an AI-detected facility boundary on a satellite image.

The facility "{name}" should be centered in this image. A red rectangle has been drawn showing the detected boundary.

Check the following:
1. Is the target facility (the building/campus at or near the image center) FULLY inside the red rectangle, with no parts cut off at the edges?
2. For each edge of the rectangle (top, bottom, left, right), assess whether it closely follows the facility boundary on that side. A small margin (10-20%) is acceptable. Note which specific edges, if any, extend too far and include areas clearly belonging to neighboring properties or unrelated to the facility.
3. Is the rectangle centered roughly on the facility, or is it significantly shifted away from center?

Respond with ONLY a JSON object:
{{
  "approved": true/false,
  "reason": "brief explanation",
  "issues": "summary of problems",
  "edge_feedback": {{
    "top": "ok or description of problem",
    "bottom": "ok or description of problem",
    "left": "ok or description of problem",
    "right": "ok or description of problem"
  }}
}}"""


@dataclass
class VerificationResult:
    approved: bool
    reason: str
    issues: str
    edge_feedback: dict = None
    raw_response: str = ""
    prompt_text: str = ""
    usage: Optional[TokenUsage] = None


def verify_facility_boundary(
    image_path: str,
    name: str,
    boundary: BoundaryResult,
    api_key: str,
    model: str,
    prompt_template: str = "",
) -> Optional[VerificationResult]:
    """Draw detected bbox on overview image and ask a second model to verify it.

    Returns VerificationResult or None on error.
    """
    if not api_key:
        return None

    try:
        img = Image.open(image_path).copy()
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        # Draw red rectangle showing detected boundary (3px thick)
        for offset in range(3):
            draw.rectangle(
                [boundary.left_x - offset, boundary.top_y - offset,
                 boundary.right_x + offset, boundary.bottom_y + offset],
                outline="red",
            )

        # Encode annotated image
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        if not prompt_template:
            prompt_template = DEFAULT_VERIFICATION_PROMPT
        prompt = prompt_template.format(name=name)

        logger.info("Calling %s for boundary verification...", model)
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text = response.choices[0].message.content.strip()
        logger.info("Verification response: %s", text)

        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            reasoning_tokens=getattr(response.usage.completion_tokens_details, "reasoning_tokens", 0) or 0,
            total_tokens=response.usage.total_tokens,
        )

        data = _parse_json_response(text)
        return VerificationResult(
            approved=data.get("approved", False),
            reason=data.get("reason", ""),
            issues=data.get("issues", ""),
            edge_feedback=data.get("edge_feedback"),
            raw_response=text,
            prompt_text=prompt,
            usage=usage,
        )
    except Exception as e:
        logger.error("Verification API error: %s", e)
        raise


DEFAULT_CORRECTION_PROMPT = """You are correcting a facility boundary on a satellite image.
A red rectangle shows the current boundary that needs adjustment.

Current boundary coordinates (pixels):
  top_y={top_y}, bottom_y={bottom_y}, left_x={left_x}, right_x={right_x}
  Image size: {img_w}x{img_h} pixels

The boundary was REJECTED. Per-edge assessment:
- TOP: {edge_top}
- BOTTOM: {edge_bottom}
- LEFT: {edge_left}
- RIGHT: {edge_right}

Instructions:
- For edges marked "too far": Look at the image and pull that edge inward to closely
  follow the facility's actual footprint on that side.
- For edges where a building/area is "cut off": Push that edge outward to fully contain
  the facility on that side.
- For edges marked "ok": Keep the coordinate the same or very close.
- Do NOT add buffer or padding — that is handled automatically in post-processing.

Respond with ONLY a JSON object:
{{"top_y": <int>, "bottom_y": <int>, "left_x": <int>, "right_x": <int>,
  "reasoning": "<what you adjusted and why>", "confidence": "high/medium/low"}}"""


def correct_facility_boundary(
    image_path: str,
    boundary: BoundaryResult,
    verification: VerificationResult,
    api_key: str,
    model: str,
) -> Optional[BoundaryResult]:
    """Adjust an existing boundary using lightweight correction instead of full re-detection.

    Uses the verifier's per-edge feedback to make targeted coordinate adjustments.
    Much cheaper than a full detect_facility_boundary() call.

    Returns a new BoundaryResult with adjusted coordinates, or None on error.
    """
    if not api_key:
        return None

    try:
        # Draw the current boundary as a red rectangle on the image (same as verify)
        img = Image.open(image_path).copy()
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        for offset in range(3):
            draw.rectangle(
                [boundary.left_x - offset, boundary.top_y - offset,
                 boundary.right_x + offset, boundary.bottom_y + offset],
                outline="red",
            )

        # Prepare image (downscale if needed)
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        orig_w, orig_h = img.size
        img.close()

        # Downscale annotated image for AI
        max_long_side = 2048
        long_side = max(orig_w, orig_h)
        if long_side > max_long_side:
            ratio = max_long_side / long_side
            new_w = int(orig_w * ratio)
            new_h = int(orig_h * ratio)
            img2 = Image.open(io.BytesIO(buf.getvalue()))
            resized = img2.resize((new_w, new_h), Image.LANCZOS)
            img2.close()
            buf2 = io.BytesIO()
            resized.save(buf2, format="PNG")
            resized.close()
            img_b64 = base64.b64encode(buf2.getvalue()).decode("utf-8")
            scale = orig_w / new_w
            ai_w, ai_h = new_w, new_h
            # Scale boundary coords down for the prompt
            prompt_top = round(boundary.top_y / scale)
            prompt_bottom = round(boundary.bottom_y / scale)
            prompt_left = round(boundary.left_x / scale)
            prompt_right = round(boundary.right_x / scale)
        else:
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            scale = 1.0
            ai_w, ai_h = orig_w, orig_h
            prompt_top = boundary.top_y
            prompt_bottom = boundary.bottom_y
            prompt_left = boundary.left_x
            prompt_right = boundary.right_x

        # Extract per-edge feedback, falling back to general reason/issues
        edge_fb = verification.edge_feedback or {}
        fallback = verification.reason or verification.issues or "no details"
        edge_top = edge_fb.get("top", fallback)
        edge_bottom = edge_fb.get("bottom", fallback)
        edge_left = edge_fb.get("left", fallback)
        edge_right = edge_fb.get("right", fallback)

        prompt = DEFAULT_CORRECTION_PROMPT.format(
            top_y=prompt_top, bottom_y=prompt_bottom,
            left_x=prompt_left, right_x=prompt_right,
            img_w=ai_w, img_h=ai_h,
            edge_top=edge_top, edge_bottom=edge_bottom,
            edge_left=edge_left, edge_right=edge_right,
        )

        logger.info("Calling %s for boundary correction...", model)
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text = response.choices[0].message.content.strip()
        logger.info("Correction response: %s", text)

        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            reasoning_tokens=getattr(response.usage.completion_tokens_details, "reasoning_tokens", 0) or 0,
            total_tokens=response.usage.total_tokens,
        )

        data = _parse_json_response(text)

        # Scale pixel coords back to original image space
        if scale != 1.0:
            data["top_y"] = round(data["top_y"] * scale)
            data["bottom_y"] = round(data["bottom_y"] * scale)
            data["left_x"] = round(data["left_x"] * scale)
            data["right_x"] = round(data["right_x"] * scale)

        # Carry over metadata from original boundary — only coordinates change
        return BoundaryResult(
            top_y=data["top_y"],
            bottom_y=data["bottom_y"],
            left_x=data["left_x"],
            right_x=data["right_x"],
            confidence=data.get("confidence", boundary.confidence),
            facility_type=boundary.facility_type,
            building_count=boundary.building_count,
            notes=boundary.notes,
            reasoning=data.get("reasoning", ""),
            self_check="",
            raw_response=text,
            prompt_text=prompt,
            usage=usage,
        )
    except Exception as e:
        logger.error("Boundary correction API error: %s", e)
        raise
