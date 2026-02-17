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
class ValidationResult:
    approved: bool
    reason: str
    facility_type: str
    notes: str


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


def _image_to_base64(image_path: str) -> str:
    """Read image file and return base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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
    model: str = "gpt-4o-mini",
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
    img_b64 = _image_to_base64(image_path)

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
        data = _parse_json_response(text)
        return ValidationResult(
            approved=data.get("approved", False),
            reason=data.get("reason", ""),
            facility_type=data.get("facility_type", ""),
            notes=data.get("notes", ""),
        )
    except Exception as e:
        logger.error("Validation API error: %s", e)
        return None


def detect_facility_boundary(
    image_path: str,
    name: str,
    lat: float,
    lng: float,
    width_m: float,
    height_m: float,
    api_key: str,
    model: str = "gpt-4o",
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

    img = Image.open(image_path)
    img_w, img_h = img.size
    img.close()

    # Add image context to the prompt
    context = (
        f"\n\n## IMAGE CONTEXT\n"
        f"- Facility name: {name}\n"
        f"- Center coordinates: ({lat}, {lng})\n"
        f"- Image dimensions: {img_w}px x {img_h}px\n"
        f"- Coverage: approximately {width_m:.0f}m x {height_m:.0f}m\n\n"
        f"Respond with ONLY a JSON object:\n"
        f'{{"top_y": <int>, "bottom_y": <int>, "left_x": <int>, "right_x": <int>, '
        f'"confidence": "high/medium/low", "facility_type": "<type>", '
        f'"building_count": <int>, "notes": "<observations>"}}'
    )
    full_prompt = prompt_template + context
    img_b64 = _image_to_base64(image_path)

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
        data = _parse_json_response(text)
        return BoundaryResult(
            top_y=data["top_y"],
            bottom_y=data["bottom_y"],
            left_x=data["left_x"],
            right_x=data["right_x"],
            confidence=data.get("confidence", "medium"),
            facility_type=data.get("facility_type", "unknown"),
            building_count=data.get("building_count", 0),
            notes=data.get("notes", ""),
        )
    except Exception as e:
        logger.error("Boundary detection API error: %s", e)
        return None
