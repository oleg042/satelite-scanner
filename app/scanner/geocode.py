"""Shared Serper Maps geocoding with retry + fuzzy address matching."""

import logging
from difflib import SequenceMatcher

import httpx

logger = logging.getLogger(__name__)


def _address_similarity(a: str, b: str) -> float:
    """Return 0-1 similarity ratio between two address strings."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


async def serper_maps_resolve(
    api_key: str,
    facility_name: str,
    address: str,
) -> dict | None:
    """Resolve a facility to coordinates via Serper Maps endpoint.

    Strategy:
      1. Search "name + address" → require exactly 1 result + fuzzy match ≥ 0.80
      2. If that fails, search "address only" → require exactly 1 result (no fuzzy check)
      3. Otherwise return None.
    """
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=10) as client:
        # --- Search 1: name + address ---
        query1 = f"{facility_name} {address}".strip()
        if query1:
            result = await _serper_maps_search(client, headers, query1)
            if result is not None:
                similarity = _address_similarity(result["address"], address) if address else 0.0
                if similarity >= 0.80:
                    logger.info(
                        "Serper Maps search 1 accepted (similarity=%.2f): %s",
                        similarity, query1,
                    )
                    return result
                logger.info(
                    "Serper Maps search 1 rejected (similarity=%.2f): %s",
                    similarity, query1,
                )

        # --- Search 2: address only ---
        if address.strip():
            result = await _serper_maps_search(client, headers, address)
            if result is not None:
                logger.info("Serper Maps search 2 accepted (address-only): %s", address)
                return result
            logger.info("Serper Maps search 2 failed (address-only): %s", address)

    return None


async def _serper_maps_search(
    client: httpx.AsyncClient,
    headers: dict,
    query: str,
) -> dict | None:
    """POST to Serper /maps and return the result only if exactly 1 place is returned."""
    try:
        resp = await client.post(
            "https://google.serper.dev/maps",
            headers=headers,
            json={"q": query},
        )
    except Exception as e:
        logger.warning("Serper Maps request error: %s", e)
        return None

    if resp.status_code != 200:
        logger.warning("Serper Maps API error %d for query: %s", resp.status_code, query)
        return None

    places = resp.json().get("places", [])
    if len(places) != 1:
        logger.info(
            "Serper Maps returned %d results (need exactly 1) for: %s",
            len(places), query,
        )
        return None

    p = places[0]
    if p.get("latitude") is None or p.get("longitude") is None:
        return None

    return {
        "lat": p["latitude"],
        "lng": p["longitude"],
        "display_name": f"{p.get('title', '')} — {p.get('address', '')}".strip(" —"),
        "address": p.get("address", ""),
        "source": "serper",
        "category": p.get("category"),
        "phone": p.get("phoneNumber"),
        "rating": p.get("rating"),
    }
