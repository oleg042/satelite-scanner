"""Simple session-based authentication for the Satellite Scanner."""

import json
import logging
import secrets
import time

from sqlalchemy import select

from app.database import async_session
from app.models import Setting

logger = logging.getLogger(__name__)

# In-memory session store: token → creation timestamp
_sessions: dict[str, float] = {}

SESSION_TTL = 86400  # 24 hours


async def _persist_sessions() -> None:
    """Serialize current sessions to the DB for persistence across redeploys."""
    async with async_session() as db:
        result = await db.execute(select(Setting).where(Setting.key == "app_sessions"))
        setting = result.scalar_one_or_none()
        payload = json.dumps(_sessions)
        if setting:
            setting.value = payload
        else:
            db.add(Setting(key="app_sessions", value=payload))
        await db.commit()


async def load_sessions() -> None:
    """Load persisted sessions from DB into memory, discarding expired ones."""
    async with async_session() as db:
        result = await db.execute(select(Setting).where(Setting.key == "app_sessions"))
        setting = result.scalar_one_or_none()
    if not setting or not setting.value:
        return
    try:
        stored = json.loads(setting.value)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Corrupt app_sessions in DB, ignoring")
        return
    now = time.time()
    for token, created in stored.items():
        if now - created <= SESSION_TTL:
            _sessions[token] = created
    logger.info("Restored %d sessions from DB (%d expired, discarded)", len(_sessions), len(stored) - len(_sessions))


async def clear_sessions() -> None:
    """Wipe all sessions (in-memory + DB). Used on password change."""
    _sessions.clear()
    async with async_session() as db:
        result = await db.execute(select(Setting).where(Setting.key == "app_sessions"))
        setting = result.scalar_one_or_none()
        if setting:
            await db.delete(setting)
            await db.commit()
    logger.info("All sessions cleared")


async def create_session() -> str:
    """Generate a new session token, store it, and persist to DB."""
    token = secrets.token_hex(32)
    _sessions[token] = time.time()
    await _persist_sessions()
    return token


def validate_session(token: str | None) -> bool:
    """Check if a session token is valid and not expired."""
    if not token:
        return False
    created = _sessions.get(token)
    if created is None:
        return False
    if time.time() - created > SESSION_TTL:
        _sessions.pop(token, None)
        return False
    return True


async def verify_password(password: str) -> bool:
    """Compare password against the app_password setting in the DB."""
    async with async_session() as db:
        result = await db.execute(select(Setting).where(Setting.key == "app_password"))
        setting = result.scalar_one_or_none()
    stored = setting.value if setting else "1234"
    return password == stored
