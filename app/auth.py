"""Simple session-based authentication for the Satellite Scanner."""

import secrets
import time

from sqlalchemy import select

from app.database import async_session
from app.models import Setting

# In-memory session store: token → creation timestamp
_sessions: dict[str, float] = {}

SESSION_TTL = 86400  # 24 hours


def create_session() -> str:
    """Generate a new session token and store it."""
    token = secrets.token_hex(32)
    _sessions[token] = time.time()
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
