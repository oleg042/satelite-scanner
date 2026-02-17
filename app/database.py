"""Async SQLAlchemy engine and session factory."""

import ssl

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

# Strip query params that asyncpg doesn't understand and use proper SSL
_db_url = settings.database_url.split("?")[0]

engine = create_async_engine(
    _db_url,
    echo=False,
    pool_size=5,
    max_overflow=10,
    connect_args={"ssl": "require"},
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session
