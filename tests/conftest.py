"""Shared fixtures for tests that need a real Postgres (ACL, repositories).

Schema is created via Base.metadata.create_all rather than alembic — faster
per test run, and migrations are already exercised for real in CI (see
.github/workflows/ci.yml). Each test runs inside a transaction that is rolled
back afterward, so tests never see each other's data and never need manual
cleanup.
"""

from __future__ import annotations

import os

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from notes_bot.db.models import Base


def _database_url() -> str:
    return os.environ.get(
        "DATABASE_URL", "postgresql+psycopg://postgres:test@localhost:5432/postgres"
    )


@pytest_asyncio.fixture(scope="session")
async def db_engine():
    engine = create_async_engine(_database_url())
    async with engine.begin() as conn:
        # The pgvector image ships the extension but doesn't enable it — the
        # first real migration does that (see migrations/versions/0001_*),
        # and this test setup bypasses migrations entirely for speed.
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine) -> AsyncSession:
    connection = await db_engine.connect()
    transaction = await connection.begin()
    factory = async_sessionmaker(bind=connection, expire_on_commit=False)
    session = factory()

    yield session

    await session.close()
    await transaction.rollback()
    await connection.close()
