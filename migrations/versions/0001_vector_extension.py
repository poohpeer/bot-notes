"""Create the vector extension

A separate first migration because it needs superuser rights and, on the
shared instance this bot deploys to, may already be installed by another
tenant — see docs/architecture/02-data-model.md, "Миграции".

Revision ID: 0001
Revises:
Create Date: 2026-09-13
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    # Never drop it: it may be in use by another tenant on the shared
    # instance (family-bot). Removing pgvector support is a manual, audited
    # operation, not something an automated downgrade should do.
    pass
