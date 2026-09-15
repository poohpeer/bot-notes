"""user_settings.debug_enabled

/debug toggle — see docs/architecture/03-ingest.md, "Debug: время
обработки". Off by default (server_default) so an existing row (or one
inserted concurrently with this migration) doesn't need backfilling.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-15
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "user_settings",
        sa.Column("debug_enabled", sa.Boolean(), nullable=False, server_default="false"),
    )


def downgrade() -> None:
    op.drop_column("user_settings", "debug_enabled")
