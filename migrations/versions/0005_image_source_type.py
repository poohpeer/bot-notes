"""notes.source_type: add 'image'

A photo or screenshot sent without the "/events_table" caption becomes a
note of its own — see docs/architecture/03-ingest.md, "Фото и скриншоты".
`source_url` holds the Telegram file_id(s), newline-separated for an album,
the same way a voice note keeps its file_id there.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-27
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_OLD_CK = "source_type IN ('text','voice','page','youtube','instagram','map','table_event')"
_NEW_CK = "source_type IN ('text','voice','page','youtube','instagram','map','table_event','image')"


def upgrade() -> None:
    op.drop_constraint("notes_source_type_ck", "notes", type_="check")
    op.create_check_constraint("notes_source_type_ck", "notes", _NEW_CK)


def downgrade() -> None:
    op.drop_constraint("notes_source_type_ck", "notes", type_="check")
    op.create_check_constraint("notes_source_type_ck", "notes", _OLD_CK)
