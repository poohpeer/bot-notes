"""notes.source_type: add 'table_event'

/events_table — see docs/architecture/03-ingest.md, "/events_table":
one note per event parsed from a screenshot table (exam/holiday
schedule), created by the confirm step rather than by the usual
classify_text_message/extractor path.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-18
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

_OLD_CK = "source_type IN ('text','voice','page','youtube','instagram','map')"
_NEW_CK = "source_type IN ('text','voice','page','youtube','instagram','map','table_event')"


def upgrade() -> None:
    op.drop_constraint("notes_source_type_ck", "notes", type_="check")
    op.create_check_constraint("notes_source_type_ck", "notes", _NEW_CK)


def downgrade() -> None:
    op.drop_constraint("notes_source_type_ck", "notes", type_="check")
    op.create_check_constraint("notes_source_type_ck", "notes", _OLD_CK)
