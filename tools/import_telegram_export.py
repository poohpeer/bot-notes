#!/usr/bin/env python
"""One-off importer for a Telegram Desktop export (result.json) — see
docs/architecture/03-ingest.md, "Импорт экспорта Telegram".

Stubbed until M8.
"""

from __future__ import annotations

import sys


def main() -> None:
    print("import_telegram_export: not implemented yet (lands in M8)", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
