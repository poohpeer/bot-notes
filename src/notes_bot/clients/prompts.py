"""Loads prompt files from clients/prompts/ — see 05-contracts.md,
"Промпты". One file per task, kept as plain text rather than inline
strings: prompts are edited far more often than code, and diffing/reviewing
a .md file is easier than a Python string literal.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@cache
def load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
