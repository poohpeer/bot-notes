"""Splits note text into overlapping chunks for embedding — see
docs/architecture/02-data-model.md and 05-contracts.md's `chunk` interface.

Pure function: no I/O, no model, no session — testable without
infrastructure, per docs/architecture/01-context.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...


class WhitespaceTokenizer:
    """Counts whitespace-separated words as a stand-in token count.

    The real tokenizer name comes from the embedding service's `GET /model`
    (see 05-contracts.md) and would give an exact count; wiring that in means
    pulling a full HF tokenizer into the bot's own dependencies for a number
    only used to size chunks. This estimate is consistently within the same
    order of magnitude for the languages this bot targets (ru/en/he) and
    `target`/`max_seq_length` already carry headroom — see the embedding
    service's own token_count column, which records the real count per chunk
    for diagnostics once a chunk is actually embedded.
    """

    def count(self, text: str) -> int:
        return len(text.split())


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    token_count: int


def chunk(
    text: str,
    *,
    tokenizer: Tokenizer | None = None,
    target: int = 200,
    overlap: int = 40,
    min_size: int = 20,
    max_chunks: int = 50,
) -> list[Chunk]:
    """Split `text` into overlapping word-windows sized by `tokenizer`.

    - `target`: desired token count per chunk.
    - `overlap`: tokens repeated at the start of the next chunk, so a
      sentence spanning a boundary is not orphaned in one half.
    - `min_size`: a trailing remainder smaller than this is merged into the
      previous chunk instead of becoming its own — avoids a near-empty last
      chunk that mostly just wastes an embedding call.
    - `max_chunks`: hard cap so one very long note cannot generate an
      unbounded number of chunks (and embedding calls) in a single job.
    """
    tokenizer = tokenizer or WhitespaceTokenizer()
    words = text.split()
    if not words:
        return []

    if overlap >= target:
        raise ValueError(f"overlap ({overlap}) must be smaller than target ({target})")

    step = target - overlap
    windows: list[list[str]] = []
    start = 0
    while start < len(words):
        window = words[start : start + target]
        windows.append(window)
        if start + target >= len(words):
            break
        start += step

    # Merge a too-small trailing window into the one before it, per
    # `min_size` — a lone leftover word is not worth its own vector.
    if len(windows) > 1 and tokenizer.count(" ".join(windows[-1])) < min_size:
        overlap_words = windows[-2][-overlap:] if overlap else []
        tail_start = len(overlap_words)
        windows[-2] = windows[-2] + windows[-1][tail_start:]
        windows.pop()

    windows = windows[:max_chunks]

    return [
        Chunk(index=i, text=(t := " ".join(w)), token_count=tokenizer.count(t))
        for i, w in enumerate(windows)
    ]
