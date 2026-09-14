#!/usr/bin/env python
"""Compare embedding models on real notes — see docs/architecture/08-roadmap.md, M1.

This is the manual step the roadmap calls for: "30-50 своих заметок, 15-20
запросов с размеченным ожидаемым результатом, сравнение recall@5 и
латентности." It needs a real, hand-labeled dataset — there is none in this
repo, and fabricating one would just pick a model on invented numbers. Run
it yourself once you have notes.

Usage:
    uv run python scripts/benchmark_models.py dataset.json

dataset.json:
    {
      "notes": [{"id": "1", "text": "..."}, ...],
      "queries": [{"query": "...", "relevant_ids": ["1", "7"]}, ...]
    }

For each model in CANDIDATE_MODELS: embeds every note as a passage, every
query as a query, ranks notes by cosine distance, and reports recall@5
(fraction of a query's relevant_ids found in its top 5) averaged over all
queries, plus mean embedding latency per text.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from embeddings_service.model import SentenceTransformerModel

CANDIDATE_MODELS = [
    "intfloat/multilingual-e5-base",
    "paraphrase-multilingual-mpnet-base-v2",
]


def recall_at_5(ranked_ids: list[str], relevant_ids: list[str]) -> float:
    if not relevant_ids:
        return 0.0
    top5 = set(ranked_ids[:5])
    hits = sum(1 for rid in relevant_ids if rid in top5)
    return hits / len(relevant_ids)


def benchmark_model(model_name: str, notes: list[dict], queries: list[dict]) -> None:
    model = SentenceTransformerModel(model_name)

    start = time.perf_counter()
    note_vectors = np.array(model.embed([n["text"] for n in notes], "passage"))
    passage_latency_ms = (time.perf_counter() - start) * 1000 / max(len(notes), 1)
    note_ids = [n["id"] for n in notes]

    recalls = []
    query_latencies = []
    for q in queries:
        start = time.perf_counter()
        query_vector = np.array(model.embed([q["query"]], "query"))[0]
        query_latencies.append((time.perf_counter() - start) * 1000)

        # Vectors are L2-normalized, so cosine similarity is a dot product —
        # same relationship the real `<=>` operator relies on, see
        # 05-contracts.md.
        similarities = note_vectors @ query_vector
        ranked = [note_ids[i] for i in np.argsort(-similarities)]
        recalls.append(recall_at_5(ranked, q["relevant_ids"]))

    print(f"\n=== {model_name} ===")
    print(f"dim={model.dim} max_seq_length={model.max_seq_length}")
    print(f"recall@5: {sum(recalls) / len(recalls):.3f} over {len(queries)} queries")
    mean_query_latency_ms = sum(query_latencies) / len(query_latencies)
    print(f"latency: passage={passage_latency_ms:.1f}ms/text  query={mean_query_latency_ms:.1f}ms")


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        raise SystemExit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        dataset = json.load(f)

    notes = dataset["notes"]
    queries = dataset["queries"]
    if len(notes) < 10 or len(queries) < 5:
        print(
            f"warning: only {len(notes)} notes / {len(queries)} queries — "
            "the roadmap asks for 30-50 notes and 15-20 queries for a "
            "meaningful comparison",
            file=sys.stderr,
        )

    for model_name in CANDIDATE_MODELS:
        benchmark_model(model_name, notes, queries)


if __name__ == "__main__":
    main()
