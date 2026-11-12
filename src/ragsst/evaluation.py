"""Retrieval evaluation helpers used from Module 05 onwards.

A *retrieve function* takes ``(query, k)`` and returns the top-k chunk texts in
ranked order. Every technique in Module 06 is written in that shape, so all of
them can be scored with the same code.
"""

import json
from collections.abc import Callable
from pathlib import Path

RetrieveFn = Callable[[str, int], list[str]]


def load_gold_dataset(path: str | Path) -> list[dict]:
    """Load a list of ``{question, expected_fragment, source}`` dicts from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_hit(chunks: list[str], expected_fragment: str) -> bool:
    """True if any chunk contains the expected fragment (case-insensitive)."""
    fragment = expected_fragment.lower()
    return any(fragment in chunk.lower() for chunk in chunks)


def per_question_hits(retrieve: RetrieveFn, dataset: list[dict], k: int) -> list[dict]:
    """Run *retrieve* for every gold question and record whether it was a hit."""
    rows = []
    for item in dataset:
        chunks = retrieve(item["question"], k)
        rows.append(
            {"question": item["question"], "hit": is_hit(chunks, item["expected_fragment"])}
        )
    return rows


def recall_at_k(retrieve: RetrieveFn, dataset: list[dict], k: int) -> float:
    """Share of gold questions whose answer fragment appears in the top-k chunks."""
    rows = per_question_hits(retrieve, dataset, k)
    return sum(r["hit"] for r in rows) / len(rows)
