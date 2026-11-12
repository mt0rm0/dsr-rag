import json
from pathlib import Path

import pytest

from ragsst.evaluation import is_hit, load_gold_dataset, per_question_hits, recall_at_k

CORPUS = {
    "q1": ["The answer is Wednesday.", "noise"],
    "q2": ["noise", "Hotels up to 135 euros."],
    "q3": ["noise", "more noise"],
}

DATASET = [
    {"question": "q1", "expected_fragment": "wednesday", "source": "a.txt"},
    {"question": "q2", "expected_fragment": "135 euros", "source": "b.txt"},
    {"question": "q3", "expected_fragment": "missing", "source": "c.txt"},
]


def fake_retrieve(query: str, k: int) -> list[str]:
    return CORPUS[query][:k]


def test_is_hit_is_case_insensitive():
    assert is_hit(["Every WEDNESDAY afternoon"], "wednesday")


def test_is_hit_false_when_absent():
    assert not is_hit(["nothing here"], "wednesday")


def test_is_hit_empty_chunks():
    assert not is_hit([], "anything")


@pytest.mark.parametrize("k, expected", [(1, 1 / 3), (2, 2 / 3)])
def test_recall_at_k(k, expected):
    assert recall_at_k(fake_retrieve, DATASET, k) == pytest.approx(expected)


def test_per_question_hits():
    rows = per_question_hits(fake_retrieve, DATASET, k=2)
    assert [r["hit"] for r in rows] == [True, True, False]
    assert rows[0]["question"] == "q1"


def test_load_gold_dataset(tmp_path):
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(DATASET), encoding="utf-8")
    assert load_gold_dataset(path) == DATASET


def test_shipped_gold_dataset_fragments_exist_in_sources():
    root = Path(__file__).resolve().parents[1]
    for item in load_gold_dataset(root / "data" / "gold_dataset.json"):
        source = root / "data" / "sample_docs" / item["source"]
        assert item["expected_fragment"].lower() in source.read_text(encoding="utf-8").lower()
