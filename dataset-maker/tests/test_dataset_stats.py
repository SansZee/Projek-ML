from __future__ import annotations

import pandas as pd
import pytest

from emosi import dataset_stats

CLASSES = ["senang", "sedih", "marah", "takut", "netral"]


def selected_frame() -> pd.DataFrame:
    rows = []
    for index, label in enumerate(["senang", "senang", "sedih", "netral"]):
        rows.append(
            {
                "utterance_id": f"u{index}",
                "clip_id": f"u{index}",
                "film_id": "F001" if index < 2 else "F002",
                "title": "Film",
                "label": label,
                "confidence": 4,
                "agreement": 1.0,
                "duration_s": 3.0 + index,
            }
        )
    return pd.DataFrame(rows)


def test_class_summary_counts_and_shares():
    table = dataset_stats.class_summary(selected_frame(), CLASSES).set_index("label")
    assert table.loc["senang", "n"] == 2
    assert table.loc["senang", "share"] == pytest.approx(0.5)
    assert table.loc["takut", "n"] == 0
    assert table["share"].sum() == pytest.approx(1.0)


def test_film_summary_and_shares():
    table = dataset_stats.film_summary(selected_frame()).set_index("film_id")
    assert table.loc["F001", "n"] == 2
    assert table.loc["F001", "classes"] == 1
    assert table["share"].sum() == pytest.approx(1.0)


def test_balance_uses_present_classes_only():
    assert dataset_stats.balance(pd.Series([10, 5, 0])) == pytest.approx(2.0)
    assert dataset_stats.balance(pd.Series([0, 0])) == 0.0


def test_split_summary_has_a_table_per_variant():
    splits = pd.DataFrame(
        {
            "utterance_id": [f"u{i}" for i in range(6)],
            "label": ["senang", "senang", "sedih", "sedih", "netral", "netral"],
            "split_A": ["train", "train", "train", "val", "val", "test"],
            "split_B": ["train", "train", "train", "train", "val", "test"],
            "split_C": ["train"] * 4 + ["val", "test"],
        }
    )
    tables = dataset_stats.split_summary(splits, CLASSES)
    assert set(tables) == {"A", "B", "C"}
    assert tables["A"].loc["train", "total"] == 3
    assert tables["A"].loc["train", "senang"] == 2
    assert list(tables["A"].columns)[:5] == CLASSES


def test_media_summary_totals_only_finished_clips():
    clips = pd.DataFrame(
        {
            "status": ["ok", "ada", "gagal"],
            "clip_bytes": [1_000_000, 2_000_000, 0],
            "audio_bytes": [100_000, 100_000, 0],
            "face_bytes": [50_000, 50_000, 0],
        }
    )
    media = dataset_stats.media_summary(clips)
    assert media["clips"] == 2
    assert media["total_mb"] == pytest.approx(3.15, abs=0.05)


def test_compare_marks_agreement():
    labels = pd.DataFrame(
        [
            {"utterance_id": "u1", "label": "senang"},
            {"utterance_id": "u2", "label": "sedih"},
        ]
    )
    predictions = pd.DataFrame(
        [
            {"utterance_id": "u1", "predicted": "senang", "score": 0.9},
            {"utterance_id": "u2", "predicted": "marah", "score": 0.6},
        ]
    )
    merged = dataset_stats.compare(labels, predictions)
    assert list(merged["benar"]) == [True, False]


def test_compare_rejects_missing_columns():
    labels = pd.DataFrame([{"utterance_id": "u1", "label": "senang"}])
    with pytest.raises(ValueError, match="utterance_id dan predicted"):
        dataset_stats.compare(labels, pd.DataFrame([{"utterance_id": "u1"}]))


def test_confusion_and_audit_summary():
    labels = pd.DataFrame(
        [{"utterance_id": "u1", "label": "senang"}, {"utterance_id": "u2", "label": "sedih"}]
    )
    predictions = pd.DataFrame(
        [
            {"utterance_id": "u1", "predicted": "senang", "score": 0.9},
            {"utterance_id": "u2", "predicted": "senang", "score": 0.8},
        ]
    )
    merged = dataset_stats.compare(labels, predictions)
    matrix = dataset_stats.confusion(merged, CLASSES)
    assert matrix.loc["senang", "senang"] == 1
    assert matrix.loc["sedih", "senang"] == 1
    summary = dataset_stats.audit_summary(merged)
    assert summary == {"n": 2, "benar": 1, "akurasi": 0.5, "salah": 1}
