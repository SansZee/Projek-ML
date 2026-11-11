"""Ringkasan dataset (``13_stats.py``) dan pembanding prediksi (``12_audit.py``).

Dua bagian kecil yang dipakai bersama:

  - ``class_summary``/``film_summary``/``split_summary`` untuk dataset card;
  - ``compare``/``confusion`` untuk audit label terhadap prediksi model.

Semuanya murni pandas supaya bisa diuji tanpa media.
"""

from __future__ import annotations

import pandas as pd

SPLITS = ("train", "val", "test")


def class_summary(frame: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    total = max(1, len(frame))
    rows = []
    for label in classes:
        part = frame[frame["label"].astype(str) == label]
        rows.append(
            {
                "label": label,
                "n": int(len(part)),
                "share": round(len(part) / total, 4),
                "duration_mean": round(float(part["duration_s"].mean()), 2)
                if len(part)
                else 0.0,
                "duration_median": round(float(part["duration_s"].median()), 2)
                if len(part)
                else 0.0,
                "confidence_mean": round(float(part["confidence"].mean()), 2)
                if len(part)
                else 0.0,
                "agreement_mean": round(float(part["agreement"].mean()), 3)
                if "agreement" in part and len(part)
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def film_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["film_id", "title", "n", "classes", "share"])
    total = max(1, len(frame))
    grouped = (
        frame.groupby("film_id", sort=True)
        .agg(
            title=("title", "first"),
            n=("utterance_id", "size"),
            classes=("label", "nunique"),
        )
        .reset_index()
    )
    grouped["share"] = (grouped["n"] / total).round(4)
    return grouped


def split_summary(
    splits: pd.DataFrame, classes: list[str]
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for variant, column in (("A", "split_A"), ("B", "split_B"), ("C", "split_C")):
        if column not in splits.columns:
            continue
        counts = splits.groupby([column, "label"]).size().unstack(fill_value=0)
        for label in classes:
            if label not in counts.columns:
                counts[label] = 0
        counts = counts[classes]
        counts["total"] = counts.sum(axis=1)
        result[variant] = counts
    return result


def balance(counts: pd.Series) -> float:
    """Rasio kelas terbanyak terhadap tersedikit (1,0 = seimbang)."""
    values = [int(value) for value in counts if int(value) > 0]
    if not values:
        return 0.0
    return round(max(values) / min(values), 3)


def media_summary(clips: pd.DataFrame) -> dict[str, float]:
    if clips.empty:
        return {}
    ok = clips[clips["status"].isin(("ok", "ada"))] if "status" in clips else clips
    if ok.empty:
        return {}
    total = (
        ok[["clip_bytes", "audio_bytes", "face_bytes"]].sum().sum() / 1048576
        if {"clip_bytes", "audio_bytes", "face_bytes"} <= set(ok.columns)
        else 0.0
    )
    return {
        "clips": int(len(ok)),
        "total_mb": round(float(total), 1),
        "clip_mb": round(float(ok["clip_bytes"].mean()) / 1048576, 3)
        if "clip_bytes" in ok
        else 0.0,
        "audio_kb": round(float(ok["audio_bytes"].mean()) / 1024, 1)
        if "audio_bytes" in ok
        else 0.0,
        "face_kb": round(float(ok["face_bytes"].mean()) / 1024, 1)
        if "face_bytes" in ok
        else 0.0,
    }


# -----------------------------------------------------------------------------
# Audit prediksi
# -----------------------------------------------------------------------------


def compare(
    labels: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    """Gabungkan label final dengan prediksi model; ``benar`` bila sama."""
    needed = {"utterance_id", "predicted"}
    if not needed <= set(predictions.columns):
        raise ValueError("prediksi harus punya kolom utterance_id dan predicted")
    pred = predictions.copy()
    pred["predicted"] = pred["predicted"].astype(str)
    if "score" not in pred.columns:
        pred["score"] = 0.0
    merged = labels.merge(
        pred[["utterance_id", "predicted", "score"]],
        on="utterance_id",
        how="inner",
        validate="one_to_one",
    )
    merged["benar"] = merged["label"].astype(str) == merged["predicted"]
    return merged


def confusion(
    frame: pd.DataFrame, classes: list[str]
) -> pd.DataFrame:
    table = pd.crosstab(
        frame["label"].astype(str), frame["predicted"].astype(str)
    )
    for label in classes:
        if label not in table.index:
            table.loc[label] = 0
    for label in classes:
        if label not in table.columns:
            table[label] = 0
    return table.loc[classes, classes]


def audit_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    total = int(len(frame))
    correct = int(frame["benar"].sum()) if total else 0
    return {
        "n": total,
        "benar": correct,
        "akurasi": round(correct / total, 4) if total else 0.0,
        "salah": total - correct,
    }
