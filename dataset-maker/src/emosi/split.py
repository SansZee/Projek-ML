"""Pembagian data (``11_split.py``).

Tiga varian, semuanya dilaporkan supaya terlihat berapa besar kebocoran yang
disebabkan aktor atau film:

  A  acak per ujaran, distratifikasi per kelas
  B  per film — satu film hanya ada di satu split
  C  per aktor — satu aktor hanya ada di satu split

Tidak ada varian yang "benar" sendiri: A memberi angka optimistis, B dan C
mengukur seberapa jauh angka itu turun ketika kebocoran ditutup.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

SPLITS = ("train", "val", "test")


@dataclass
class Assignment:
    name: str
    split: pd.Series
    key: pd.Series | None = None


def assign_random(
    frame: pd.DataFrame, *, ratios: tuple[float, float, float], seed: int
) -> pd.Series:
    """Acak per ujaran, distratifikasi per kelas."""
    rng = random.Random(seed)
    result = pd.Series(index=frame.index, dtype=object)
    for _, group in frame.groupby("label", sort=True):
        ids = list(group.index)
        rng.shuffle(ids)
        total = len(ids)
        first = min(total, int(round(total * ratios[0])))
        second = min(total, first + int(round(total * ratios[1])))
        for position, index in enumerate(ids):
            result.at[index] = (
                SPLITS[0] if position < first
                else SPLITS[1] if position < second
                else SPLITS[2]
            )
    return result


def assign_groups(
    keys: pd.Series, *, ratios: tuple[float, float, float]
) -> pd.Series:
    """Bagi kelompok utuh (film/aktor) ke split sesuai proporsi target.

    Kelompok terbesar ditempatkan lebih dulu ke split yang kekurangan paling
    banyak. Dengan sedikit kelompok, proporsi tidak bisa pas — itu dilaporkan
    apa adanya.
    """
    sizes = keys.value_counts()
    total = int(sizes.sum())
    targets = {name: total * ratio for name, ratio in zip(SPLITS, ratios, strict=True)}
    current = {name: 0 for name in SPLITS}
    mapping: dict[str, str] = {}
    order = sorted(sizes.index, key=lambda key: (-int(sizes[key]), str(key)))
    for key in order:
        # Kekurangan relatif, bukan absolut: dengan tiga kelompok, target train
        # 14 dan val 3 membuat defisit absolut train (4) selalu menang, padahal
        # val masih kosong sama sekali.
        best = max(
            SPLITS,
            key=lambda name: (
                (targets[name] - current[name]) / max(targets[name], 1e-9),
                -SPLITS.index(name),
            ),
        )
        mapping[str(key)] = best
        current[best] += int(sizes[key])
    return keys.astype(str).map(mapping)


def group_keys(
    frame: pd.DataFrame, *, actor_labels: dict[str, str] | None = None
) -> pd.Series:
    """Kunci kelompok untuk split C: aktor, atau ujaran sendiri kalau tak ada wajah."""
    keys: list[str] = []
    for row in frame.itertuples():
        actor_key = str(getattr(row, "actor_key", "") or "")
        label = ""
        if actor_labels:
            label = str(actor_labels.get(actor_key) or "").strip()
        if label:
            keys.append(f"aktor:{label}")
        elif actor_key and int(getattr(row, "emb_frames", 0) or 0) > 0:
            keys.append(f"klaster:{actor_key}")
        else:
            # Tanpa wajah, ujaran tidak bisa membocorkan identitas aktor.
            keys.append(f"ujaran:{getattr(row, 'utterance_id', '')}")
    return pd.Series(keys, index=frame.index, dtype=object)


def leakage(assignments: pd.Series, keys: pd.Series) -> dict[str, int]:
    """Berapa kelompok yang muncul di lebih dari satu split."""
    table = pd.DataFrame({"split": assignments, "key": keys.astype(str)})
    counts = table.groupby("key")["split"].nunique()
    leaked = counts[counts > 1]
    return {"groups": int(len(counts)), "leaked": int(len(leaked))}


def share(assignments: pd.Series) -> dict[str, float]:
    total = max(1, len(assignments))
    counts = assignments.value_counts()
    return {name: round(float(counts.get(name, 0)) / total, 4) for name in SPLITS}


def class_table(
    assignments: pd.Series, labels: pd.Series, classes: list[str]
) -> pd.DataFrame:
    table = pd.crosstab(labels.astype(str), assignments.astype(str))
    for name in SPLITS:
        if name not in table.columns:
            table[name] = 0
    table = table[list(SPLITS)]
    for label in classes:
        if label not in table.index:
            table.loc[label] = 0
    return table.loc[classes]
