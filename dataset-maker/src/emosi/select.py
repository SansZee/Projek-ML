"""Seleksi berkuota (``09_select.py``).

Label sudah final, tetapi tidak semuanya dipakai. Dataset diseimbangkan:

  - **600 per kelas** (``selection.per_class``);
  - **maksimal 15% satu kelas dari satu film** (``max_film_share_per_class``),
    supaya film yang paling banyak menyumbang takut tidak mendominasi kelas
    itu dan split per film tetap masuk akal;
  - **keyakinan minimal 3** (``min_label_conf``), supaya label yang ragu tidak
    ikut hanya karena kuotanya kurang.

Karena batas 15% itu, 600 per kelas baru bisa tercapai kalau ada sekitar tujuh
film atau lebih per kelas. Dengan tiga film, plafonnya 270 per kelas — skrip
melaporkan kekurangan itu apa adanya, bukan menurunkan batas diam-diam.

Urutan pengambilan: bergiliran antar film, dan di dalam satu film dipilih
keyakinan tertinggi lalu klip yang lebih panjang. Hasilnya deterministik
untuk data yang sama.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

AMBIGUOUS = "ambigu"
NOT_ELIGIBLE = "tidak_layak"

SELECTED_COLUMNS = [
    "clip_id",
    "utterance_id",
    "film_id",
    "scene_id",
    "owner",
    "title",
    "label",
    "confidence",
    "status",
    "n_raters",
    "agreement",
    "start_s",
    "end_s",
    "duration_s",
    "actor_key",
    "emb_frames",
]


@dataclass
class Selection:
    rows: pd.DataFrame
    shortfall: dict[str, int] = field(default_factory=dict)
    per_film: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def complete(self) -> bool:
        return not any(self.shortfall.values())


def eligible(
    labels: pd.DataFrame,
    *,
    classes: list[str],
    min_conf: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Buang label yang tidak boleh masuk dataset, beserta alasannya."""
    frame = labels.copy()
    frame["label"] = frame["label"].astype(str)
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0)

    known = frame["label"].isin(classes)
    reasons = pd.Series("", index=frame.index, dtype=object)
    reasons[~known] = "bukan_kelas"
    reasons[known & (frame["confidence"] < float(min_conf))] = "keyakinan_rendah"
    keep = reasons == ""
    return frame[keep].copy(), reasons[~keep].value_counts()


def _order(candidates: pd.DataFrame) -> pd.DataFrame:
    """Urutan pilihan di dalam satu film: yakin dulu, lalu klip lebih panjang."""
    return candidates.assign(
        _confidence=-pd.to_numeric(candidates["confidence"], errors="coerce").fillna(0),
        _duration=-pd.to_numeric(candidates["duration_s"], errors="coerce").fillna(0),
    ).sort_values(
        ["_confidence", "_duration", "utterance_id"], kind="stable"
    )


def select_rows(
    frame: pd.DataFrame,
    *,
    classes: list[str],
    per_class: int,
    max_share: float,
) -> Selection:
    """Ambil kuota per kelas dengan batas per film dan sebaran antar film."""
    frame = frame.copy()
    frame["label"] = frame["label"].astype(str)
    cap = max(1, math.floor(float(per_class) * float(max_share)))

    chosen: list[pd.DataFrame] = []
    shortfall: dict[str, int] = {}
    for label in classes:
        candidates = frame[frame["label"] == label]
        if candidates.empty:
            shortfall[label] = per_class
            continue

        per_film = {
            str(film): _order(group)
            for film, group in candidates.groupby("film_id", sort=False)
        }
        # Film dengan kandidat terbanyak lebih dulu, supaya putaran pertama
        # sudah menyebar dan film kecil tidak kehabisan jatah karena urutan nama.
        order = sorted(per_film, key=lambda film: (-len(per_film[film]), film))
        taken = {film: 0 for film in order}
        picked: dict[str, list] = {film: [] for film in order}
        total = 0

        while total < per_class:
            progressed = False
            for film in order:
                if total >= per_class:
                    break
                if taken[film] >= cap or taken[film] >= len(per_film[film]):
                    continue
                row = per_film[film].iloc[taken[film]]
                picked[film].append(row)
                taken[film] += 1
                total += 1
                progressed = True
            if not progressed:
                break

        if total < per_class:
            shortfall[label] = per_class - total
        for film in order:
            if picked[film]:
                chosen.append(pd.DataFrame(picked[film]))

    if not chosen:
        empty = pd.DataFrame(columns=SELECTED_COLUMNS)
        return Selection(rows=empty, shortfall=shortfall)

    selected = pd.concat(chosen, ignore_index=True)
    selected["clip_id"] = selected["utterance_id"].astype(str)
    for column in SELECTED_COLUMNS:
        if column not in selected.columns:
            selected[column] = ""
    selected = selected[SELECTED_COLUMNS].sort_values(
        ["label", "film_id", "utterance_id"], kind="stable"
    ).reset_index(drop=True)

    counts = (
        selected.groupby(["film_id", "label"]).size().unstack(fill_value=0)
    )
    for label in classes:
        if label not in counts.columns:
            counts[label] = 0
    counts = counts[classes]
    counts["total"] = counts.sum(axis=1)
    counts = counts.reset_index().sort_values("film_id", kind="stable")

    return Selection(rows=selected, shortfall=shortfall, per_film=counts)


def class_counts(selected: pd.DataFrame, classes: list[str]) -> dict[str, int]:
    values = selected["label"].astype(str).value_counts().to_dict() if len(selected) else {}
    return {label: int(values.get(label, 0)) for label in classes}
