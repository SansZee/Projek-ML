"""Kesepakatan antar-anotator dan adjudikasi (``08_agree.py``).

Label masuk dari beberapa laptop sebagai satu CSV per anotator. Modul ini
menggabungkannya, menghitung kesepakatan, lalu memutuskan label akhir per
ujaran:

  - **kappa** dihitung pada label mentah, sebelum adjudikasi, karena justru
    ketidaksepakatan itu yang ingin diukur;
  - **keputusan** mayoritas: satu label harus dipilih lebih dari separuh
    penilai. Kalau tidak ada, ujaran masuk daftar adjudikasi;
  - **label tunggal** (hanya satu penilai) tetap dipakai, tetapi ditandai
    ``tunggal`` supaya tidak tercampur dengan yang benar-benar disepakati.

Dua ukuran dipakai:

  - **Cohen's kappa** untuk dua penilai, pada ujaran yang dinilai keduanya;
  - **Fleiss' kappa** untuk tiga penilai atau lebih, termasuk jumlah penilai
    yang berbeda per ujaran (gold set bisa 3 atau 4 orang kalau ada yang
    tertinggal).

Kappa hanya bermakna kalau ada ujaran yang dinilai lebih dari satu orang.
Itulah gunanya gold set dan tukar silang di tahap 07.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

STATUS_SINGLE = "tunggal"
STATUS_AGREE = "sepakat"
STATUS_MAJORITY = "mayoritas"
STATUS_DECIDED = "adjudikasi"
STATUS_PENDING = "perlu_adjudikasi"
STATUSES = (STATUS_SINGLE, STATUS_AGREE, STATUS_MAJORITY, STATUS_DECIDED, STATUS_PENDING)

BASIS_GOLD = "gold"
BASIS_CROSS = "silang"
BASIS_OWN = "milik"
BASES = (BASIS_GOLD, BASIS_CROSS, BASIS_OWN)

FINAL_COLUMNS = [
    "utterance_id",
    "film_id",
    "scene_id",
    "label",
    "confidence",
    "n_raters",
    "agreement",
    "status",
    "basis",
    "decided_by",
    "notes",
]

ADJUDICATION_COLUMNS = [
    "utterance_id",
    "film_id",
    "scene_id",
    "n_raters",
    "suara",
    "status",
    "decision",
    "decided_by",
    "notes",
]

REQUIRED_LABEL_COLUMNS = (
    "annotator",
    "utterance_id",
    "film_id",
    "scene_id",
    "role",
    "label",
    "confidence",
)


class AgreementError(ValueError):
    """Label tidak bisa digabung atau dihitung."""


@dataclass(frozen=True)
class Kappa:
    kind: str  # "cohen" | "fleiss"
    label: str  # "kesan–budi" atau "4 penilai"
    n_items: int
    observed: float
    value: float
    basis: str = "semua"

    @property
    def ok(self) -> bool:
        return not math.isnan(self.value)


# -----------------------------------------------------------------------------
# Kappa
# -----------------------------------------------------------------------------


def cohen_kappa(first: Iterable[str], second: Iterable[str]) -> tuple[float, float]:
    """Kappa Cohen untuk dua penilai: ``(kappa, kesepakatan teramati)``."""
    a = list(first)
    b = list(second)
    if len(a) != len(b):
        raise AgreementError("dua daftar label harus sama panjang")
    if not a:
        return float("nan"), float("nan")

    categories = sorted(set(a) | set(b))
    index = {category: number for number, category in enumerate(categories)}
    matrix = np.zeros((len(categories), len(categories)), dtype=float)
    for left, right in zip(a, b, strict=True):
        matrix[index[left], index[right]] += 1

    total = matrix.sum()
    observed = float(np.trace(matrix) / total)
    expected = float((matrix.sum(axis=0) / total) @ (matrix.sum(axis=1) / total))
    if math.isclose(expected, 1.0):
        # Kedua penilai memakai satu label yang sama terus; kappa tidak terdefinisi.
        return float("nan"), observed
    return (observed - expected) / (1.0 - expected), observed


def fleiss_kappa(counts: np.ndarray) -> tuple[float, float]:
    """Fleiss kappa dengan jumlah penilai berbeda per butir.

    ``counts`` berbentuk ``butir × kategori``. Rumus umum (Randolph) dipakai
    karena gold set bisa dinilai 3 orang di satu sesi dan 4 orang di sesi lain.
    Butir dengan kurang dari dua penilai dibuang.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2:
        raise AgreementError("counts harus matriks butir × kategori")
    per_item = counts.sum(axis=1)
    keep = per_item >= 2
    counts = counts[keep]
    per_item = per_item[keep]
    if counts.shape[0] == 0:
        return float("nan"), float("nan")

    agreement = ((counts**2).sum(axis=1) - per_item) / (per_item * (per_item - 1))
    observed = float((agreement * per_item).sum() / per_item.sum())
    shares = counts.sum(axis=0) / per_item.sum()
    expected = float((shares**2).sum())
    if math.isclose(expected, 1.0):
        return float("nan"), observed
    return (observed - expected) / (1.0 - expected), observed


def _counts(labels: pd.DataFrame, categories: list[str]) -> np.ndarray:
    table = (
        labels.groupby(["utterance_id", "label"]).size().unstack(fill_value=0)
    )
    for category in categories:
        if category not in table.columns:
            table[category] = 0
    return table[categories].to_numpy(dtype=float)


def pairwise_kappa(
    labels: pd.DataFrame, *, min_items: int = 10
) -> list[Kappa]:
    """Cohen kappa untuk setiap pasangan anotator yang punya irisan."""
    people = sorted(labels["annotator"].astype(str).unique())
    results: list[Kappa] = []
    for first, second in combinations(people, 2):
        left = labels[labels["annotator"].astype(str) == first][
            ["utterance_id", "label"]
        ].rename(columns={"label": "left"})
        right = labels[labels["annotator"].astype(str) == second][
            ["utterance_id", "label"]
        ].rename(columns={"label": "right"})
        shared = left.merge(right, on="utterance_id", how="inner")
        if len(shared) < min_items:
            results.append(
                Kappa(
                    "cohen",
                    f"{first}–{second}",
                    len(shared),
                    float("nan"),
                    float("nan"),
                    basis="",
                )
            )
            continue
        value, observed = cohen_kappa(shared["left"], shared["right"])
        results.append(
            Kappa("cohen", f"{first}–{second}", len(shared), observed, value)
        )
    return results


def fleiss_for(
    labels: pd.DataFrame, *, min_items: int = 10
) -> Kappa | None:
    """Fleiss kappa untuk butir yang dinilai tiga orang atau lebih."""
    multi = labels.groupby("utterance_id").filter(lambda group: len(group) >= 3)
    n_items = multi["utterance_id"].nunique()
    if n_items < min_items:
        return None
    categories = sorted(multi["label"].astype(str).unique())
    value, observed = fleiss_kappa(_counts(multi, categories))
    n_raters = int(multi.groupby("utterance_id").size().median())
    return Kappa("fleiss", f"{n_raters} penilai", n_items, observed, value)


def interpret(value: float) -> str:
    """Tafsir Landis & Koch, dipakai apa adanya di laporan."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "tidak terdefinisi"
    if value < 0:
        return "buruk"
    if value < 0.20:
        return "sedikit"
    if value < 0.40:
        return "cukup"
    if value < 0.60:
        return "sedang"
    if value < 0.80:
        return "baik"
    return "sangat baik"


# -----------------------------------------------------------------------------
# Penggabungan label
# -----------------------------------------------------------------------------


def load_labels(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Gabungkan label dari banyak berkas; satu ujaran satu label per orang.

    Kalau satu anotator sempat memakai dua laptop, baris paling baru
    (``labeled_at`` terbesar) yang dipakai.
    """
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise AgreementError("tidak ada label untuk digabung")
    labels = pd.concat(frames, ignore_index=True)
    for column in REQUIRED_LABEL_COLUMNS:
        if column not in labels.columns:
            raise AgreementError(f"kolom label tidak ada: {column}")
    labels = labels.copy()
    labels["annotator"] = labels["annotator"].astype(str)
    labels["utterance_id"] = labels["utterance_id"].astype(str)
    labels["label"] = labels["label"].astype(str)
    labels["confidence"] = pd.to_numeric(labels["confidence"], errors="coerce").fillna(0)
    if "labeled_at" not in labels.columns:
        labels["labeled_at"] = ""
    labels = labels.sort_values("labeled_at", kind="stable")
    labels = labels.drop_duplicates(["annotator", "utterance_id"], keep="last")
    return labels.reset_index(drop=True)


def validate_labels(labels: pd.DataFrame, allowed: Iterable[str]) -> list[str]:
    """Kembalikan label yang tidak dikenal (bukan kelas, ambigu, atau tidak_layak)."""
    allowed = set(allowed)
    found = sorted(set(labels["label"]) - allowed)
    return found


def basis_of(roles: Iterable[str]) -> str:
    """Jenis tugas sebuah ujaran: gold menang atas silang, silang atas milik."""
    roles = set(roles)
    if BASIS_GOLD in roles:
        return BASIS_GOLD
    if BASIS_CROSS in roles:
        return BASIS_CROSS
    return BASIS_OWN


def add_basis(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    labels["basis"] = labels.groupby("utterance_id")["role"].transform(
        lambda roles: basis_of(roles)
    )
    return labels


def overlap_matrix(labels: pd.DataFrame) -> pd.DataFrame:
    """Berapa ujaran yang dinilai bersama oleh setiap pasangan anotator."""
    people = sorted(labels["annotator"].unique())
    table = pd.DataFrame(0, index=people, columns=people, dtype=int)
    by_item = labels.groupby("utterance_id")["annotator"].apply(list)
    for people_here in by_item:
        unique = sorted(set(people_here))
        for first, second in combinations(unique, 2):
            table.loc[first, second] += 1
            table.loc[second, first] += 1
    for person in people:
        table.loc[person, person] = int(
            (labels["annotator"] == person).sum()
        )
    return table


# -----------------------------------------------------------------------------
# Suara dan keputusan
# -----------------------------------------------------------------------------


def item_votes(labels: pd.DataFrame) -> pd.DataFrame:
    """Hitung suara per ujaran, tanpa adjudikasi."""
    rows: list[dict] = []
    for utterance_id, group in labels.groupby("utterance_id", sort=False):
        counts = group["label"].value_counts()
        n_raters = int(len(group))
        top_label = str(counts.index[0])
        top_n = int(counts.iloc[0])

        if n_raters == 1:
            status = STATUS_SINGLE
        elif top_n == n_raters:
            status = STATUS_AGREE
        elif top_n * 2 > n_raters:
            status = STATUS_MAJORITY
        else:
            status = STATUS_PENDING

        if status == STATUS_PENDING:
            chosen = group
            label = ""
        else:
            chosen = group[group["label"] == top_label]
            label = top_label

        first = group.iloc[0]
        rows.append(
            {
                "utterance_id": str(utterance_id),
                "film_id": str(first["film_id"]),
                "scene_id": str(first["scene_id"]),
                "label": label,
                "confidence": round(float(chosen["confidence"].mean()), 2)
                if len(chosen)
                else 0.0,
                "n_raters": n_raters,
                "agreement": round(top_n / n_raters, 3),
                "status": status,
                "basis": basis_of(group["role"]),
                "suara": ", ".join(
                    f"{name}×{int(count)}" for name, count in counts.items()
                ),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["film_id", "scene_id", "utterance_id"], kind="stable"
    ).reset_index(drop=True)


def apply_decisions(
    votes: pd.DataFrame,
    decisions: dict[str, dict],
    *,
    allowed: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Terapkan keputusan adjudikasi pada ujaran yang belum sepakat.

    Keputusan hanya mengisi baris berstatus ``perlu_adjudikasi``. Kalau label
    mentah berubah setelah keputusan dibuat, keputusan lama tetap dipakai —
    itu keputusan manusia, bukan hasil hitungan.
    """
    allowed = set(allowed or [])
    frame = votes.copy()
    if "decided_by" not in frame.columns:
        frame["decided_by"] = ""
    if "notes" not in frame.columns:
        frame["notes"] = ""
    for index, row in frame.iterrows():
        if row["status"] != STATUS_PENDING:
            continue
        decision = decisions.get(str(row["utterance_id"]))
        if not decision:
            continue
        label = _text(decision.get("decision"))
        if not label:
            continue
        if allowed and label not in allowed:
            raise AgreementError(
                f"keputusan untuk {row['utterance_id']} bukan label sah: {label}"
            )
        frame.at[index, "label"] = label
        frame.at[index, "status"] = STATUS_DECIDED
        frame.at[index, "decided_by"] = _text(decision.get("decided_by"))
        frame.at[index, "notes"] = _text(decision.get("notes"))
    return frame


def final_rows(votes: pd.DataFrame) -> pd.DataFrame:
    """Baris untuk ``labels_final.csv``; ujaran yang belum diputuskan dibuang."""
    decided = votes[votes["status"] != STATUS_PENDING].copy()
    for column in ("decided_by", "notes"):
        if column not in decided.columns:
            decided[column] = ""
    decided = decided[FINAL_COLUMNS]
    return decided.sort_values(
        ["film_id", "scene_id", "utterance_id"], kind="stable"
    ).reset_index(drop=True)


def adjudication_rows(votes: pd.DataFrame) -> pd.DataFrame:
    """Daftar ujaran yang perlu diputuskan manusia, plus yang sudah diputuskan.

    Baris yang sudah diputuskan tetap ditulis supaya jejak keputusannya tidak
    hilang saat skrip dijalankan ulang.
    """
    pending = votes[votes["status"].isin((STATUS_PENDING, STATUS_DECIDED))].copy()
    for column in ADJUDICATION_COLUMNS:
        if column not in pending.columns:
            pending[column] = ""
    # Baris yang sudah diputuskan tetap membawa keputusannya supaya jejaknya
    # tidak hilang saat skrip dijalankan ulang.
    pending.loc[pending["status"] == STATUS_DECIDED, "decision"] = pending.loc[
        pending["status"] == STATUS_DECIDED, "label"
    ]
    return pending[ADJUDICATION_COLUMNS].reset_index(drop=True)


def _text(value: object) -> str:
    """Sel CSV sebagai teks; NaN dan sel kosong menjadi string kosong."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def read_decisions(frame: pd.DataFrame) -> dict[str, dict]:
    """Ambil keputusan dari ``adjudication.csv`` yang sudah diisi."""
    if frame.empty or "decision" not in frame.columns:
        return {}
    decisions: dict[str, dict] = {}
    for row in frame.itertuples():
        decision = _text(getattr(row, "decision", ""))
        if not decision:
            continue
        decisions[str(row.utterance_id)] = {
            "decision": decision,
            "decided_by": _text(getattr(row, "decided_by", "")),
            "notes": _text(getattr(row, "notes", "")),
        }
    return decisions


def status_counts(votes: pd.DataFrame) -> dict[str, int]:
    return {
        status: int((votes["status"] == status).sum())
        for status in STATUSES
        if (votes["status"] == status).any()
    }


def agreement_table(
    labels: pd.DataFrame, *, min_items: int = 10
) -> tuple[list[Kappa], pd.DataFrame]:
    """Kappa per jenis tugas dan satu baris ringkasan keseluruhan."""
    labels = add_basis(labels)
    rows: list[Kappa] = []
    for basis in BASES:
        subset = labels[labels["basis"] == basis]
        if subset.empty:
            continue
        for row in pairwise_kappa(subset, min_items=min_items):
            rows.append(
                Kappa(row.kind, row.label, row.n_items, row.observed, row.value, basis)
            )
        fleiss = fleiss_for(subset, min_items=min_items)
        if fleiss is not None:
            rows.append(
                Kappa(
                    fleiss.kind,
                    fleiss.label,
                    fleiss.n_items,
                    fleiss.observed,
                    fleiss.value,
                    basis,
                )
            )
    fleiss_all = fleiss_for(labels, min_items=min_items)
    if fleiss_all is not None:
        rows.append(
            Kappa(
                fleiss_all.kind,
                "semua penilai",
                fleiss_all.n_items,
                fleiss_all.observed,
                fleiss_all.value,
                "semua",
            )
        )
    table = pd.DataFrame(
        [
            {
                "basis": row.basis,
                "metode": row.kind,
                "penilai": row.label,
                "butir": row.n_items,
                "setuju": round(row.observed, 3),
                "kappa": round(row.value, 3) if row.ok else float("nan"),
                "tafsir": interpret(row.value),
            }
            for row in rows
        ]
    )
    return rows, table
