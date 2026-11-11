"""Penyaringan kolam kandidat (``06_qc.py``).

Tahap 00–05 menghasilkan katalog gabungan, tetapi tidak semua ujaran layak
dilabeli. Modul ini memutuskan mana yang masuk kolam, dengan dua lapis
pemeriksaan:

  - **per ujaran** — teks, jumlah kata, dan face ratio. Ambang face di sini
    memakai ``qc.face_ratio_min`` (0,50), lebih longgar daripada ambang awal
    ``face.face_ratio_ok`` (0,60) yang dipakai tahap 04 untuk menyaring cepat.
  - **per film** — SNR dan clip ratio. Keduanya diukur sekali per film pada
    tahap 00, jadi nilainya berlaku untuk semua ujaran film itu.

Selain menyaring, modul ini juga menyusun **rencana adegan**. Pelabelan
berjalan per adegan utuh (2–3 menit), bukan per klip terpisah: anotator perlu
konteks sebelum menilai satu ujaran. Karena itu yang dibatasi bukan jumlah
ujaran, melainkan adegan yang diambil, dan setiap adegan diambil utuh.

Yang masuk rencana label adalah ujaran yang lolos QC di dalam adegan terpilih.
Adegannya tetap diputar utuh, jadi konteks tidak hilang, tetapi ujaran yang
gagal QC tidak perlu diklik karena tidak akan pernah lolos seleksi.

Batas per film diturunkan dari config yang sudah ada, tanpa kunci baru: satu
film tidak boleh menyumbang lebih dari ``selection.max_film_share_per_class``
bagian tiap kelas, jadi jumlah ujaran yang masuk akal dilabeli dari satu film
adalah ``per_class × max_share × jumlah_kelas`` — 450 untuk 600/0,15/5. Film
yang genrenya memuat kelas langka (horor, thriller) mendapat batas lebih
longgar, karena di situlah takut dan marah paling banyak muncul. Tanpa
kelonggaran itu, kuota dua kelas tersebut tidak akan pernah terpenuhi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Genre yang biasanya memasok takut/marah. Dicocokkan sebagai substring huruf
# kecil karena nilai di films.csv ditulis bebas ("horor", "Horror", ...).
RARE_GENRES = ("horor", "horror", "thriller", "thriler", "misteri", "mystery")

# Penanda lirik lagu di subtitle. Baris seperti "# And life is like a song #"
# adalah nyanyian, bukan dialog: emosinya beda domain dan teksnya lirik
# berhak cipta, jadi tidak dipakai.
LYRIC_MARKS = ("#", "♪", "♫")

POOL_COLUMNS = [
    "film_id",
    "owner",
    "title",
    "utterance_id",
    "scene_id",
    "index",
    "start_s",
    "end_s",
    "duration_s",
    "text",
    "n_words",
    "speaker",
    "timing",
    "n_cues",
    "face_ratio",
    "emb_frames",
    "actor_key",
    "genre",
    "snr_db",
    "clip_ratio",
    "qc_pass",
    "qc_reason",
    "label_plan",
]

_FACE_COLUMNS = ("utterance_id", "face_ratio", "emb_frames", "actor_key")
_FILM_COLUMNS = ("film_id", "owner", "title", "genre", "snr_db", "clip_ratio")
_DEFAULTS: dict[str, Any] = {
    "owner": "",
    "title": "",
    "genre": "",
    "speaker": "",
    "timing": "",
    "n_cues": 0,
    "n_words": 0,
    "emb_frames": 0,
    "actor_key": "",
    "snr_db": 0.0,
    "clip_ratio": 0.0,
}


@dataclass(frozen=True)
class QcSettings:
    face_ratio_min: float
    snr_min_db: float
    clip_ratio_max: float
    min_words: int


def is_rare_genre(genre: object) -> bool:
    text = str(genre or "").lower()
    return any(token in text for token in RARE_GENRES)


def film_cap(
    *,
    per_class: int,
    max_share: float,
    n_classes: int,
    genre: object = "",
    rare_multiplier: float = 1.5,
) -> int:
    """Batas jumlah ujaran yang dilabeli dari satu film.

    ``per_class × max_share × n_classes`` adalah jumlah ujaran terbanyak yang
    masih bisa dipakai film itu kalau labelnya tersebar rata. Film genre langka
    dikalikan ``rare_multiplier`` karena kelas yang dicari justru minoritas di
    sana; kalau batasnya sama, kuota takut/marah tidak akan tercapai.
    """
    base = math.ceil(float(per_class) * float(max_share) * int(n_classes))
    if is_rare_genre(genre):
        base = math.ceil(base * float(rare_multiplier))
    return int(base)


def _number(value: object, default: float = 0.0) -> float:
    """Angka dari sel CSV, termasuk NaN dan sel kosong."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else number


def _reasons(row: pd.Series, settings: QcSettings) -> list[str]:
    """Alasan satu ujaran gagal QC. Kosong berarti lolos."""
    reasons: list[str] = []

    text = "" if pd.isna(row.get("text")) else str(row.get("text"))
    if not text.strip():
        reasons.append("teks_kosong")
    elif any(mark in text for mark in LYRIC_MARKS):
        reasons.append("lirik")

    words = int(_number(row.get("n_words")))
    if words < settings.min_words:
        reasons.append(f"kata_sedikit:{words}")

    if pd.isna(row.get("face_ratio")):
        reasons.append("tanpa_wajah")
    else:
        ratio = _number(row.get("face_ratio"))
        if ratio < settings.face_ratio_min:
            reasons.append(f"wajah_jarang:{ratio:.2f}")

    snr = _number(row.get("snr_db"))
    if snr < settings.snr_min_db:
        reasons.append(f"snr_rendah:{snr:.1f}")

    clipping = _number(row.get("clip_ratio"))
    if clipping > settings.clip_ratio_max:
        reasons.append(f"audio_klip:{clipping:.3f}")

    return reasons


def evaluate(
    films: pd.DataFrame,
    utterances: pd.DataFrame,
    faces: pd.DataFrame,
    settings: QcSettings,
) -> pd.DataFrame:
    """Gabungkan film + ujaran + wajah, lalu tandai ``qc_pass`` dan alasannya."""
    film_columns = [column for column in _FILM_COLUMNS if column in films.columns]
    face_columns = [column for column in _FACE_COLUMNS if column in faces.columns]

    frame = utterances.merge(
        films[film_columns].drop_duplicates("film_id"),
        on="film_id",
        how="left",
        validate="many_to_one",
    )
    frame = frame.merge(
        faces[face_columns].drop_duplicates("utterance_id"),
        on="utterance_id",
        how="left",
        validate="one_to_one",
    )
    # face_ratio dibiarkan NaN kalau barisnya tidak ada, supaya "tanpa wajah"
    # tidak tertukar dengan "wajahnya sedikit".
    if "face_ratio" not in frame.columns:
        frame["face_ratio"] = np.nan
    for column, default in _DEFAULTS.items():
        if column not in frame.columns:
            frame[column] = default
        elif default == "":
            frame[column] = frame[column].fillna("")
        else:
            frame[column] = frame[column].fillna(default)

    reasons = frame.apply(lambda row: _reasons(row, settings), axis=1)
    frame["qc_pass"] = reasons.map(len) == 0
    frame["qc_reason"] = reasons.map(";".join)
    frame["label_plan"] = False

    order = ["film_id", "index"] if "index" in frame.columns else ["film_id", "start_s"]
    frame = frame.sort_values(order, kind="stable").reset_index(drop=True)
    return frame[POOL_COLUMNS]


def caps_for(films: pd.DataFrame, caps: dict[str, int] | None) -> dict[str, int]:
    """Samakan kunci batas dengan film yang ada di katalog."""
    if caps is None:
        return {}
    return {str(film_id): int(value) for film_id, value in caps.items()}


def choose_scenes(
    order: list[str], counts: dict[str, int], cap: int | None
) -> list[str]:
    """Pilih adegan utuh sampai batas tercapai, tersebar sepanjang film.

    ``order`` adalah ``scene_id`` dalam urutan film. Kalau semuanya masih muat,
    semua diambil. Kalau tidak, adegan diambil dengan langkah tetap
    (``0, k, 2k, …`` lalu ``1, 1+k, …``) supaya pilihan tidak menumpuk di babak
    pertama. Satu adegan terakhir boleh melewati batas sedikit: memotong adegan
    di tengah akan menghilangkan konteks yang justru dibutuhkan anotator.
    """
    if not order:
        return []
    if cap is None:
        return list(order)

    total = sum(counts.get(scene, 0) for scene in order)
    if total <= cap:
        return list(order)

    stride = max(1, math.ceil(total / cap))
    pick_order = [
        scene for residue in range(stride) for scene in order[residue::stride]
    ]
    chosen: list[str] = []
    used = 0
    for scene in pick_order:
        if used >= cap:
            break
        chosen.append(scene)
        used += counts.get(scene, 0)
    selected = set(chosen)
    return [scene for scene in order if scene in selected]


def plan_scenes(
    pool: pd.DataFrame,
    scenes: pd.DataFrame,
    caps: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Tentukan adegan mana yang dilabeli per film.

    Adegan tanpa satu pun ujaran lolos QC tidak pernah dipilih, berapa pun
    batasnya: anotator tidak akan punya apa pun untuk dilabeli di sana.
    """
    passing = pool[pool["qc_pass"]]
    counts = (
        passing.groupby(["film_id", "scene_id"], sort=False).size().to_dict()
    )
    owners = (
        pool.groupby("film_id")["owner"].first().to_dict() if "owner" in pool else {}
    )

    rows: list[dict] = []
    for film_id, group in scenes.groupby("film_id", sort=True):
        if "index" in group.columns:
            group = group.sort_values("index", kind="stable")
        else:
            group = group.sort_values("start_s", kind="stable")

        order: list[str] = []
        for scene_id in group["scene_id"].astype(str):
            if counts.get((film_id, scene_id), 0) > 0:
                order.append(scene_id)

        cap = (caps or {}).get(str(film_id))
        chosen = set(choose_scenes(order, {s: counts[(film_id, s)] for s in order}, cap))

        for scene in group.itertuples():
            scene_id = str(scene.scene_id)
            rows.append(
                {
                    "film_id": str(film_id),
                    "owner": owners.get(str(film_id), ""),
                    "scene_id": scene_id,
                    "index": int(getattr(scene, "index", 0)),
                    "start_s": float(scene.start_s),
                    "end_s": float(scene.end_s),
                    "n_qc_pass": int(counts.get((film_id, scene_id), 0)),
                    "selected": scene_id in chosen,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "film_id",
            "owner",
            "scene_id",
            "index",
            "start_s",
            "end_s",
            "n_qc_pass",
            "selected",
        ],
    )


def apply_plan(pool: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """Tandai ``label_plan`` pada ujaran yang adegannya terpilih."""
    if plan.empty:
        pool = pool.copy()
        pool["label_plan"] = False
        return pool
    wanted = set(
        zip(
            plan.loc[plan["selected"], "film_id"].astype(str),
            plan.loc[plan["selected"], "scene_id"].astype(str),
            strict=False,
        )
    )
    keys = list(zip(pool["film_id"].astype(str), pool["scene_id"].astype(str), strict=False))
    pool = pool.copy()
    # Adegan diputar utuh supaya konteksnya terlihat, tetapi yang dilabeli hanya
    # ujaran yang lolos QC: ujaran tanpa wajah atau satu kata tidak akan pernah
    # lolos seleksi, jadi melabelinya hanya membuang waktu.
    pool["label_plan"] = [
        bool(pass_) and key in wanted
        for pass_, key in zip(pool["qc_pass"], keys, strict=True)
    ]
    return pool


def reason_counts(pool: pd.DataFrame) -> pd.Series:
    """Hitung setiap alasan gagal, termasuk ujaran yang gagal karena >1 hal."""
    failed = pool.loc[~pool["qc_pass"], "qc_reason"]
    tokens = [
        token.split(":", 1)[0]
        for value in failed.astype(str)
        for token in value.split(";")
        if token
    ]
    if not tokens:
        return pd.Series(dtype="int64")
    return pd.Series(tokens).value_counts()


def summarize(pool: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan per film untuk laporan dan layar."""
    frames = []
    for film_id, group in pool.groupby("film_id", sort=True):
        frames.append(
            {
                "film_id": film_id,
                "utterances": len(group),
                "qc_pass": int(group["qc_pass"].sum()),
                "qc_share": float(group["qc_pass"].mean()),
                "emb_share": float((group["emb_frames"] > 0).mean()),
                "label_plan": int(group["label_plan"].sum()),
            }
        )
    return pd.DataFrame(frames).reset_index(drop=True)


def cap_summary(pool: pd.DataFrame, caps: dict[str, int] | None) -> pd.DataFrame:
    """Bandingkan rencana dengan batas, supaya overshoot kelihatan."""
    summary = summarize(pool)
    summary["cap"] = [
        np.nan if caps is None else caps.get(str(film_id), np.nan)
        for film_id in summary["film_id"]
    ]
    return summary
