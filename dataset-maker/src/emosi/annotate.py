"""Penugasan pelabelan dan penyimpanan label (``07_annotate.py``).

Pelabelan berjalan **lokal di laptop pemilik film**, karena media tidak pernah
dipindahkan. Modul ini mengurus tiga hal yang harus sama di semua laptop:

  - **penugasan** — siapa melabeli ujaran mana, tanpa tumpang-tindih yang tidak
    disengaja dan tanpa ada ujaran yang terlewat;
  - **tukar silang** — satu set ujaran per pemilik film, dinilai juga oleh
    orang lain, supaya kappa bisa dihitung;
  - **berkas label** — satu CSV per anotator, ditulis ulang secara atomik
    setiap kali satu label disimpan, sehingga sesi bisa dilanjutkan.

Penugasan memakai kunci ``(annotator, utterance_id)``. Kunci itu yang membuat
``08_agree.py`` bisa menggabungkan label dari orang yang sama walau ia sempat
memakai dua laptop.
"""

from __future__ import annotations

import os
import random
import zipfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .qc import is_rare_genre

ROLE_OWN = "milik"
ROLE_GOLD = "gold"
ROLE_CROSS = "silang"
ROLES = (ROLE_OWN, ROLE_GOLD, ROLE_CROSS)
ROLE_ORDER = {role: index for index, role in enumerate(ROLES)}

ASSIGNMENT_COLUMNS = [
    "annotator",
    "role",
    "film_id",
    "owner",
    "scene_id",
    "utterance_id",
]

LABEL_COLUMNS = [
    "annotator",
    "utterance_id",
    "film_id",
    "scene_id",
    "role",
    "label",
    "confidence",
    "notes",
    "labeled_at",
]

# Adegan dengan lebih sedikit ujaran dari ini tidak dipilih sebagai gold set:
# satu adegan gold sebaiknya cukup berisi untuk mengukur kesepakatan.
GOLD_MIN_UTTERANCES = 8


class AssignmentError(ValueError):
    """Penugasan tidak bisa dibuat dari data yang ada."""


# -----------------------------------------------------------------------------
# Penugasan
# -----------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def pick_gold_scenes(planned: pd.DataFrame, count: int, *, seed: int = 42) -> list[str]:
    """Pilih adegan gold: sedikit, berisi, dan kalau bisa dari film berbeda.

    Satu adegan dari genre langka didahulukan kalau ada, karena di situlah
    label takut dan marah paling sering muncul dan kesepakatan paling mudah
    diuji.
    """
    if count <= 0:
        return []
    stats = (
        planned.groupby(["film_id", "scene_id"], sort=True)
        .size()
        .reset_index(name="n")
    )
    eligible = stats[stats["n"] >= GOLD_MIN_UTTERANCES]
    if eligible.empty:
        eligible = stats
    eligible = eligible.sort_values(["film_id", "scene_id"]).reset_index(drop=True)

    genre_of = planned.groupby("film_id")["genre"].first().to_dict()
    rare = [
        row.scene_id
        for row in eligible.itertuples()
        if is_rare_genre(genre_of.get(row.film_id))
    ]
    rare_set = set(rare)
    rest = [row.scene_id for row in eligible.itertuples() if row.scene_id not in rare_set]

    rng = random.Random(f"{seed}:gold")
    rng.shuffle(rare)
    rng.shuffle(rest)

    chosen: list[str] = []
    used_films: set[str] = set()
    film_of = planned.groupby("scene_id")["film_id"].first().to_dict()
    for scene in rare + rest:
        if len(chosen) >= count:
            break
        film = str(film_of.get(scene, ""))
        if film in used_films:
            continue
        chosen.append(scene)
        used_films.add(film)
    # Kalau filmnya kurang dari jumlah gold yang diminta, ambil sisanya tanpa
    # syarat beda film.
    for scene in rare + rest:
        if len(chosen) >= count:
            break
        if scene not in chosen:
            chosen.append(scene)
    return chosen[:count]


def sample_source(
    planned: pd.DataFrame, owner: str, count: int, *, seed: int = 42
) -> list[str]:
    """Ambil ``count`` ujaran milik satu orang, dari sesedikit mungkin adegan.

    Media yang harus dikirim ke penilai lain sebanding dengan jumlah adegan,
    bukan jumlah ujaran. Karena itu adegan diacak utuh lalu diisi sampai cukup,
    bukan ujaran diacak satu per satu.
    """
    source = planned[planned["owner"].astype(str) == str(owner)]
    if source.empty or count <= 0:
        return []
    scenes = sorted(source["scene_id"].astype(str).unique())
    random.Random(f"{seed}:{owner}").shuffle(scenes)

    picked: list[str] = []
    for scene in scenes:
        rows = source[source["scene_id"].astype(str) == scene].sort_values(
            "index", kind="stable"
        )
        picked.extend(rows["utterance_id"].astype(str))
        if len(picked) >= count:
            break
    return picked[:count]


def _pairs(people: list[str]) -> dict[str, str]:
    """Pasangan tetap untuk tukar silang: 0↔1, 2↔3, dst."""
    partners: dict[str, str] = {}
    for first, second in zip(people[::2], people[1::2], strict=False):
        partners[first] = second
        partners[second] = first
    return partners


def choose_source(
    person: str,
    planned: pd.DataFrame,
    owners: list[str],
    partners: dict[str, str],
    *,
    minimum: int,
) -> str | None:
    """Pilih pemilik film yang ujarannya dinilai oleh ``person``.

    Pasangan tetap didahulukan. Kalau pasangan tidak punya film (pilot: semua
    film di satu laptop), pakai pemilik dengan ujaran terbanyak supaya semua
    orang menilai set yang sama dan kappa tetap bisa dibandingkan.
    """
    candidates = [
        owner
        for owner in owners
        if owner != person
        and len(planned[planned["owner"].astype(str) == owner]) >= minimum
    ]
    if not candidates:
        return None
    partner = partners.get(person)
    if partner in candidates:
        return partner
    return max(
        candidates,
        key=lambda owner: (len(planned[planned["owner"].astype(str) == owner]), owner),
    )


def build_assignments(
    pool: pd.DataFrame,
    *,
    people: Iterable[str],
    gold: int = 2,
    cross: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """Susun tugas semua anotator dari kolam yang sudah lolos QC.

    Tiga jenis tugas:

      milik    seluruh adegan terpilih di film yang ada di laptop sendiri
      gold     dua adegan yang dinilai semua orang, untuk kappa global
      silang   150 ujaran milik orang lain, satu set per pemilik film

    Setiap ujaran hanya muncul sekali per anotator. Prioritasnya milik → gold →
    silang, karena tugas milik yang paling lengkap.
    """
    roster = list(dict.fromkeys(str(person) for person in people if str(person)))
    if len(roster) < 2:
        raise AssignmentError("butuh minimal dua nama anotator")

    planned = pool[pool["label_plan"]].copy()
    if planned.empty:
        raise AssignmentError("tidak ada ujaran dengan label_plan=True di pool.csv")

    owners = sorted(planned["owner"].astype(str).unique())
    unknown = [owner for owner in owners if owner not in roster]
    if unknown:
        raise AssignmentError(
            "film milik "
            + ", ".join(unknown)
            + " tidak ada di daftar anotator; tambahkan namanya atau pindahkan filmnya"
        )

    rows: list[dict] = []
    own_ids: dict[str, set[str]] = {person: set() for person in roster}

    for film_id, group in planned.groupby("film_id", sort=True):
        owner = str(group["owner"].iloc[0])
        for row in group.itertuples():
            rows.append(
                {
                    "annotator": owner,
                    "role": ROLE_OWN,
                    "film_id": str(film_id),
                    "owner": owner,
                    "scene_id": str(row.scene_id),
                    "utterance_id": str(row.utterance_id),
                }
            )
            own_ids[owner].add(str(row.utterance_id))

    gold_scenes = pick_gold_scenes(planned, gold, seed=seed)
    gold_set = set(gold_scenes)
    gold_ids: dict[str, set[str]] = {person: set() for person in roster}
    if gold_set:
        gold_rows = planned[planned["scene_id"].astype(str).isin(gold_set)]
        for person in roster:
            for row in gold_rows.itertuples():
                if str(row.utterance_id) in own_ids[person]:
                    continue
                rows.append(
                    {
                        "annotator": person,
                        "role": ROLE_GOLD,
                        "film_id": str(row.film_id),
                        "owner": str(row.owner),
                        "scene_id": str(row.scene_id),
                        "utterance_id": str(row.utterance_id),
                    }
                )
                gold_ids[person].add(str(row.utterance_id))

    # Ujaran gold tidak dipakai untuk tukar silang supaya satu ujaran tidak
    # punya dua peran di berkas label.
    crossable = planned[~planned["scene_id"].astype(str).isin(gold_set)]
    partners = _pairs(roster)
    cache: dict[str, list[str]] = {}
    for person in roster:
        source = choose_source(person, crossable, owners, partners, minimum=cross)
        if source is None:
            continue
        if source not in cache:
            cache[source] = sample_source(crossable, source, cross, seed=seed)
        source_rows = crossable.set_index("utterance_id", drop=False)
        for utterance_id in cache[source]:
            row = source_rows.loc[utterance_id]
            if str(utterance_id) in own_ids[person]:
                continue
            if str(utterance_id) in gold_ids[person]:
                continue
            rows.append(
                {
                    "annotator": person,
                    "role": ROLE_CROSS,
                    "film_id": str(row["film_id"]),
                    "owner": str(row["owner"]),
                    "scene_id": str(row["scene_id"]),
                    "utterance_id": str(utterance_id),
                }
            )

    frame = pd.DataFrame(rows, columns=ASSIGNMENT_COLUMNS)
    if frame.empty:
        raise AssignmentError("tidak ada tugas yang bisa dibuat")
    frame["_role"] = frame["role"].map(ROLE_ORDER)
    frame = frame.sort_values(
        ["annotator", "_role", "film_id", "scene_id", "utterance_id"], kind="stable"
    )
    frame = frame.drop(columns="_role")
    frame = frame.drop_duplicates(["annotator", "utterance_id"], keep="first")
    return frame.reset_index(drop=True)


def workload(assignments: pd.DataFrame) -> pd.DataFrame:
    """Jumlah tugas per anotator dan per jenis."""
    if assignments.empty:
        return pd.DataFrame(columns=["annotator", *ROLES, "total"])
    table = assignments.pivot_table(
        index="annotator",
        columns="role",
        values="utterance_id",
        aggfunc="count",
        fill_value=0,
    )
    for role in ROLES:
        if role not in table.columns:
            table[role] = 0
    table = table[list(ROLES)]
    table["total"] = table.sum(axis=1)
    return table.reset_index()


# -----------------------------------------------------------------------------
# Adegan dan preview
# -----------------------------------------------------------------------------


def scene_needs(assignments: pd.DataFrame) -> pd.DataFrame:
    """Adegan yang dibutuhkan, beserta siapa saja yang menilainya."""
    if assignments.empty:
        return pd.DataFrame(
            columns=["film_id", "owner", "scene_id", "n_utterances", "annotators"]
        )
    grouped = (
        assignments.groupby(["film_id", "owner", "scene_id"], sort=True)
        .agg(
            n_utterances=("utterance_id", "size"),
            annotators=("annotator", lambda values: ",".join(sorted(set(values)))),
        )
        .reset_index()
    )
    return grouped


def scenes_for(needs: pd.DataFrame, annotator: str) -> list[str]:
    """Adegan yang perlu diputar oleh satu anotator."""
    if needs.empty:
        return []
    wanted = [
        row.scene_id
        for row in needs.itertuples()
        if annotator in str(row.annotators).split(",")
    ]
    return sorted(wanted)


def preview_command(
    source: Path,
    dest: Path,
    start_s: float,
    end_s: float,
    *,
    scale: int = 480,
    crf: int = 30,
) -> list[str]:
    """Perintah ffmpeg untuk satu preview adegan.

    Browser tidak bisa memutar ``.mkv`` dan ``st.video`` tidak punya ``end_time``
    yang bisa diandalkan, jadi tiap adegan dipotong jadi mp4 480p dengan audio
    tetap ada: anotator menilai wajah dan suara sekaligus.
    """
    duration = max(0.5, float(end_s) - float(start_s))
    return [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-ss",
        f"{max(0.0, float(start_s)):.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        # Ambil satu video dan satu audio saja: sebagian .mkv membawa track
        # timecode atau subtitle yang tidak perlu dan membuat mp4 aneh.
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-dn",
        "-sn",
        "-map_chapters",
        "-1",
        "-vf",
        f"scale=-2:{int(scale)}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ac",
        "1",
        "-b:a",
        "64k",
        "-movflags",
        "+faststart",
        str(dest),
    ]


def write_preview_zip(
    dest: Path, previews: Iterable[Path], *, annotator: str, scenes: Iterable[str]
) -> Path:
    """Bungkus preview yang dibutuhkan satu anotator untuk dikirim."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED) as bundle:
        for path in sorted(previews):
            bundle.write(path, arcname=f"preview/{path.name}")
        manifest = (
            f"annotator: {annotator}\n"
            f"created_at: {_now()}\n"
            f"scenes: {', '.join(sorted(scenes))}\n"
        )
        bundle.writestr("preview_manifest.txt", manifest)
    return dest


# -----------------------------------------------------------------------------
# Berkas label
# -----------------------------------------------------------------------------


def read_labels(path: Path) -> pd.DataFrame:
    """Baca berkas label; berkas yang belum ada dianggap kosong."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS)
    frame = pd.read_csv(path)
    for column in LABEL_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[LABEL_COLUMNS]


def label_row(
    *,
    annotator: str,
    utterance: pd.Series,
    label: str,
    confidence: int,
    role: str,
    notes: str = "",
) -> dict:
    return {
        "annotator": str(annotator),
        "utterance_id": str(utterance["utterance_id"]),
        "film_id": str(utterance["film_id"]),
        "scene_id": str(utterance["scene_id"]),
        "role": str(role),
        "label": str(label),
        "confidence": int(confidence),
        "notes": str(notes),
        "labeled_at": _now(),
    }


def upsert_labels(path: Path, rows: Iterable[dict]) -> pd.DataFrame:
    """Simpan label tanpa menghapus milik orang lain atau mengulang ujaran.

    Ditulis lewat berkas sementara lalu ``replace``, supaya mati listrik di
    tengah penulisan tidak meninggalkan CSV setengah jadi.
    """
    path = Path(path)
    fresh = pd.DataFrame(list(rows), columns=LABEL_COLUMNS)
    old = read_labels(path)
    combined = pd.concat([old, fresh], ignore_index=True)
    combined = combined.drop_duplicates(["annotator", "utterance_id"], keep="last")
    combined = combined.sort_values(
        ["annotator", "film_id", "scene_id", "utterance_id"], kind="stable"
    ).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    os.replace(temporary, path)
    return combined


def progress(
    labels: pd.DataFrame, assignments: pd.DataFrame, annotator: str
) -> tuple[int, int]:
    """Berapa ujaran tugas orang ini yang sudah dilabeli."""
    mine = assignments[assignments["annotator"].astype(str) == str(annotator)]
    if mine.empty:
        return 0, 0
    # Saring per anotator: di laptop bersama, label orang lain tidak boleh
    # dihitung sebagai progres kita.
    mine_labels = labels[labels["annotator"].astype(str) == str(annotator)]
    done = len(
        set(mine_labels["utterance_id"].astype(str))
        & set(mine["utterance_id"].astype(str))
    )
    return done, len(mine)
