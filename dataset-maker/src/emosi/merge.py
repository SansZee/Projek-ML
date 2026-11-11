"""Penggabungan kiriman antar-anggota (``05_merge.py``).

Tahap 00–04 berjalan di laptop masing-masing anggota, jadi hasilnya tersebar.
Yang digabung di sini hanya berkas kecil: tabel dan embedding wajah, sekitar
60 MB untuk 20 film. Filmnya sendiri tidak pernah ikut — kolom ``path`` dibuang
dan diganti ``owner`` supaya katalog gabungan tidak menyimpan path mesin orang
lain. Setiap laptop menerjemahkan ``film_id`` ke path-nya sendiri lewat
``data/films.csv`` lokal.

Dua peran:

  anggota       ``--pack <nama>``      membungkus hasil sendiri jadi satu zip
  koordinator   ``--from <kiriman>``   memeriksa lalu menulis ``data/merged/``

Pemeriksaan yang menggagalkan penggabungan (tidak ada berkas ditulis):

  - hash ``config.yaml`` berbeda antar laptop
  - ``film_id`` yang sama datang dari dua anggota
  - ``utterance_id`` ganda
  - jumlah baris ``.npy`` tidak sama dengan ``faces/<film_id>.csv``
  - ada ujaran tanpa baris wajah, atau sebaliknya
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TABLES = ("films.csv", "utterances.csv", "faces.csv", "actors.csv", "scenes.csv")
OPTIONAL_TABLES = ("reject_log.csv",)
META_FILES = (
    "faces/faces_meta.json",
    "scenes/scenes_meta.json",
    "utterances/utterances_meta.json",
)
MANIFEST = "handover.json"
FORMAT = 1

# Kolom films.csv yang hanya berarti di laptop pemilik film.
MACHINE_COLUMNS = ("path", "srt_path")


class MergeError(RuntimeError):
    """Kiriman tidak bisa dibaca atau digabung."""


@dataclass
class Problem:
    level: str  # "error" | "warning"
    message: str

    def __str__(self) -> str:
        return f"{self.level}: {self.message}"


@dataclass
class Contributor:
    owner: str
    manifest: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    embeddings: dict[str, np.ndarray]
    film_rows: dict[str, int] = field(default_factory=dict)
    # Hash config yang tercatat saat 04 membuat embedding, per film.
    film_config: dict[str, str] = field(default_factory=dict)

    def table(self, name: str) -> pd.DataFrame:
        return self.tables.get(name, pd.DataFrame())


# -----------------------------------------------------------------------------
# Membungkus kiriman
# -----------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audition_passed(films: pd.DataFrame) -> pd.Series:
    """Film yang lolos audisi; tanpa kolom ``pass_audition`` semua dianggap lolos.

    Film yang ditolak tidak pernah diproses 01–04, jadi tidak punya ``.npy``.
    Pemeriksaan embedding harus melewatinya, bukan menggagalkan kiriman.
    """
    if "pass_audition" not in films.columns:
        return pd.Series(True, index=films.index)
    return films["pass_audition"].astype(str).str.lower().isin(("true", "1", "yes"))


def pack_files(data_dir: Path) -> list[Path]:
    """Berkas yang ikut dalam kiriman.

    Semua kecil dan bisa digabung. Video, audio, dan klip sengaja tidak ikut:
    ukurannya ratusan GB dan tidak diperlukan tahap 05–09.
    """
    files: list[Path] = []
    for name in (*TABLES, *OPTIONAL_TABLES):
        path = data_dir / name
        if path.exists():
            files.append(path)
    for name in META_FILES:
        path = data_dir / name
        if path.exists():
            files.append(path)
    faces_dir = data_dir / "faces"
    if faces_dir.is_dir():
        for pattern in ("*.csv", "*.npy"):
            files.extend(sorted(faces_dir.glob(pattern)))
    return files


def build_manifest(
    *,
    owner: str,
    data_dir: Path,
    config_sha256: str,
    uv_lock_sha256: str,
) -> dict[str, Any]:
    """Ringkasan isi kiriman, ditulis di dalam zip supaya bisa diperiksa."""
    films = pd.read_csv(data_dir / "films.csv")
    utterances = pd.read_csv(data_dir / "utterances.csv")
    faces_path = data_dir / "faces.csv"
    n_face_ok = 0
    if faces_path.exists():
        faces = pd.read_csv(faces_path)
        if "face_ok" in faces.columns:
            n_face_ok = int(faces["face_ok"].sum())
    return {
        "format": FORMAT,
        "owner": owner,
        "packed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_sha256": config_sha256,
        "uv_lock_sha256": uv_lock_sha256,
        "films": sorted(films.loc[audition_passed(films), "film_id"].astype(str)),
        "n_utterances": int(len(utterances)),
        "n_face_ok": n_face_ok,
    }


def write_bundle(dest: Path, project_root: Path, files: Iterable[Path], manifest: dict) -> Path:
    """Tulis zip kiriman; ``arcname`` relatif ke root projek."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, ensure_ascii=False)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        bundle.writestr(MANIFEST, payload)
        for path in files:
            bundle.write(path, arcname=path.relative_to(project_root).as_posix())
    return dest


# -----------------------------------------------------------------------------
# Membaca kiriman
# -----------------------------------------------------------------------------


def _read_tree(entry: Path) -> dict[str, bytes]:
    """Baca seluruh isi kiriman jadi ``{path relatif: isi}``.

    Ukurannya puluhan MB, jadi menahan di memori lebih murah daripada menulis
    ekstraksi sementara lalu menghapusnya lagi.
    """
    if entry.is_dir():
        return {
            path.relative_to(entry).as_posix(): path.read_bytes()
            for path in sorted(entry.rglob("*"))
            if path.is_file()
        }
    if entry.suffix.lower() == ".zip":
        tree: dict[str, bytes] = {}
        with zipfile.ZipFile(entry) as bundle:
            for info in bundle.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                if name.startswith("/") or ".." in Path(name).parts:
                    raise MergeError(f"{entry.name}: entri tidak aman: {name}")
                tree[name] = bundle.read(info)
        return tree
    raise MergeError(f"{entry} bukan folder atau .zip")


def _find_root(tree: dict[str, bytes], entry: Path) -> str:
    """Cari awalan folder yang memuat ``data/utterances.csv``."""
    marker = "data/utterances.csv"
    hits = [key for key in tree if key == marker or key.endswith("/" + marker)]
    if not hits:
        raise MergeError(f"{entry.name}: tidak ada data/utterances.csv di dalamnya")
    prefixes = {key[: -len(marker)] for key in hits}
    if len(prefixes) > 1:
        raise MergeError(f"{entry.name}: ada lebih dari satu folder data/ di dalamnya")
    return prefixes.pop()


def _default_owner(entry: Path) -> str:
    return entry.stem if entry.suffix.lower() == ".zip" else entry.name


def _read_table(payload: bytes) -> pd.DataFrame:
    """Baca CSV di dalam kiriman; berkas kosong dianggap tabel kosong."""
    try:
        return pd.read_csv(io.BytesIO(payload))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_contributor(entry: Path, *, owner: str | None = None) -> Contributor:
    """Baca satu kiriman (folder hasil ekstrak atau zip)."""
    entry = Path(entry).expanduser()
    tree = _read_tree(entry)
    root = _find_root(tree, entry)

    manifest: dict[str, Any] = {}
    manifest_key = root + MANIFEST
    if manifest_key in tree:
        try:
            manifest = json.loads(tree[manifest_key].decode("utf-8"))
        except json.JSONDecodeError as error:
            raise MergeError(f"{entry.name}: {MANIFEST} rusak: {error}") from error

    name = owner or str(manifest.get("owner") or _default_owner(entry))

    tables: dict[str, pd.DataFrame] = {}
    for table in (*TABLES, *OPTIONAL_TABLES):
        key = root + "data/" + table
        if key in tree:
            tables[table] = _read_table(tree[key])

    missing = [name_ for name_ in TABLES if name_ not in tables]
    for required in ("films.csv", "utterances.csv"):
        if required in tables and tables[required].empty:
            raise MergeError(f"{name}: data/{required} kosong")
    if missing:
        raise MergeError(
            f"{name}: berkas wajib tidak ada: {', '.join(missing)}. "
            "Jalankan scripts/05_merge.py --pack di laptop anggota."
        )

    film_config: dict[str, str] = {}
    meta_key = root + "data/faces/faces_meta.json"
    if meta_key in tree:
        try:
            entries = json.loads(tree[meta_key].decode("utf-8")).get("films") or []
        except json.JSONDecodeError:
            entries = []
        for entry in entries:
            if "film_id" in entry and entry.get("config_sha256"):
                film_config[str(entry["film_id"])] = str(entry["config_sha256"])

    embeddings: dict[str, np.ndarray] = {}
    film_rows: dict[str, int] = {}
    films = tables["films.csv"]
    for film_id in films.loc[audition_passed(films), "film_id"].astype(str):
        csv_key = root + f"data/faces/{film_id}.csv"
        npy_key = root + f"data/faces/{film_id}.npy"
        if npy_key not in tree:
            raise MergeError(f"{name}: embedding wajah {film_id}.npy tidak ada")
        if csv_key in tree:
            film_rows[film_id] = len(_read_table(tree[csv_key]))
        embeddings[film_id] = np.load(io.BytesIO(tree[npy_key]))

    return Contributor(
        owner=name,
        manifest=manifest,
        tables=tables,
        embeddings=embeddings,
        film_rows=film_rows,
        film_config=film_config,
    )


# -----------------------------------------------------------------------------
# Pemeriksaan
# -----------------------------------------------------------------------------


def embedding_problems(faces: pd.DataFrame, n_rows: int) -> list[str]:
    """Cek ``emb_row`` benar-benar menunjuk baris di ``.npy`` film itu."""
    problems: list[str] = []
    if "emb_row" not in faces.columns:
        return ["kolom emb_row tidak ada"]
    if len(faces) != n_rows:
        problems.append(f"npy {n_rows} baris, faces.csv {len(faces)} baris")
    rows = faces["emb_row"].astype(int).to_numpy()
    if len(rows) == 0:
        return problems
    if len(set(rows.tolist())) != len(rows):
        problems.append("emb_row ada yang ganda")
    elif rows.min() != 0 or rows.max() != len(rows) - 1:
        problems.append("emb_row bukan 0..n-1")
    return problems


def duplicate_films(contributors: Iterable[Contributor]) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for contributor in contributors:
        for film_id in contributor.table("films.csv")["film_id"].astype(str):
            owners.setdefault(film_id, []).append(contributor.owner)
    return {film: names for film, names in owners.items() if len(names) > 1}


def resolve_duplicates(
    contributors: list[Contributor], *, keep: str | None = None
) -> tuple[list[Contributor], list[Problem]]:
    """Buang film ganda dari semua anggota kecuali satu.

    Film yang sama diproses dua orang membuat dataset membengkak dan, lebih
    buruk, film itu bisa masuk train dan test sekaligus.
    """
    problems: list[Problem] = []
    owners = duplicate_films(contributors)
    if not owners:
        return contributors, problems

    drop: dict[str, set[str]] = {}
    for film_id, names in sorted(owners.items()):
        unique = sorted(set(names))
        if keep and keep in unique:
            drop[film_id] = {name for name in unique if name != keep}
            problems.append(
                Problem(
                    "warning",
                    f"{film_id} datang dari {', '.join(unique)}; "
                    f"dipakai punya {keep}, sisanya dibuang",
                )
            )
        else:
            problems.append(
                Problem(
                    "error",
                    f"{film_id} datang dari {', '.join(unique)}. "
                    "Hapus salah satu dari kiriman, atau pakai --keep <nama>",
                )
            )
    if any(problem.level == "error" for problem in problems):
        return contributors, problems

    cleaned: list[Contributor] = []
    for contributor in contributors:
        drop_here = {
            film for film, names in drop.items() if contributor.owner in names
        }
        if not drop_here:
            cleaned.append(contributor)
            continue
        tables = {
            name: frame[~frame["film_id"].astype(str).isin(drop_here)]
            for name, frame in contributor.tables.items()
        }
        cleaned.append(
            Contributor(
                owner=contributor.owner,
                manifest=contributor.manifest,
                tables=tables,
                embeddings={
                    film: matrix
                    for film, matrix in contributor.embeddings.items()
                    if film not in drop_here
                },
                film_rows={
                    film: rows
                    for film, rows in contributor.film_rows.items()
                    if film not in drop_here
                },
                film_config={
                    film: value
                    for film, value in contributor.film_config.items()
                    if film not in drop_here
                },
            )
        )
    return cleaned, problems


def validate(
    contributors: Iterable[Contributor],
    *,
    config_sha256: str,
    uv_lock_sha256: str | None = None,
) -> list[Problem]:
    """Periksa semua kiriman sebelum ada satu berkas pun ditulis."""
    problems: list[Problem] = []
    contributors = list(contributors)

    for contributor in contributors:
        manifest = contributor.manifest
        if not manifest:
            problems.append(
                Problem(
                    "warning",
                    f"{contributor.owner}: tanpa {MANIFEST}, "
                    "hash config tidak bisa diverifikasi",
                )
            )
        got = str(manifest.get("config_sha256") or "")
        if got and got != config_sha256:
            problems.append(
                Problem(
                    "error",
                    f"{contributor.owner}: config.yaml berbeda "
                    f"({got[:12]}… vs {config_sha256[:12]}…). "
                    "Samakan config dulu, lalu kirim ulang",
                )
            )
        lock = str(manifest.get("uv_lock_sha256") or "")
        if lock and uv_lock_sha256 and lock != uv_lock_sha256:
            problems.append(
                Problem(
                    "warning",
                    f"{contributor.owner}: uv.lock berbeda "
                    f"({lock[:12]}… vs {uv_lock_sha256[:12]}…); "
                    "versi pustaka mungkin tidak sama",
                )
            )

        films = contributor.table("films.csv")
        films = films[audition_passed(films)]
        faces = contributor.table("faces.csv")
        utterances = contributor.table("utterances.csv")

        ids = utterances["utterance_id"].astype(str)
        repeated = ids[ids.duplicated()].unique()
        if len(repeated):
            problems.append(
                Problem(
                    "error",
                    f"{contributor.owner}: utterance_id ganda: "
                    f"{', '.join(repeated[:5])}",
                )
            )

        face_ids = set(faces["utterance_id"].astype(str))
        utt_ids = set(ids)
        missing_faces = utt_ids - face_ids
        if missing_faces:
            problems.append(
                Problem(
                    "error",
                    f"{contributor.owner}: {len(missing_faces)} ujaran tanpa baris wajah "
                    f"(contoh {sorted(missing_faces)[0]}); jalankan 04_face_screen.py",
                )
            )
        extra_faces = face_ids - utt_ids
        if extra_faces:
            problems.append(
                Problem(
                    "warning",
                    f"{contributor.owner}: {len(extra_faces)} baris wajah "
                    "tanpa ujaran pasangannya, dibuang",
                )
            )

        for film_id in films["film_id"].astype(str):
            matrix = contributor.embeddings.get(film_id)
            if matrix is None:
                problems.append(
                    Problem("error", f"{contributor.owner}: {film_id}.npy tidak ada")
                )
                continue
            subset = faces[faces["film_id"].astype(str) == film_id]
            for detail in embedding_problems(subset, len(matrix)):
                problems.append(Problem("error", f"{contributor.owner}/{film_id}: {detail}"))
            recorded = contributor.film_config.get(film_id)
            if recorded and recorded != config_sha256:
                problems.append(
                    Problem(
                        "warning",
                        f"{contributor.owner}/{film_id}: embedding dibuat dengan config "
                        f"{recorded[:12]}…, bukan {config_sha256[:12]}…; "
                        "jalankan ulang 04_face_screen.py",
                    )
                )
            expected = contributor.film_rows.get(film_id)
            if expected is not None and expected != len(subset):
                problems.append(
                    Problem(
                        "warning",
                        f"{contributor.owner}/{film_id}: faces/{film_id}.csv "
                        f"{expected} baris, faces.csv {len(subset)} baris; "
                        "gabungan tidak sinkron dengan berkas per film",
                    )
                )
    return problems


def has_errors(problems: Iterable[Problem]) -> bool:
    return any(problem.level == "error" for problem in problems)


# -----------------------------------------------------------------------------
# Penggabungan
# -----------------------------------------------------------------------------


@dataclass
class Merged:
    films: pd.DataFrame
    utterances: pd.DataFrame
    faces: pd.DataFrame
    actors: pd.DataFrame
    scenes: pd.DataFrame
    reject_log: pd.DataFrame
    embeddings: dict[str, np.ndarray]
    dropped: list[str] = field(default_factory=list)


def neutralise_films(films: pd.DataFrame, owner: str) -> pd.DataFrame:
    """Buang kolom yang hanya berarti di laptop pemilik, lalu tandai pemiliknya."""
    machine = [column for column in MACHINE_COLUMNS if column in films.columns]
    frame = films.drop(columns=machine, errors="ignore").copy()
    frame.insert(0, "owner", owner)
    return frame


def _stack(contributors: list[Contributor], table: str) -> pd.DataFrame:
    frames = [frame for frame in (c.table(table) for c in contributors) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _filter_films(frame: pd.DataFrame, accepted: set[str]) -> pd.DataFrame:
    if frame.empty or "film_id" not in frame.columns:
        return frame
    return frame[frame["film_id"].astype(str).isin(accepted)].reset_index(drop=True)


def merge_contributors(contributors: list[Contributor]) -> Merged:
    """Gabungkan seluruh kiriman jadi satu katalog.

    Film yang tidak lolos audisi dibuang beserta seluruh turunannya, supaya
    yang dihitung di tahap QC hanya film yang memang layak.
    """
    films = pd.concat(
        [neutralise_films(c.table("films.csv"), c.owner) for c in contributors],
        ignore_index=True,
    )
    dropped: list[str] = []
    passed = audition_passed(films)
    if "pass_audition" in films.columns:
        dropped = sorted(films.loc[~passed, "film_id"].astype(str))
        films = films[passed].reset_index(drop=True)

    accepted = set(films["film_id"].astype(str))
    utterances = _filter_films(_stack(contributors, "utterances.csv"), accepted)
    faces = _filter_films(_stack(contributors, "faces.csv"), accepted)
    actors = _filter_films(_stack(contributors, "actors.csv"), accepted)
    scenes = _filter_films(_stack(contributors, "scenes.csv"), accepted)
    reject_log = _stack(contributors, "reject_log.csv")
    if not reject_log.empty:
        machine = [column for column in MACHINE_COLUMNS if column in reject_log.columns]
        reject_log = reject_log.drop(columns=machine, errors="ignore")

    # Baris wajah tanpa ujaran pasangan tidak bisa dipakai apa pun.
    if not faces.empty and not utterances.empty:
        faces = faces[faces["utterance_id"].isin(utterances["utterance_id"])]
        faces = faces.reset_index(drop=True)

    if not films.empty:
        films = films.sort_values("film_id", kind="stable").reset_index(drop=True)
    if not utterances.empty:
        order = ["film_id"] + (["index"] if "index" in utterances.columns else ["start_s"])
        utterances = utterances.sort_values(order, kind="stable").reset_index(drop=True)
    if not faces.empty:
        faces = faces.sort_values(["film_id", "start_s"], kind="stable").reset_index(drop=True)
    if not scenes.empty:
        scenes = scenes.sort_values(["film_id", "scene_id"], kind="stable").reset_index(drop=True)
    if not actors.empty:
        actors = actors.sort_values(["film_id", "actor_key"], kind="stable").reset_index(drop=True)

    embeddings = {
        film: matrix
        for contributor in contributors
        for film, matrix in contributor.embeddings.items()
        if film in accepted
    }
    return Merged(
        films=films,
        utterances=utterances,
        faces=faces,
        actors=actors,
        scenes=scenes,
        reject_log=reject_log,
        embeddings=embeddings,
        dropped=dropped,
    )


# -----------------------------------------------------------------------------
# Aktor lintas film
# -----------------------------------------------------------------------------


def actor_centroids(
    faces: pd.DataFrame, embeddings: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Rata-rata embedding per ``actor_key`` untuk membandingkan antar film."""
    if faces.empty or "actor_key" not in faces.columns:
        return {}
    centroids: dict[str, np.ndarray] = {}
    usable = faces[faces["actor_key"].astype(str).str.len() > 0]
    for key, group in usable.groupby("actor_key", sort=True):
        film_id = str(group["film_id"].iloc[0])
        matrix = embeddings.get(film_id)
        if matrix is None or not len(matrix):
            continue
        rows = group["emb_row"].astype(int).to_numpy()
        rows = rows[(rows >= 0) & (rows < len(matrix))]
        if not len(rows):
            continue
        vectors = matrix[rows]
        norms = np.linalg.norm(vectors, axis=1)
        vectors = vectors[norms > 0]
        if not len(vectors):
            continue
        centroid = vectors.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroids[str(key)] = (centroid / norm).astype(np.float32)
    return centroids


def cross_film_candidates(
    centroids: dict[str, np.ndarray], *, distance: float = 0.45
) -> pd.DataFrame:
    """Pasangan aktor dari film berbeda yang wajahnya mirip.

    Hanya usulan untuk diperiksa manual: aktor yang sama di dua film harus
    memakai ``actor_label`` yang persis sama, kalau tidak split C akan bocor.
    """
    keys = sorted(centroids)
    film_of = {key: key.rsplit("-A", 1)[0] for key in keys}
    rows = []
    for index, first in enumerate(keys):
        for second in keys[index + 1 :]:
            if film_of[first] == film_of[second]:
                continue
            similarity = float(np.dot(centroids[first], centroids[second]))
            gap = 1.0 - similarity
            if gap <= distance:
                rows.append(
                    {
                        "actor_a": first,
                        "actor_b": second,
                        "cosine_distance": round(gap, 4),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["actor_a", "actor_b", "cosine_distance"])
    return pd.DataFrame(rows).sort_values("cosine_distance", kind="stable").reset_index(drop=True)
