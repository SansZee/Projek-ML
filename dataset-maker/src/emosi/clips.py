"""Bungkus dan gabung klip antar-laptop (``10b_klip.py``).

``10_cut.py`` berjalan di laptop pemilik film, jadi hasilnya terpencar: setiap
orang hanya bisa memotong film yang ada di sana. Modul ini menutup dua celah:

  - **pack** membungkus klip yang sudah jadi menjadi satu zip siap kirim
    beserta ``clips.csv`` yang sudah disaring;
  - **merge** memeriksa zip dari beberapa orang, mengekstraknya ke satu
    ``media/``, lalu menyusun ``data/clips.csv`` final dan melaporkan klip
    yang belum dikirim siapa pun.

Pemeriksaan sengaja ketat: lebih baik berhenti dengan pesan jelas daripada
menggabungkan katalog yang diam-diam tumpang tindih.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

CATALOG = "clips.csv"
MANIFEST = "manifest.json"
OK_STATUS = ("ok", "ada")
MISSING_STATUS = "hilang"
CATALOG_COLUMNS = [
    "clip_id",
    "utterance_id",
    "film_id",
    "label",
    "confidence",
    "duration_s",
    "clip_path",
    "audio_path",
    "face_path",
    "clip_bytes",
    "audio_bytes",
    "face_bytes",
    "status",
]
MEDIA_COLUMNS = ("clip_path", "audio_path", "face_path")
SIZE_COLUMNS = ("clip_bytes", "audio_bytes", "face_bytes")


class ClipError(RuntimeError):
    """Zip klip tidak lengkap atau tidak konsisten."""


def ok_rows(catalog: pd.DataFrame) -> pd.DataFrame:
    """Baris yang benar-benar punya berkas di laptop ini."""
    rows = catalog[catalog["status"].isin(OK_STATUS)].copy()
    return rows


def pack(
    catalog: pd.DataFrame,
    media_dir: Path,
    dest: Path,
    *,
    owner: str,
    films: set[str] | None = None,
) -> tuple[int, int]:
    """Tulis zip berisi klip ``ok``/``ada`` dan katalognya yang sudah disaring.

    Mengembalikan ``(jumlah klip, ukuran zip byte)``.
    """
    rows = ok_rows(catalog)
    if films is not None:
        rows = rows[rows["film_id"].astype(str).isin({str(value) for value in films})]
    if rows.empty:
        raise ClipError(
            "tidak ada klip berstatus ok/ada. Jalankan 10_cut.py dulu di laptop ini."
        )

    media_dir = Path(media_dir)
    entries: list[tuple[str, Path]] = []
    for row in rows.itertuples():
        for column, size_column in zip(MEDIA_COLUMNS, SIZE_COLUMNS, strict=True):
            relative = str(getattr(row, column) or "").strip()
            if not relative:
                continue
            path = media_dir / relative
            if not path.exists():
                raise ClipError(f"{row.clip_id}: berkas hilang: {relative}")
            expected = int(getattr(row, size_column) or 0)
            if expected and path.stat().st_size != expected:
                raise ClipError(
                    f"{row.clip_id}: ukuran {relative} tidak cocok dengan katalog"
                )
            entries.append((relative, path))

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "owner": str(owner),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "clips": int(len(rows)),
        "bytes": int(sum(path.stat().st_size for _, path in entries)),
    }
    buffer = io.StringIO()
    rows[CATALOG_COLUMNS].to_csv(buffer, index=False)

    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED) as bundle:
        for relative, path in entries:
            bundle.write(path, arcname=relative)
        bundle.writestr(CATALOG, buffer.getvalue())
        bundle.writestr(MANIFEST, json.dumps(manifest, indent=2))
    return len(rows), dest.stat().st_size


def _read_catalog(bundle: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with bundle.open(CATALOG) as handle:
        frame = pd.read_csv(handle)
    missing = [column for column in CATALOG_COLUMNS if column not in frame.columns]
    if missing:
        raise ClipError(f"{name}: clips.csv tidak punya kolom: {', '.join(missing)}")
    return frame


def _manifest(bundle: zipfile.ZipFile) -> dict:
    try:
        return json.loads(bundle.read(MANIFEST))
    except KeyError:
        return {}


def merge_bundles(
    paths: list[Path],
    media_dir: Path,
    *,
    films: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Ekstrak klip dari beberapa zip dan gabungkan katalognya.

    ``films`` (kolom ``film_id`` dan ``owner``) dipakai untuk memastikan setiap
    pengirim hanya membawa film miliknya. Mengembalikan katalog gabungan dan
    ringkasan per zip.
    """
    media_dir = Path(media_dir)
    film_owner = (
        dict(zip(films["film_id"].astype(str), films["owner"].astype(str), strict=False))
        if films is not None and {"film_id", "owner"} <= set(films.columns)
        else {}
    )

    frames: list[pd.DataFrame] = []
    report: list[dict] = []
    seen: dict[str, str] = {}

    for path in sorted(Path(item) for item in paths):
        with zipfile.ZipFile(path) as bundle:
            names = set(bundle.namelist())
            if CATALOG not in names:
                raise ClipError(f"{path.name}: tidak ada {CATALOG} di dalam zip")
            catalog = _read_catalog(bundle, path.name)
            owner = str(_manifest(bundle).get("owner") or "")
            rows = ok_rows(catalog)
            if rows.empty:
                raise ClipError(f"{path.name}: tidak ada klip ok/ada")

            for row in rows.itertuples():
                clip_id = str(row.clip_id)
                if clip_id in seen:
                    raise ClipError(
                        f"{clip_id} ada di dua zip: {seen[clip_id]} dan {path.name}"
                    )
                seen[clip_id] = path.name
                expected_owner = film_owner.get(str(row.film_id))
                if owner and expected_owner and expected_owner != owner:
                    raise ClipError(
                        f"{path.name}: {clip_id} milik film {row.film_id} "
                        f"yang seharusnya dikirim {expected_owner}"
                    )
                for column, size_column in zip(
                    MEDIA_COLUMNS, SIZE_COLUMNS, strict=True
                ):
                    relative = str(getattr(row, column) or "").strip()
                    if not relative:
                        continue
                    if relative not in names:
                        raise ClipError(f"{path.name}: {relative} tidak ada di zip")
                    expected = int(getattr(row, size_column) or 0)
                    actual = bundle.getinfo(relative).file_size
                    if expected and actual != expected:
                        raise ClipError(
                            f"{path.name}: {relative} {actual} byte, katalog {expected}"
                        )
                    target = media_dir / relative
                    if target.exists() and target.stat().st_size == actual:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(relative) as source, target.open("wb") as sink:
                        sink.write(source.read())

            frames.append(rows[CATALOG_COLUMNS])
            report.append(
                {
                    "zip": path.name,
                    "owner": owner or "?",
                    "klip": int(len(rows)),
                    "mb": round(path.stat().st_size / 1048576, 1),
                }
            )

    if not frames:
        raise ClipError("tidak ada zip klip untuk digabungkan")
    merged = pd.concat(frames, ignore_index=True)
    duplicates = merged["clip_id"][merged["clip_id"].duplicated()]
    if not duplicates.empty:
        raise ClipError("clip ganda: " + ", ".join(sorted(set(duplicates))))
    return merged, report


def final_catalog(
    merged: pd.DataFrame, selected: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Katalog lengkap 3.300 baris; yang belum dikirim ditandai ``hilang``."""
    present = set(merged["clip_id"].astype(str))
    absent = selected[~selected["clip_id"].astype(str).isin(present)].copy()
    missing = pd.DataFrame(
        {
            "clip_id": absent["clip_id"].astype(str),
            "utterance_id": absent["utterance_id"].astype(str),
            "film_id": absent["film_id"].astype(str),
            "label": absent["label"].astype(str),
            "confidence": absent["confidence"],
            "duration_s": absent["duration_s"],
            "clip_path": "",
            "audio_path": "",
            "face_path": "",
            "clip_bytes": 0,
            "audio_bytes": 0,
            "face_bytes": 0,
            "status": MISSING_STATUS,
        }
    )
    catalog = pd.concat(
        [merged[CATALOG_COLUMNS], missing[CATALOG_COLUMNS]], ignore_index=True
    )
    catalog = catalog.sort_values("clip_id", kind="stable").reset_index(drop=True)
    return catalog, missing
