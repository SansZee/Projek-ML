"""Berkas pendamping tabular: penggabungan per film dan catatan meta.

Semua tahap berjalan per film, jadi hasil satu film tidak boleh menghapus hasil
film lain ketika skrip dijalankan ulang sebagian. Karena itu setiap skrip
menulis CSV per film lalu menggabungkannya dengan kunci ``film_id``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def merge_rows(path: Path, rows: list[dict], *, key: str = "film_id") -> pd.DataFrame:
    """Tulis ``rows`` ke ``path`` tanpa menghapus baris film lain.

    Baris lama untuk film yang sama dibuang, karena hasil terbaru film itulah
    yang berlaku.
    """
    frame = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        if key in old.columns and key in frame.columns:
            fresh = set(frame[key].astype(str))
            old = old[~old[key].astype(str).isin(fresh)]
        frame = pd.concat([old, frame], ignore_index=True)
    if key in frame.columns:
        frame = frame.sort_values(key, kind="stable").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def load_meta(path: Path) -> dict[str, dict]:
    """Baca catatan meta per film; berkas rusak dianggap kosong."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    entries = data.get("films") if isinstance(data, dict) else data
    return {str(entry["film_id"]): entry for entry in entries or [] if "film_id" in entry}


def save_meta(path: Path, entries: dict[str, dict[str, Any]], *, generated_by: str) -> None:
    films = []
    for key in sorted(entries):
        entry = dict(entries[key])
        entry.setdefault("film_id", key)
        films.append(entry)
    payload = {"generated_by": generated_by, "films": films}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
