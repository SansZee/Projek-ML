#!/usr/bin/env python3
"""02_scenes.py — deteksi potongan gambar dan pengelompokannya jadi adegan.

Untuk setiap film yang lolos audisi:

  1. filter ``scdet`` ffmpeg menandai setiap cut gambar (satu kali dekode)
  2. cut digabung menjadi adegan 2-3 menit yang batasnya jatuh di cut
  3. setiap adegan dicatat jumlah shot dan dialognya

Adegan adalah **satuan anotasi**, bukan satuan dataset. Anotator menonton satu
adegan utuh lalu melabeli semua ujaran di dalamnya, karena emosi butuh konteks
cerita. Klip 4 detik akan menghasilkan label yang menebak-nebak.

Keluaran:

  data/scenes/<film_id>.csv   scene_id, waktu, timecode, shot, cue, dialog
  data/scenes.csv             gabungan semua film
  data/scenes/scenes_meta.json  jumlah cut, ambang, hash config per film

Contoh:

  uv run python scripts/02_scenes.py --film ABT13 --seconds 600   # uji cepat
  uv run python scripts/02_scenes.py --force                      # semua film

Perkiraan waktu: ~10 menit per film (dekode CPU 1080p HEVC). Deteksi sengaja
tidak memakai GPU: hasilnya sama dan tidak bergantung pada driver.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from emosi import scenes as scene_lib
from emosi.config import Config, load_config
from emosi.media import require_tools
from emosi.subtitle import parse_srt
from emosi.tables import load_meta, merge_rows, save_meta


# --------------------------------------------------------------------------- #
# satu film
# --------------------------------------------------------------------------- #
def scene_rows(
    film_id: str,
    spans: list[tuple[float, float]],
    cuts: list[float],
    cues,
) -> list[dict]:
    rows: list[dict] = []
    for index, (start, end) in enumerate(spans, start=1):
        n_cues, dialogue = scene_lib.dialogue_stats(cues, start, end)
        duration = end - start
        rows.append(
            {
                "film_id": film_id,
                "scene_id": f"{film_id}-S{index:03d}",
                "index": index,
                "start_s": round(start, 3),
                "end_s": round(end, 3),
                "duration_s": round(duration, 3),
                "start_tc": scene_lib.timecode(start),
                "end_tc": scene_lib.timecode(end),
                "n_shots": scene_lib.count_shots(cuts, start, end),
                "n_cues": n_cues,
                "dialogue_s": dialogue,
                "dialogue_ratio": round(dialogue / duration, 3) if duration > 0 else 0.0,
            }
        )
    return rows


def window_for(film: pd.Series, cfg: Config) -> tuple[float, float]:
    return scene_lib.annotation_window(
        float(film["duration_s"]),
        _number(film.get("srt_first_s")),
        _number(film.get("srt_last_s")),
        head_pad_s=float(cfg.get("scenes.head_pad_s")),
        tail_pad_s=float(cfg.get("scenes.tail_pad_s")),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--films-csv", default=None, help="default: <data_dir>/films.csv")
    parser.add_argument("--film", action="append", default=None, help="batasi ke film_id ini")
    parser.add_argument("--limit", type=int, default=None, help="batasi jumlah film")
    parser.add_argument("--force", action="store_true", help="timpa hasil yang sudah ada")
    parser.add_argument(
        "--seconds", type=float, default=None, help="potong N detik pertama, untuk uji cepat"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data_dir = cfg.resolve("paths.data_dir")
    scenes_dir = data_dir / "scenes"
    combined_csv = data_dir / "scenes.csv"

    smoke = args.seconds is not None
    if smoke:
        scenes_dir = scenes_dir / "_smoke"
        combined_csv = scenes_dir / "scenes.csv"

    films_csv = Path(args.films_csv) if args.films_csv else data_dir / "films.csv"
    if not films_csv.exists():
        print(f"{films_csv} tidak ada. Jalankan scripts/00_inventory.py dulu.")
        return 1

    films = pd.read_csv(films_csv)
    if "pass_audition" in films.columns:
        films = films[films["pass_audition"].astype(bool)]
    if args.film:
        films = films[films["film_id"].isin(args.film)]
    if args.limit:
        films = films.head(args.limit)
    if films.empty:
        print("Tidak ada film yang dipilih.")
        return 1

    require_tools("ffmpeg")

    threshold = float(cfg.get("scenes.scdet_threshold"))
    width = int(cfg.get("scenes.scale_width"))
    target_s = float(cfg.get("scenes.target_s"))
    min_s = float(cfg.get("scenes.min_s"))
    max_s = float(cfg.get("scenes.max_s"))

    print(
        f"scdet ambang {threshold:g} · lebar {width} px · adegan "
        f"{target_s:.0f} s ({min_s:.0f}-{max_s:.0f} s)"
    )

    meta = load_meta(scenes_dir / "scenes_meta.json")
    rows: list[dict] = []

    for _, film in films.iterrows():
        film_id = str(film["film_id"])
        video_path = Path(str(film["path"]))
        out_path = scenes_dir / f"{film_id}.csv"

        print()
        print(f"── {film_id}  {film.get('title', '')}")

        if out_path.exists() and not args.force:
            print(f"   {out_path.name} sudah ada, dilewati (pakai --force untuk ulang)")
            continue
        if not video_path.exists():
            print(f"   berkas tidak ditemukan: {video_path}")
            print("   hard disk eksternal belum dipasang? dilewati.")
            continue

        start_s, end_s = window_for(film, cfg)
        if smoke:
            end_s = min(end_s, start_s + args.seconds)
        print(
            f"   rentang {scene_lib.timecode(start_s)} → {scene_lib.timecode(end_s)} "
            f"({end_s - start_s:.0f} s)"
        )

        bar = tqdm(total=100, unit="%", desc="   deteksi cut", leave=False, ncols=72)

        def progress(fraction: float, _bar: tqdm = bar) -> None:
            target = int(fraction * 100)
            if target > _bar.n:
                _bar.update(target - _bar.n)

        began = time.monotonic()
        cuts = scene_lib.detect_cuts(
            video_path,
            start_s=start_s,
            end_s=end_s,
            threshold=threshold,
            width=width,
            progress=progress,
        )
        elapsed = time.monotonic() - began
        bar.close()

        if not cuts:
            print("   PERINGATAN: tidak ada cut terdeteksi; adegan dipotong di titik target")

        spans = scene_lib.build_scenes(
            cuts, start_s, end_s, target_s=target_s, min_s=min_s, max_s=max_s
        )

        srt_raw = film.get("srt_path")
        cues: list = []
        if pd.notna(srt_raw):
            srt_path = Path(str(srt_raw))
            if srt_path.exists():
                cues, errors = parse_srt(srt_path)
                if errors:
                    print(f"   SRT: {errors} baris rusak diabaikan")
            else:
                print(f"   SRT tidak ditemukan, jumlah cue tidak dihitung: {srt_path}")

        film_rows = scene_rows(film_id, spans, cuts, cues)
        scenes_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(film_rows).to_csv(out_path, index=False)
        rows.extend(film_rows)

        durations = [span[1] - span[0] for span in spans]
        median = sorted(durations)[len(durations) // 2] if durations else 0.0
        n_cues = sum(row["n_cues"] for row in film_rows)
        dialogue = sum(row["dialogue_s"] for row in film_rows)
        covered = max(1e-9, end_s - start_s)
        print(
            f"   cut {len(cuts)} · adegan {len(spans)} · median "
            f"{scene_lib.timecode(median)} · cue {n_cues} "
            f"({dialogue / covered:.0%} durasi)"
        )
        print(f"   {elapsed:.0f} s deteksi → {out_path}")

        if not smoke:
            meta[film_id] = {
                "film_id": film_id,
                "window_start_s": start_s,
                "window_end_s": end_s,
                "n_cuts": len(cuts),
                "n_scenes": len(spans),
                "scdet_threshold": threshold,
                "scale_width": width,
                "scene_target_s": target_s,
                "scene_min_s": min_s,
                "scene_max_s": max_s,
                "runtime_s": round(elapsed, 1),
                "config_sha256": cfg.sha256,
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            save_meta(
                scenes_dir / "scenes_meta.json", meta, generated_by="scripts/02_scenes.py"
            )

    print()
    if rows:
        frame = merge_rows(combined_csv, rows)
        print_summary(frame)
        print(f"adegan → {combined_csv}")
    elif combined_csv.exists():
        print_summary(pd.read_csv(combined_csv))
        print(f"adegan → {combined_csv}")
    else:
        print("Tidak ada adegan yang dihasilkan.")
    return 0


def print_summary(frame: pd.DataFrame) -> None:
    print("=" * 82)
    print(
        f"{'film':<9}{'adegan':>7}{'shot':>8}{'cue':>7}"
        f"{'dialog':>8}{'median':>9}{'mulai':>11}{'selesai':>11}"
    )
    print("-" * 82)
    for film_id, group in frame.groupby("film_id", sort=True):
        durations = group["duration_s"].astype(float)
        median = durations.median()
        covered = float(durations.sum())
        dialogue = float(group["dialogue_s"].sum())
        print(
            f"{film_id:<9}{len(group):>7}{int(group['n_shots'].sum()):>8}"
            f"{int(group['n_cues'].sum()):>7}"
            f"{(dialogue / covered if covered else 0):>8.0%}"
            f"{scene_lib.timecode(median):>9}"
            f"{scene_lib.timecode(group['start_s'].min()):>11}"
            f"{scene_lib.timecode(group['end_s'].max()):>11}"
        )
    print("=" * 82)


def _number(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


if __name__ == "__main__":
    sys.exit(main())
