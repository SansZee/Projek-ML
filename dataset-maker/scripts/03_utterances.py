#!/usr/bin/env python3
"""03_utterances.py — bentuk ujaran 2-10 detik dari subtitle + timestamp ASR.

Untuk setiap film:

  1. cue subtitle dikelompokkan per adegan (dari ``scenes.csv``)
  2. cue yang berdekatan digabung sampai kalimatnya utuh dan cukup panjang
  3. awal/akhir ujaran diambil dari kata Whisper, teksnya dari subtitle

Keluaran:

  data/utterances/<film_id>.csv   satu baris per ujaran
  data/utterances.csv             gabungan semua film
  data/utterances/utterances_meta.json  hasil per film + hash config

Contoh:

  uv run python scripts/03_utterances.py --film ABT13 --force
  uv run python scripts/03_utterances.py --limit 1

Tidak ada dekode video di tahap ini, jadi seluruh film selesai dalam hitungan
detik. Durasi ujaran yang tetap di bawah 2 detik tidak dibuang di sini: QC
(05_qc.py) yang memutuskan, supaya keputusannya tercatat.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from emosi import asr
from emosi.config import Config, load_config
from emosi.subtitle import parse_srt
from emosi.tables import load_meta, merge_rows, save_meta
from emosi.utterance import MergeSettings, build_utterances


def merge_settings(cfg: Config) -> MergeSettings:
    return MergeSettings(
        merge_gap_s=float(cfg.get("utterance.merge_gap_s")),
        short_gap_s=float(cfg.get("utterance.short_gap_s")),
        dur_min_s=float(cfg.get("utterance.dur_min_s")),
        dur_max_s=float(cfg.get("utterance.dur_max_s")),
        pad_head_s=float(cfg.get("utterance.pad_head_s")),
        pad_tail_s=float(cfg.get("utterance.pad_tail_s")),
        asr_head_tol_s=float(cfg.get("utterance.asr_head_tol_s")),
        asr_tail_tol_s=float(cfg.get("utterance.asr_tail_tol_s")),
        trim_overlap_max_s=float(cfg.get("utterance.trim_overlap_max_s")),
        terminal_punct=tuple(cfg.get("utterance.terminal_punct")),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--films-csv", default=None, help="default: <data_dir>/films.csv")
    parser.add_argument("--scenes-csv", default=None, help="default: <data_dir>/scenes.csv")
    parser.add_argument("--film", action="append", default=None, help="batasi ke film_id ini")
    parser.add_argument("--limit", type=int, default=None, help="batasi jumlah film")
    parser.add_argument("--force", action="store_true", help="timpa hasil yang sudah ada")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data_dir = cfg.resolve("paths.data_dir")
    asr_dir = data_dir / "asr"
    utterance_dir = data_dir / "utterances"
    combined_csv = data_dir / "utterances.csv"

    films_csv = Path(args.films_csv) if args.films_csv else data_dir / "films.csv"
    scenes_csv = Path(args.scenes_csv) if args.scenes_csv else data_dir / "scenes.csv"
    for path, hint in ((films_csv, "00_inventory.py"), (scenes_csv, "02_scenes.py")):
        if not path.exists():
            print(f"{path} tidak ada. Jalankan scripts/{hint} dulu.")
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

    scenes = pd.read_csv(scenes_csv)
    settings = merge_settings(cfg)
    meta = load_meta(utterance_dir / "utterances_meta.json")
    rows: list[dict] = []

    print(
        f"ujaran {settings.dur_min_s:.0f}-{settings.dur_max_s:.0f} s · "
        f"gabung jarak {settings.merge_gap_s:.1f} s "
        f"({settings.short_gap_s:.1f} s bila masih pendek)"
    )

    for _, film in films.iterrows():
        film_id = str(film["film_id"])
        out_path = utterance_dir / f"{film_id}.csv"
        print()
        print(f"── {film_id}  {film.get('title', '')}")

        if out_path.exists() and not args.force:
            print(f"   {out_path.name} sudah ada, dilewati (pakai --force untuk ulang)")
            continue

        film_scenes = scenes[scenes["film_id"] == film_id].sort_values("index")
        if film_scenes.empty:
            print(f"   tidak ada adegan untuk {film_id} di {scenes_csv.name}, dilewati")
            continue

        asr_json = asr_dir / f"{film_id}.json"
        if not asr_json.exists():
            print(f"   {asr_json} belum ada, jalankan scripts/01_asr.py dulu")
            continue

        srt_raw = film.get("srt_path")
        srt_path = Path(str(srt_raw)) if pd.notna(srt_raw) else None
        if srt_path is None or not srt_path.exists():
            print(f"   SRT tidak ditemukan, dilewati: {srt_raw}")
            continue

        cues, errors = parse_srt(srt_path)
        index = asr.WordIndex.from_segments(asr.load_asr(asr_json).get("segments") or [])
        triples = [
            (str(row.scene_id), float(row.start_s), float(row.end_s))
            for row in film_scenes.itertuples()
        ]

        utterances = build_utterances(film_id, cues, triples, index, settings)
        if not utterances:
            print("   tidak ada ujaran yang terbentuk")
            continue

        utterance_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(asdict(item) for item in utterances).to_csv(out_path, index=False)
        rows.extend(asdict(item) for item in utterances)

        durations = [item.duration_s for item in utterances]
        asr_timed = sum(1 for item in utterances if item.timing == "asr")
        short = sum(1 for value in durations if value < settings.dur_min_s)
        single = sum(1 for item in utterances if item.n_words <= 1)
        median = sorted(durations)[len(durations) // 2]
        print(
            f"   adegan {len(film_scenes)} · ujaran {len(utterances)} · "
            f"waktu dari ASR {asr_timed / len(utterances):.0%} · median {median:.2f} s"
        )
        if short or single:
            print(
                f"   di bawah {settings.dur_min_s:.0f} s: {short} "
                f"({short / len(utterances):.0%}) · satu kata: {single}"
            )
        if errors:
            print(f"   SRT: {errors} blok rusak diabaikan")
        print(f"   → {out_path}")

        meta[film_id] = {
            "film_id": film_id,
            "n_scenes": len(film_scenes),
            "n_utterances": len(utterances),
            "asr_timed": asr_timed,
            "asr_share": round(asr_timed / len(utterances), 3),
            "cue_timed": len(utterances) - asr_timed,
            "median_duration_s": round(median, 3),
            "below_min": short,
            "single_word": single,
            "srt_parse_errors": errors,
            "config_sha256": cfg.sha256,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        save_meta(
            utterance_dir / "utterances_meta.json",
            meta,
            generated_by="scripts/03_utterances.py",
        )

    print()
    if rows:
        frame = merge_rows(combined_csv, rows)
        print_summary(frame)
        print(f"ujaran → {combined_csv}")
    elif combined_csv.exists():
        print_summary(pd.read_csv(combined_csv))
        print(f"ujaran → {combined_csv}")
    else:
        print("Tidak ada ujaran yang dihasilkan.")
    return 0


def print_summary(frame: pd.DataFrame) -> None:
    covered = max(1e-9, float(frame["duration_s"].sum()))
    print("=" * 76)
    print(
        f"{'film':<9}{'adegan':>7}{'ujaran':>8}{'ASR':>7}"
        f"{'median':>9}{'<2s':>8}{'1 kata':>8}"
    )
    print("-" * 76)
    for film_id, group in frame.groupby("film_id", sort=True):
        durations = group["duration_s"].astype(float)
        asr_share = (group["timing"] == "asr").mean()
        print(
            f"{film_id:<9}{group['scene_id'].nunique():>7}{len(group):>8}"
            f"{asr_share:>7.0%}{durations.median():>9.2f}"
            f"{int((durations < 2).sum()):>8}{int((group['n_words'] <= 1).sum()):>8}"
        )
    print("=" * 76)
    print(
        f"total {len(frame)} ujaran · {frame['scene_id'].nunique()} adegan · "
        f"{float(frame['n_words'].sum()):.0f} kata · dialog {covered / 60:.0f} menit"
    )


if __name__ == "__main__":
    sys.exit(main())
