#!/usr/bin/env python3
"""01_asr.py — transkripsi film dan pemeriksaan sinkronisasi subtitle.

Untuk setiap film yang lolos audisi:

  1. track audio diekstrak ke FLAC mono 16 kHz (sekali, dipakai ulang tahap lain)
  2. faster-whisper menulis transkrip dengan timestamp per kata
  3. teks subtitle dibandingkan dengan transkrip: WER, geseran waktu, cakupan

Keluaran:

  media/audio/<film_id>.flac      audio mono untuk ASR dan fitur suara
  data/asr/<film_id>.json         segmen + kata + timestamp
  data/asr/asr_meta.json          model, perangkat, durasi proses per film
  data/asr/srt_check.csv          bukti subtitle sinkron dengan berkas film

Contoh:

  uv run python scripts/01_asr.py --check-only --film ABT13
  uv run python scripts/01_asr.py --film ABT13 --seconds 180      # uji cepat
  uv run python scripts/01_asr.py                                 # semua film

Perkiraan waktu di RTX 4050 (large-v3, int8_float16): 15-30 menit per film.
Di CPU: 4-8 jam per film, dan itu bukan pilihan yang wajar untuk 20 film.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from emosi import asr
from emosi.config import Config, load_config
from emosi.media import require_tools
from emosi.subtitle import parse_srt
from emosi.tables import load_meta, merge_rows, save_meta


# --------------------------------------------------------------------------- #
# satu film
# --------------------------------------------------------------------------- #
def check_row_for(cfg: Config, film_id: str, srt_path: Path, asr_json: Path) -> dict:
    settings = asr.CheckSettings(
        block_s=float(cfg.get("asr.check.block_s")),
        max_blocks=int(cfg.get("asr.check.max_blocks")),
        onset_window_s=float(cfg.get("asr.check.onset_window_s")),
        pad_s=float(cfg.get("asr.check.pad_s")),
        wer_ok=float(cfg.get("asr.check.wer_ok")),
        wer_poor=float(cfg.get("asr.check.wer_poor")),
    )
    cues, _errors = parse_srt(srt_path)
    payload = asr.load_asr(asr_json)

    # Transkrip yang sengaja dipotong (uji cepat) tidak boleh diadu dengan cue
    # dari akhir film: WER-nya akan tinggi bukan karena subtitle salah.
    segments = payload.get("segments") or []
    asr_end = max((float(segment.get("end") or 0.0) for segment in segments), default=0.0)
    if asr_end > 0:
        cues = [cue for cue in cues if cue.start <= asr_end + 1.0]

    return asr.check_sync(film_id, cues, payload, settings)


def versions() -> dict:
    info: dict = {"python": platform.python_version()}
    for module in ("faster_whisper", "ctranslate2", "numpy"):
        try:
            info[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001 - hanya catatan, bukan kegagalan
            info[module] = None
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--films-csv", default=None, help="default: <data_dir>/films.csv")
    parser.add_argument("--film", action="append", default=None, help="batasi ke film_id ini")
    parser.add_argument("--limit", type=int, default=None, help="batasi jumlah film")
    parser.add_argument("--force", action="store_true", help="timpa transkrip yang sudah ada")
    parser.add_argument(
        "--check-only", action="store_true", help="hanya cek SRT, tanpa transkripsi"
    )
    parser.add_argument("--model", default=None, help="timpa asr.model")
    parser.add_argument("--device", default=None, help="auto | cuda | cpu")
    parser.add_argument(
        "--seconds", type=float, default=None, help="potong N detik pertama, untuk uji cepat"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data_dir = cfg.resolve("paths.data_dir")
    media_dir = cfg.resolve("paths.media_dir")
    asr_dir = data_dir / "asr"
    audio_dir = media_dir / "audio"

    smoke = args.seconds is not None
    if smoke:
        asr_dir = asr_dir / "_smoke"
        audio_dir = audio_dir / "_smoke"

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

    if not args.check_only:
        require_tools("ffmpeg", "ffprobe")
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            print("faster-whisper belum terpasang. Jalankan: uv sync --extra asr")
            return 1

    model_name = args.model or cfg.get("asr.model")
    requested_device = args.device or cfg.get("asr.device")
    device = asr.resolve_device(requested_device)
    compute_type = str(
        cfg.get("asr.compute_type") if device == "cuda" else cfg.get("asr.compute_type_cpu")
    )
    settings = cfg.raw["asr"]

    print(f"model {model_name} · device {device} · compute {compute_type}")
    if device == "cpu" and not args.check_only:
        print("PERINGATAN: tanpa GPU, satu film bisa memakan 4-8 jam.")

    model = None
    meta = load_meta(asr_dir / "asr_meta.json")
    checks: list[dict] = []

    for _, film in films.iterrows():
        film_id = str(film["film_id"])
        video_path = Path(str(film["path"]))
        srt_raw = film.get("srt_path")
        srt_path = Path(str(srt_raw)) if pd.notna(srt_raw) else None

        asr_json = asr_dir / f"{film_id}.json"
        print()
        print(f"── {film_id}  {film.get('title', '')}")

        if not video_path.exists() and not args.check_only:
            print(f"   berkas tidak ditemukan: {video_path}")
            print("   hard disk eksternal belum dipasang? dilewati.")
            continue

        # ---- transkripsi ----
        if args.check_only:
            if not asr_json.exists():
                print(f"   {asr_json} belum ada, dilewati")
                continue
        elif asr_json.exists() and not args.force:
            print(f"   {asr_json.name} sudah ada, dilewati (pakai --force untuk ulang)")
        else:
            audio_path = asr.extract_speech_audio(
                video_path,
                audio_dir / f"{film_id}.flac",
                limit_s=args.seconds,
                force=args.force,
            )
            audio_mb = audio_path.stat().st_size / 1024**2
            print(f"   audio {audio_path.name} ({audio_mb:.0f} MB)")

            if model is None:
                model = asr.load_model(model_name, device, compute_type, media_dir / "models")

            bar = tqdm(total=100, unit="%", desc="   transkripsi", leave=False, ncols=72)

            def progress(fraction: float, _bar: tqdm = bar) -> None:
                target = int(fraction * 100)
                if target > _bar.n:
                    _bar.update(target - _bar.n)

            result = asr.transcribe(
                model,
                audio_path,
                settings,
                partial_path=asr_json.with_suffix(".partial.jsonl"),
                duration_s=float(film["duration_s"]) if not smoke else args.seconds,
                progress=progress,
            )
            bar.close()

            payload = {
                "film_id": film_id,
                "title": film.get("title"),
                "model": model_name,
                "device": device,
                "compute_type": compute_type,
                "language": result.language,
                "language_probability": result.language_probability,
                "duration_s": result.audio_duration_s,
                "truncated_s": args.seconds,
                "n_segments": len(result.segments),
                "n_words": result.n_words,
                "segments": result.segments,
            }
            asr_dir.mkdir(parents=True, exist_ok=True)
            asr_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            asr_json.with_suffix(".partial.jsonl").unlink(missing_ok=True)

            suspicious = sum(1 for s in result.segments if s["suspicious"])
            print(
                f"   {result.n_words} kata · {len(result.segments)} segmen "
                f"({suspicious} dicurigai) · {result.runtime_s / 60:.1f} menit "
                f"· {result.speedup:.1f}x realtime"
            )
            if not smoke:
                meta[film_id] = {
                    "film_id": film_id,
                    "model": model_name,
                    "device": device,
                    "compute_type": compute_type,
                    "language": result.language,
                    "language_probability": result.language_probability,
                    "audio_s": result.audio_duration_s,
                    "runtime_s": result.runtime_s,
                    "speedup": result.speedup,
                    "n_segments": len(result.segments),
                    "n_words": result.n_words,
                    "suspicious_segments": suspicious,
                    "audio_path": str(audio_path),
                    "config_sha256": cfg.sha256,
                    "versions": versions(),
                    "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
                save_meta(
                    asr_dir / "asr_meta.json", meta, generated_by="scripts/01_asr.py"
                )

        # ---- pemeriksaan subtitle ----
        if srt_path is None or not srt_path.exists():
            print("   tidak ada SRT, cek sinkronisasi dilewati")
            continue
        row = check_row_for(cfg, film_id, srt_path, asr_json)
        checks.append(row)
        print(
            f"   SRT: WER {_f(row['wer'])} · geseran {_fmt_signed(row['offset'])} s "
            f"· drift {_f(row['drift_s'])} s · cakupan {_f(row['coverage'])}"
            f" → {row['verdict']}"
        )
        if row["notes"]:
            print(f"   catatan: {row['notes']}")

    # ---- ringkasan ----
    print()
    if checks:
        frame = merge_rows(asr_dir / "srt_check.csv", checks)
        print("=" * 84)
        print(
            f"{'film':<9}{'cue':>6}{'kata SRT':>10}{'kata ASR':>10}"
            f"{'WER':>7}{'geser':>8}{'drift':>8}{'cakup':>7}  status"
        )
        print("-" * 84)
        for _, row in frame.iterrows():
            print(
                f"{row['film_id']:<9}{int(row['n_cues']):>6}{int(row['srt_words']):>10}"
                f"{int(row['asr_words']):>10}{_f(row['wer']):>7}"
                f"{_fmt_signed(row['offset']):>8}{_f(row['drift_s']):>8}"
                f"{_f(row['coverage']):>7}  {row['verdict']}"
            )
        print("=" * 84)
        print(f"bukti sinkronisasi → {asr_dir / 'srt_check.csv'}")

    done = [key for key in meta if (asr_dir / f"{key}.json").exists()]
    if done:
        print(f"transkrip selesai: {', '.join(sorted(done))} → {asr_dir}")
    return 0


def _f(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.{digits}f}"


def _fmt_signed(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:+.2f}"


if __name__ == "__main__":
    sys.exit(main())
