#!/usr/bin/env python3
"""00_inventory.py — inventaris film dan audisi otomatis (protokol Bab 5).

Mengukur, untuk setiap film:
  - metadata wadah dan stream (ffprobe)
  - statistik subtitle: jumlah cue, kata, WPM, celah, watermark
  - SNR pada jendela probe 3 menit
  - face ratio pada jendela probe yang sama

Keluaran:
  data/films.csv        satu baris per film, termasuk pass_audition
  data/reject_log.csv   film yang gagal beserta alasannya

Contoh:
  uv run python scripts/00_inventory.py
  uv run python scripts/00_inventory.py --limit 2 --skip-faces
  uv run python scripts/00_inventory.py --force
"""

from __future__ import annotations

import argparse
import difflib
import re
import statistics
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from emosi import audio, video
from emosi.config import Config, load_config
from emosi.media import SUB_EXT, VIDEO_EXT, MediaInfo, probe_media, require_tools
from emosi.subtitle import SubtitleStats, parse_srt, subtitle_stats

YEAR = re.compile(r"(?:19|20)\d{2}")
RELEASE_NOISE = re.compile(
    r"\b(1080p|720p|2160p|bluray|blu-ray|webrip|web-dl|x265|x264|hevc|aac|ac3|10bit)\b.*$",
    re.IGNORECASE,
)
EN_HINTS = ("en", "eng", "english")
STOPWORDS = {"the", "a", "an", "of", "and", "in", "on", "to"}

COLUMNS = [
    "film_id",
    "title",
    "year",
    "genre",
    "lead_actors",
    "path",
    "duration_s",
    "fps",
    "width",
    "height",
    "vcodec",
    "acodec",
    "audio_channels",
    "n_sub_text",
    "n_sub_bitmap",
    "sub_langs",
    "srt_path",
    "srt_cues",
    "srt_words",
    "wpm",
    "srt_first_s",
    "srt_last_s",
    "tail_to_end_s",
    "max_gap_s",
    "n_speaker_labeled",
    "n_watermark",
    "snr_db",
    "clip_ratio",
    "face_ratio",
    "face_ratio_min_window",
    "probe_windows",
    "probe_frames",
    "probe_start_s",
    "nonenglish_ratio",
    "pass_audition",
    "reject_reason",
]


# --------------------------------------------------------------------------- #
# penemuan berkas
# --------------------------------------------------------------------------- #
def discover_candidates(raw_dir: Path, excludes: list[str]) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir tidak ditemukan: {raw_dir}")

    patterns = [p.lower() for p in excludes]
    found: list[Path] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXT:
            continue
        lowered = str(path).lower()
        if any(pattern in lowered for pattern in patterns):
            continue
        found.append(path)
    return found


def dedupe_by_folder(infos: dict[Path, MediaInfo]) -> dict[Path, MediaInfo]:
    """Satu folder biasanya berisi satu film. Bila lebih, ambil yang terlama."""
    best: dict[Path, tuple[float, Path]] = {}
    for path, info in infos.items():
        key = path.parent
        if key not in best or info.duration_s > best[key][0]:
            best[key] = (info.duration_s, path)
    keep = {path for _, path in best.values()}
    return {p: i for p, i in infos.items() if p in keep}


RELEASE_WORDS = {
    "1080p", "720p", "480p", "2160p", "bluray", "brrip", "bdrip", "webrip",
    "hdtv", "remux", "x264", "x265", "h264", "h265", "hevc", "10bit", "8bit",
    "hdr", "aac", "ac3", "dts", "ddp", "repack", "proper", "extended", "yify",
    "yawntic", "rarbg", "sparks", "tigole", "bokutox", "drones", "ethd",
    "english", "sub", "subs", "srt",
}


def title_tokens(stem: str) -> set[str]:
    """Kata yang menamai filmnya, tanpa istilah rilis dan angka resolusi."""
    words = re.split(r"[^a-z0-9]+", stem.lower())
    return {
        word
        for word in words
        if len(word) >= 4 and not word.isdigit() and word not in RELEASE_WORDS
    }


def find_subtitle(video_path: Path, raw_dir: Path) -> Path | None:
    """Cari subtitle yang memang untuk film ini.

    Folder film diutamakan. Kalau subtitle-nya ada di folder induk, hanya
    kandidat yang berbagi kata judul yang diterima: tanpa syarat itu, film
    tanpa subtitle sendiri akan diam-diam memakai subtitle film lain — teksnya
    salah total dan tidak ada tahap yang berhenti.
    """
    search_dirs = [video_path.parent]
    if video_path.parent != raw_dir and video_path.parent.parent not in (None, raw_dir.parent):
        search_dirs.append(video_path.parent.parent)

    candidates: list[Path] = []
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        candidates.extend(
            p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUB_EXT
        )
    if not candidates:
        return None

    wanted = title_tokens(video_path.stem)
    if wanted:
        candidates = [p for p in candidates if title_tokens(p.stem) & wanted]
    if not candidates:
        return None

    def score(path: Path) -> float:
        stem = path.stem.lower()
        similarity = difflib.SequenceMatcher(None, video_path.stem.lower(), stem).ratio()
        return similarity + (0.3 if any(hint in stem for hint in EN_HINTS) else 0.0)

    return max(candidates, key=score)


# --------------------------------------------------------------------------- #
# registri film
# --------------------------------------------------------------------------- #
def load_registry(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    registry = pd.read_csv(path)
    if "film_id" not in registry.columns:
        raise ValueError(f"{path} harus punya kolom film_id")
    if "path_contains" not in registry.columns:
        registry["path_contains"] = ""
    return registry


def lookup_registry(registry: pd.DataFrame, path: Path) -> tuple[dict, bool]:
    if registry.empty:
        return {}, False
    needle = str(path).lower()
    for _, row in registry.iterrows():
        pattern = str(row.get("path_contains") or "").strip().lower()
        if pattern and pattern in needle:
            return row.to_dict(), True
    return {}, False


def infer_title_year(path: Path) -> tuple[str, int | None]:
    text = path.parent.name if path.parent.name.lower() not in {"film", "movies"} else path.stem
    years = YEAR.findall(text)
    year = int(years[-1]) if years else None
    title = re.sub(r"[\._\[\]\(\)]", " ", text)
    title = RELEASE_NOISE.sub("", title)
    title = re.sub(r"\s+", " ", title).strip(" -")
    return title or path.stem, year


def make_film_id(title: str, year: int | None) -> str:
    words = re.findall(r"[A-Za-z]+", title)
    initials = "".join(w[0] for w in words if w.lower() not in STOPWORDS)[:3].upper() or "FLM"
    return f"{initials}{(str(year) if year else '0000')[-2:]}"


# --------------------------------------------------------------------------- #
# pengukuran per film
# --------------------------------------------------------------------------- #
def probe_windows(
    duration_s: float, fracs: list[float], window_s: float
) -> list[tuple[float, float]]:
    """Beberapa jendela uji tersebar, bukan satu probe di tengah film.

    Satu jendela 3 menit bisa salah menilai film: adegan lebar tanpa wajah
    selama 3 menit bukan berarti filmnya tidak punya wajah.
    """
    windows: list[tuple[float, float]] = []
    for frac in fracs:
        start = max(0.0, duration_s * frac)
        length = min(window_s, duration_s - start - 1.0)
        if length < 10.0:
            continue
        windows.append((round(start, 3), round(length, 3)))
    if not windows:
        fallback = max(10.0, min(window_s, duration_s * 0.8))
        windows.append((round(duration_s * 0.1, 3), round(fallback, 3)))
    return windows


def build_row(
    path: Path,
    cfg: Config,
    registry: pd.DataFrame,
    raw_dir: Path,
    skip_faces: bool,
    info: MediaInfo,
) -> dict:
    row: dict = {column: None for column in COLUMNS}
    row.update(info.as_dict())

    meta, from_registry = lookup_registry(registry, path)
    title, year = infer_title_year(path)
    if from_registry:
        title = str(meta.get("title") or title)
        year = int(meta["year"]) if pd.notna(meta.get("year")) else year
        row["film_id"] = str(meta["film_id"])
        row["genre"] = meta.get("genre")
        row["lead_actors"] = meta.get("lead_actors")
    else:
        row["film_id"] = make_film_id(title, year)

    row["title"] = title
    row["year"] = year
    row["path"] = str(path)

    # ---- subtitle ----
    subtitle_path = find_subtitle(path, raw_dir)
    stats = SubtitleStats()
    if subtitle_path is not None:
        cues, errors = parse_srt(subtitle_path)
        stats = subtitle_stats(cues, info.duration_s, errors)
        row["srt_path"] = str(subtitle_path)
        row["srt_cues"] = stats.n_cues
        row["srt_words"] = stats.n_words
        row["wpm"] = stats.wpm
        row["srt_first_s"] = stats.first_s
        row["srt_last_s"] = stats.last_s
        row["tail_to_end_s"] = stats.tail_to_end_s
        row["max_gap_s"] = stats.max_gap_s
        row["n_speaker_labeled"] = stats.n_speaker_labeled
        row["n_watermark"] = stats.n_watermark

    # ---- probe audio + wajah di beberapa jendela ----
    windows = probe_windows(
        info.duration_s,
        cfg.get("audition.probe_fracs"),
        cfg.get("audition.probe_window_s"),
    )
    row["probe_windows"] = len(windows)
    row["probe_start_s"] = windows[0][0]

    snrs: list[float] = []
    clips: list[float] = []
    ratios: list[float] = []
    probe_frames = 0

    for start_s, length_s in windows:
        samples = audio.extract_audio(path, start_s, length_s)
        snrs.append(audio.snr_db(samples))
        clips.append(audio.clip_ratio(samples))

        if skip_faces or info.width == 0:
            continue
        faces = video.face_stats(
            path,
            start_s,
            length_s,
            fps=cfg.get("audition.fps_sample"),
            width=cfg.get("audition.frame_width"),
            height=cfg.get("audition.frame_height"),
            min_size_px=cfg.get("audition.face_min_size_px"),
        )
        ratios.append(faces["face_ratio"])
        probe_frames += faces["probe_frames"]

    if snrs:
        row["snr_db"] = round(statistics.median(snrs), 2)
        row["clip_ratio"] = round(max(clips), 5)
    row["probe_frames"] = probe_frames
    if ratios:
        row["face_ratio"] = round(statistics.fmean(ratios), 4)
        row["face_ratio_min_window"] = round(min(ratios), 4)

    # ---- keputusan ----
    reasons: list[str] = []
    if info.duration_s < cfg.get("audition.duration_min_s"):
        reasons.append("durasi_pendek")
    if subtitle_path is None:
        reasons.append("tanpa_subtitle")
    elif stats.n_cues == 0:
        reasons.append("subtitle_kosong")
    else:
        if stats.wpm < cfg.get("audition.wpm_min"):
            reasons.append(f"wpm_rendah:{stats.wpm:.0f}")
        elif stats.wpm > cfg.get("audition.wpm_max"):
            reasons.append(f"wpm_tinggi:{stats.wpm:.0f}")
    if row["snr_db"] is not None and row["snr_db"] < cfg.get("audition.snr_min_db"):
        reasons.append(f"snr_rendah:{row['snr_db']:.0f}")
    if row["face_ratio"] is not None and row["face_ratio"] < cfg.get("audition.face_ratio_min"):
        reasons.append(f"wajah_jarang:{row['face_ratio']:.2f}")

    row["pass_audition"] = not reasons
    row["reject_reason"] = ";".join(reasons)
    return row


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--raw-dir", default=None, help="menimpa paths.raw_dir")
    parser.add_argument("--out-dir", default=None, help="menimpa paths.data_dir")
    parser.add_argument(
        "--skip-faces", action="store_true", help="lewati deteksi wajah, jauh lebih cepat"
    )
    parser.add_argument("--limit", type=int, default=None, help="batasi film, untuk uji cepat")
    parser.add_argument("--force", action="store_true", help="timpa films.csv yang sudah ada")
    args = parser.parse_args(argv)

    require_tools("ffmpeg", "ffprobe")
    cfg = load_config(args.config)

    raw_dir = Path(args.raw_dir).expanduser() if args.raw_dir else cfg.resolve("paths.raw_dir")
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else cfg.resolve("paths.data_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    films_csv = out_dir / "films.csv"
    reject_csv = out_dir / "reject_log.csv"
    if films_csv.exists() and not args.force:
        print(f"{films_csv} sudah ada. Pakai --force untuk menimpa.")
        return 1

    excludes = list(cfg.get("audition.exclude_patterns", []))
    candidates = discover_candidates(raw_dir, excludes)
    if args.limit:
        candidates = candidates[: args.limit]
    if not candidates:
        print(f"Tidak ada berkas video di {raw_dir} setelah penyaringan.")
        return 1

    print(f"Menemukan {len(candidates)} berkas video di {raw_dir}")
    infos: dict[Path, MediaInfo] = {}
    for path in tqdm(candidates, desc="ffprobe", unit="film"):
        try:
            infos[path] = probe_media(path)
        except Exception as exc:  # noqa: BLE001 - satu berkas rusak tidak boleh menghentikan batch
            print(f"  lewati {path.name}: {exc}")
    infos = dedupe_by_folder(infos)

    registry = load_registry(cfg.resolve("paths.films_registry"))
    if registry.empty:
        print("films_registry.csv tidak ada — film_id dibuat otomatis dari nama berkas.")

    rows: list[dict] = []
    for path, info in tqdm(sorted(infos.items()), desc="audisi", unit="film"):
        try:
            rows.append(build_row(path, cfg, registry, raw_dir, args.skip_faces, info))
        except Exception as exc:  # noqa: BLE001
            print(f"  gagal {path.name}: {exc}")

    if not rows:
        print("Tidak ada film yang berhasil diproses.")
        return 1

    films = pd.DataFrame(rows)[COLUMNS]
    films.to_csv(films_csv, index=False)

    rejected = films[~films["pass_audition"]].copy()
    if not rejected.empty:
        rejected[["film_id", "title", "year", "path", "reject_reason"]].to_csv(
            reject_csv, index=False
        )

    # ---- ringkasan ----
    print()
    print("=" * 78)
    header = (
        f"{'film_id':<9}{'tahun':<7}{'menit':>7}{'wpm':>7}"
        f"{'snr':>7}{'wajah':>7}{'cue':>7}  status"
    )
    print(header)
    print("-" * 78)
    for _, row in films.iterrows():
        status = "LOLOS" if row["pass_audition"] else str(row["reject_reason"])
        print(
            f"{row['film_id']:<9}{str(row['year'] or '-'):<7}{row['duration_s'] / 60:>7.1f}"
            f"{_fmt(row['wpm']):>7}{_fmt(row['snr_db']):>7}{_fmt(row['face_ratio']):>7}"
            f"{_fmt(row['srt_cues'], 0):>7}  {status}"
        )
    print("=" * 78)
    passed = int(films["pass_audition"].sum())
    print(f"{passed}/{len(films)} film lolos audisi -> {films_csv}")
    if not rejected.empty:
        print(f"{len(rejected)} film ditolak -> {reject_csv}")
    return 0


def _fmt(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.{digits}f}"


if __name__ == "__main__":
    sys.exit(main())
