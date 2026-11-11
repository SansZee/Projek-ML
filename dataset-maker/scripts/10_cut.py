#!/usr/bin/env python3
"""10_cut.py — potong klip, audio, dan wajah dari film asli.

Dijalankan **di laptop pemilik film**. Koordinator hanya mengirim
`data/selected.csv`; media tidak berpindah sampai tahap ini, dan setelah ini
hanya 3.000 klip kecil yang dikumpulkan.

Untuk setiap klip:

  media/clips/<clip_id>.mp4    720p + audio, untuk model video
  media/audio/<clip_id>.wav    16 kHz mono, untuk model suara
  media/faces/<clip_id>.mp4    wajah dominan 224×224 @ 5 fps

Katalog `data/clips.csv` mencatat path relatif dan ukuran berkas; repo
pelatihan membaca katalog itu, bukan folder media secara langsung.

Contoh:

  uv run python scripts/10_cut.py
  uv run python scripts/10_cut.py --film FDS09 --limit 3
  uv run python scripts/10_cut.py --no-faces
  uv run python scripts/10_cut.py --force --height 480
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from emosi import cut, face
from emosi.config import Config, load_config
from emosi.media import require_tools

CATALOG = "clips.csv"
FACE_FPS = 5.0

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


def face_settings(cfg: Config) -> face.FaceSettings:
    return face.FaceSettings(
        model_pack=str(cfg.get("face.model_pack")),
        det_size=int(cfg.get("face.det_size")),
        det_thresh=float(cfg.get("face.det_thresh")),
        sample_fps=FACE_FPS,
        frame_width=int(cfg.get("face.frame_width")),
        min_face_px=int(cfg.get("face.min_face_px")),
        dominance_min_ratio=float(cfg.get("face.dominance_min_ratio")),
        min_face_frames=int(cfg.get("face.min_face_frames")),
        face_ratio_ok=float(cfg.get("face.face_ratio_ok")),
        cluster_threshold=float(cfg.get("face.cluster_threshold")),
        cluster_min_utterances=int(cfg.get("face.cluster_min_utterances")),
        device=str(cfg.get("face.device")),
    )


def crop_faces(
    source: Path,
    dest: Path,
    row: pd.Series,
    analyzer,
    settings: face.FaceSettings,
    cut_settings: cut.CutSettings,
) -> None:
    """Tulis video wajah 224×224; wajah terbesar di tiap frame sampel."""
    import cv2

    width, height = face.scaled_size(
        int(row["width"]), int(row["height"]), settings.frame_width
    )
    command = cut.face_command(dest, fps=FACE_FPS, settings=cut_settings)
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    last = None
    try:
        for frame_rgb in face.iter_rgb_frames(
            source,
            float(row["start_s"]),
            cut.duration(row["start_s"], row["end_s"]),
            FACE_FPS,
            width,
            height,
        ):
            detections = face.analyze_frame(
                analyzer, frame_rgb, min_face_px=settings.min_face_px
            )
            if detections:
                best = max(detections, key=lambda item: item.area)
                current = (best.x1, best.y1, best.x2, best.y2)
            elif last is not None:
                current = last
            else:
                current = cut.center_box(width, height)
            last = cut.smooth_box(last, current, alpha=cut_settings.face_alpha)

            left, top, right, bottom = cut.face_box(
                *last, width, height, margin=cut_settings.face_margin
            )
            crop = frame_rgb[top:bottom, left:right]
            if crop.size == 0:
                crop = frame_rgb
            crop = cv2.resize(
                crop,
                (cut_settings.face_size, cut_settings.face_size),
                interpolation=cv2.INTER_AREA,
            )
            proc.stdin.write(crop.tobytes())
    finally:
        proc.stdin.close()
        code = proc.wait()
        if code != 0:
            tail = (proc.stderr.read() if proc.stderr else b"").decode("utf-8", "replace")
            raise RuntimeError(f"ffmpeg wajah gagal ({code}) untuk {dest.name}: {tail[-400:]}")


def local_films(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "films.csv"
    if not path.exists():
        return pd.DataFrame()
    films = pd.read_csv(path)
    wanted = {"film_id", "path", "fps", "width", "height"}
    if not wanted <= set(films.columns):
        return pd.DataFrame()
    return films[list(wanted) + [c for c in ("title", "owner") if c in films.columns]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--selected", default=None, help="default data/selected.csv")
    parser.add_argument("--film", action="append", default=None, help="batasi ke film_id ini")
    parser.add_argument("--limit", type=int, default=None, help="batasi jumlah klip (uji cepat)")
    parser.add_argument("--force", action="store_true", help="potong ulang yang sudah ada")
    parser.add_argument("--no-faces", action="store_true", help="lewati video wajah")
    parser.add_argument("--height", type=int, default=720, help="tinggi klip video")
    parser.add_argument("--crf", type=int, default=28, help="kualitas x264 klip")
    parser.add_argument("--audio-sr", type=int, default=16000, help="laju audio WAV")
    parser.add_argument("--out-dir", default=None, help="default paths.media_dir")
    parser.add_argument("--catalog", default=None, help="default data/clips.csv")
    args = parser.parse_args(argv)

    require_tools("ffmpeg", "ffprobe")
    cfg = load_config(args.config)
    data_dir = cfg.resolve("paths.data_dir")
    media_dir = Path(args.out_dir).expanduser() if args.out_dir else cfg.resolve("paths.media_dir")
    selected_path = (
        Path(args.selected).expanduser() if args.selected else data_dir / "selected.csv"
    )
    catalog_path = (
        Path(args.catalog).expanduser() if args.catalog else data_dir / CATALOG
    )
    if not selected_path.exists():
        print(f"{selected_path} tidak ada. Jalankan scripts/09_select.py dulu.")
        return 1

    selected = pd.read_csv(selected_path)
    films = local_films(data_dir)
    if films.empty:
        print(f"{data_dir / 'films.csv'} tidak ada atau tidak punya kolom path.")
        return 1
    if args.film:
        selected = selected[selected["film_id"].isin(args.film)]
    if args.limit:
        selected = selected.head(args.limit)
    if selected.empty:
        print("Tidak ada klip yang dipilih.")
        return 1

    settings = cut.CutSettings(
        height=args.height,
        crf=args.crf,
        audio_sr=args.audio_sr,
    )
    dirs = {
        "clips": media_dir / "clips",
        "audio": media_dir / "audio",
        "faces": media_dir / "faces",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    analyzer = None
    if not args.no_faces:
        print("menyiapkan detektor wajah…")
        analyzer, provider = face.build_analyzer(face_settings(cfg))
        print(f"provider onnxruntime: {provider}")

    by_film = {str(row.film_id): row for row in films.itertuples()}
    rows: list[dict] = []
    skipped = failed = 0

    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="klip", unit="klip"):
        clip_id = str(row["clip_id"])
        film = by_film.get(str(row["film_id"]))
        catalog_row = {
            "clip_id": clip_id,
            "utterance_id": str(row["utterance_id"]),
            "film_id": str(row["film_id"]),
            "label": str(row["label"]),
            "confidence": float(row["confidence"]),
            "duration_s": float(row["duration_s"]),
            "clip_path": f"clips/{clip_id}.mp4",
            "audio_path": f"audio/{clip_id}.wav",
            "face_path": "" if args.no_faces else f"faces/{clip_id}.mp4",
            "clip_bytes": 0,
            "audio_bytes": 0,
            "face_bytes": 0,
            "status": "ok",
        }
        if film is None:
            catalog_row["status"] = "film_tidak_ada"
            skipped += 1
            rows.append(catalog_row)
            continue

        clip_path = dirs["clips"] / f"{clip_id}.mp4"
        audio_path = dirs["audio"] / f"{clip_id}.wav"
        face_path = dirs["faces"] / f"{clip_id}.mp4"
        existed = clip_path.exists() and audio_path.exists()
        if existed and (args.no_faces or face_path.exists()) and not args.force:
            skipped += 1
            catalog_row.update(
                clip_bytes=clip_path.stat().st_size,
                audio_bytes=audio_path.stat().st_size,
                face_bytes=face_path.stat().st_size if face_path.exists() else 0,
                status="ada",
            )
            rows.append(catalog_row)
            continue

        info = row.copy()
        info["width"] = film.width
        info["height"] = film.height
        try:
            source = Path(str(film.path))
            cut.run_ffmpeg(
                cut.clip_command(
                    source, clip_path, row["start_s"], row["end_s"], settings
                )
            )
            cut.run_ffmpeg(
                cut.audio_command(
                    source, audio_path, row["start_s"], row["end_s"], settings
                )
            )
            if analyzer is not None:
                crop_faces(source, face_path, info, analyzer, face_settings(cfg), settings)
            catalog_row.update(
                clip_bytes=clip_path.stat().st_size,
                audio_bytes=audio_path.stat().st_size,
                face_bytes=face_path.stat().st_size if face_path.exists() else 0,
            )
        except Exception as error:  # noqa: BLE001 - satu klip gagal tidak menghentikan batch
            failed += 1
            catalog_row["status"] = "gagal"
            print(f"  {clip_id}: {error}")
        rows.append(catalog_row)

    catalog = pd.DataFrame(rows, columns=CATALOG_COLUMNS)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(catalog_path, index=False)

    ok = catalog[catalog["status"].isin(("ok", "ada"))]
    total_mb = (
        ok[["clip_bytes", "audio_bytes", "face_bytes"]].sum().sum() / 1048576
        if len(ok)
        else 0.0
    )
    print()
    print("=" * 72)
    print(
        f"{len(ok)} klip siap · {skipped} dilewati · {failed} gagal · "
        f"{total_mb:.0f} MB"
    )
    if len(ok):
        print(
            f"rata-rata {ok['clip_bytes'].mean() / 1024:.0f} KB video · "
            f"{ok['audio_bytes'].mean() / 1024:.0f} KB audio · "
            f"{ok['face_bytes'].mean() / 1024:.0f} KB wajah"
        )
    if failed:
        print(f"{failed} klip gagal; cek pesan di atas dan jalankan ulang tanpa --force.")
    print(f"katalog → {catalog_path}")
    print(f"media   → {media_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
