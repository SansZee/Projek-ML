#!/usr/bin/env python3
"""04_face_screen.py — ukur wajah dan embedding untuk setiap ujaran.

Untuk setiap ujaran:

  - ``face_ratio``      bagian frame sampel yang memuat wajah layak
  - ``face_width_frac`` lebar wajah dominan dibanding lebar frame
  - ``dominant_share``  bagian frame yang hanya punya satu wajah besar
  - embedding 512-d    rata-rata ArcFace dari wajah dominan

Embedding disimpan terpisah (``faces/<film_id>.npy``) karena 2.467 x 512
angka tidak layak masuk CSV. Kolom ``emb_path`` dan ``emb_row`` menunjukkan
baris mana yang dipakai.

Keluaran:

  data/faces/<film_id>.csv     satu baris per ujaran
  data/faces/<film_id>.npy     embedding, sejajar baris CSV (0 = tidak ada)
  data/faces.csv               gabungan semua film
  data/actors.csv              klaster aktor per film (usulan, isi nama manual)
  data/faces/faces_meta.json   ringkasan per film + hash config

Contoh:

  uv run python scripts/04_face_screen.py --film ABT13 --force
  uv run python scripts/04_face_screen.py --out-dir data/uji --scenes 2
  uv run python scripts/04_face_screen.py --no-cluster
  uv run python scripts/04_face_screen.py --recluster

Butuh extra ``face``: ``uv sync --locked --extra face``. Tanpa GPU tahap ini
tetap jalan tetapi sekitar 10x lebih lambat.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from emosi import face
from emosi.config import Config, load_config
from emosi.media import probe_media, require_tools
from emosi.tables import load_meta, merge_rows, save_meta

COLUMNS = [
    "film_id",
    "utterance_id",
    "scene_id",
    "start_s",
    "end_s",
    "duration_s",
    "n_frames",
    "n_frames_face",
    "face_ratio",
    "n_faces_max",
    "multi_face_share",
    "dominant_share",
    "face_area_ratio",
    "face_width_frac",
    "det_score",
    "emb_frames",
    "face_ok",
    "cluster",
    "actor_key",
    "emb_path",
    "emb_row",
]

ACTOR_COLUMNS = [
    "actor_key",
    "film_id",
    "cluster",
    "n_utterances",
    "median_face_ratio",
    "median_emb_frames",
    "lead_actors",
    "actor_label",
]


def face_settings(cfg: Config) -> face.FaceSettings:
    return face.FaceSettings(
        model_pack=str(cfg.get("face.model_pack")),
        det_size=int(cfg.get("face.det_size")),
        det_thresh=float(cfg.get("face.det_thresh")),
        sample_fps=float(cfg.get("face.sample_fps")),
        frame_width=int(cfg.get("face.frame_width")),
        min_face_px=int(cfg.get("face.min_face_px")),
        dominance_min_ratio=float(cfg.get("face.dominance_min_ratio")),
        min_face_frames=int(cfg.get("face.min_face_frames")),
        face_ratio_ok=float(cfg.get("face.face_ratio_ok")),
        cluster_threshold=float(cfg.get("face.cluster_threshold")),
        cluster_min_utterances=int(cfg.get("face.cluster_min_utterances")),
        device=str(cfg.get("face.device")),
    )


def scene_frames(
    analyzer,
    path: Path,
    start_s: float,
    end_s: float,
    width: int,
    height: int,
    settings: face.FaceSettings,
) -> tuple[list[float], list[face.FrameFaces]]:
    """Dekode satu adegan sekali, deteksi tiap frame sampel."""
    times: list[float] = []
    frames: list[face.FrameFaces] = []
    duration = max(0.1, end_s - start_s)
    for index, frame_rgb in enumerate(
        face.iter_rgb_frames(path, start_s, duration, settings.sample_fps, width, height)
    ):
        time_s = start_s + index / settings.sample_fps
        if time_s > end_s + 1.0:
            break
        detections = face.analyze_frame(analyzer, frame_rgb, min_face_px=settings.min_face_px)
        times.append(round(time_s, 3))
        frames.append(face.FrameFaces(width=width, height=height, faces=detections))
    return times, frames


def summarize_utterances(
    utterances: pd.DataFrame,
    times: list[float],
    frames: list[face.FrameFaces],
    settings: face.FaceSettings,
) -> list[tuple[dict, np.ndarray | None]]:
    """Ringkas frame satu adegan menjadi baris per ujaran + embedding."""
    spans = [
        (float(row.start_s), float(row.end_s)) for row in utterances.itertuples()
    ]
    buckets = face.frames_by_utterance(times, spans)
    results: list[tuple[dict, np.ndarray | None]] = []
    for (_, row), frame_indices in zip(utterances.iterrows(), buckets, strict=True):
        summary = face.summarize_frames([frames[i] for i in frame_indices], settings)
        face_ok = (
            summary.n_face_frames >= settings.min_face_frames
            and summary.face_ratio >= settings.face_ratio_ok
        )
        record = {
            "film_id": row.film_id,
            "utterance_id": row.utterance_id,
            "scene_id": row.scene_id,
            "start_s": row.start_s,
            "end_s": row.end_s,
            "duration_s": row.duration_s,
            **summary.as_dict(),
            "face_ok": bool(face_ok),
            "cluster": -1,
            "actor_key": "",
            "emb_path": "",
            "emb_row": -1,
        }
        results.append((record, summary.embedding))
    return results


def assign_clusters(
    rows: list[dict],
    embeddings: np.ndarray,
    settings: face.FaceSettings,
    film_id: str,
    lead_actors: str = "",
) -> list[dict]:
    """Beri label klaster aktor dan tulis daftar aktor per film."""
    # Baris dari CSV lama bisa masih membawa klaster sebelumnya; bersihkan dulu
    # supaya ujaran yang tidak lagi masuk klaster tidak menyimpan label hantu.
    for row in rows:
        row["cluster"] = -1
        row["actor_key"] = ""

    eligible = [
        index
        for index, row in enumerate(rows)
        if row["emb_frames"] > 0 and np.any(embeddings[index])
    ]
    if not eligible:
        return []

    # Wajah terbaik masuk lebih dulu supaya centroid tidak dibentuk dari frame
    # profil yang buram.
    order = sorted(
        eligible,
        key=lambda index: (-float(rows[index]["face_ratio"]), -int(rows[index]["emb_frames"])),
    )
    labels = face.cluster_embeddings(
        embeddings[eligible],
        threshold=settings.cluster_threshold,
        min_size=settings.cluster_min_utterances,
        order=[eligible.index(index) for index in order],
    )

    # Nomor klaster diurutkan dari yang paling banyak muncul, supaya A01 stabil
    # dan mudah dibaca saat nama aktor diisi manual.
    found = [int(label) for label in labels if label >= 0]
    counts = pd.Series(found).value_counts()
    ranking = {label: rank + 1 for rank, label in enumerate(counts.index.tolist())}
    for index, label in zip(eligible, labels, strict=True):
        number = ranking.get(int(label), 0)
        if number:
            rows[index]["cluster"] = number
            rows[index]["actor_key"] = f"{film_id}-A{number:02d}"

    actor_rows: list[dict] = []
    for number in sorted(set(ranking.values())):
        members = [row for row in rows if row["cluster"] == number]
        actor_rows.append(
            {
                "actor_key": f"{film_id}-A{number:02d}",
                "film_id": film_id,
                "cluster": number,
                "n_utterances": len(members),
                "median_face_ratio": round(
                    float(np.median([row["face_ratio"] for row in members])), 4
                ),
                "median_emb_frames": int(
                    np.median([row["emb_frames"] for row in members])
                ),
                "lead_actors": lead_actors,
                "actor_label": "",
            }
        )
    return actor_rows


def process_film(
    analyzer,
    film: pd.Series,
    utterances: pd.DataFrame,
    scenes: pd.DataFrame,
    settings: face.FaceSettings,
    scene_limit: int | None,
) -> tuple[pd.DataFrame, np.ndarray, list[dict], dict]:
    film_id = str(film["film_id"])
    path = Path(str(film["path"]))
    info = probe_media(path)
    width, height = face.scaled_size(info.width, info.height, settings.frame_width)

    scene_ids = list(dict.fromkeys(utterances["scene_id"].astype(str)))
    if scene_limit:
        scene_ids = scene_ids[:scene_limit]
        utterances = utterances[utterances["scene_id"].astype(str).isin(scene_ids)]

    rows: list[dict] = []
    vectors: list[np.ndarray | None] = []
    started = time.time()
    frames_seen = 0

    progress = tqdm(
        scenes[scenes["scene_id"].astype(str).isin(scene_ids)].itertuples(),
        desc=f"{film_id} ({width}x{height})",
        unit="adegan",
        leave=False,
    )
    for scene in progress:
        scene_utts = utterances[utterances["scene_id"].astype(str) == str(scene.scene_id)]
        if scene_utts.empty:
            continue
        times, frames = scene_frames(
            analyzer,
            path,
            float(scene.start_s),
            float(scene.end_s),
            width,
            height,
            settings,
        )
        frames_seen += len(frames)
        progress.set_postfix_str(f"{frames_seen} frame")
        if not times:
            print(f"   adegan {scene.scene_id}: tidak ada frame terbaca")
            continue
        for record, embedding in summarize_utterances(scene_utts, times, frames, settings):
            rows.append(record)
            vectors.append(embedding)

    if not rows:
        empty = pd.DataFrame(columns=COLUMNS)
        return empty, np.zeros((0, 512), dtype=np.float32), [], {}

    dim = next((int(embedding.size) for embedding in vectors if embedding is not None), 512)
    matrix = np.zeros((len(rows), dim), dtype=np.float32)
    for index, embedding in enumerate(vectors):
        if embedding is not None:
            matrix[index] = embedding

    actor_rows: list[dict] = []
    if settings.cluster_threshold > 0:
        actor_rows = assign_clusters(
            rows, matrix, settings, film_id, str(film.get("lead_actors") or "")
        )

    frame = pd.DataFrame(rows)
    frame["emb_row"] = np.arange(len(frame))
    frame["emb_path"] = f"faces/{film_id}.npy"
    frame = frame[COLUMNS]

    elapsed = time.time() - started
    stats = {
        "film_id": film_id,
        "n_utterances": len(frame),
        "n_scenes": int(frame["scene_id"].nunique()),
        "n_frames": int(frames_seen),
        "n_face_ok": int(frame["face_ok"].sum()),
        "face_ok_share": round(float(frame["face_ok"].mean()), 3),
        "median_face_ratio": round(float(frame["face_ratio"].median()), 3),
        "emb_coverage": round(float((frame["emb_frames"] > 0).mean()), 3),
        "n_actors": len(actor_rows),
        "frame_width": width,
        "frame_height": height,
        "seconds": round(elapsed, 1),
    }
    return frame, matrix, actor_rows, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--film", action="append", default=None, help="batasi ke film_id ini")
    parser.add_argument("--limit", type=int, default=None, help="batasi jumlah film")
    parser.add_argument("--force", action="store_true", help="timpa hasil yang sudah ada")
    parser.add_argument(
        "--no-cluster", action="store_true", help="lewati pengelompokan aktor"
    )
    parser.add_argument(
        "--recluster",
        action="store_true",
        help="hitung ulang klaster aktor dari CSV+NPY yang ada, tanpa dekode video",
    )
    parser.add_argument(
        "--scenes",
        type=int,
        default=None,
        help="uji cepat: hanya N adegan pertama per film (keluaran tidak lengkap)",
    )
    parser.add_argument(
        "--out-dir", default=None, help="timpa paths.data_dir (berguna untuk uji cepat)"
    )
    args = parser.parse_args(argv)

    require_tools("ffmpeg", "ffprobe")
    cfg = load_config(args.config)
    data_dir = cfg.resolve("paths.data_dir")
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else data_dir

    films_csv = data_dir / "films.csv"
    scenes_csv = data_dir / "scenes.csv"
    utterances_csv = data_dir / "utterances.csv"
    for path, hint in (
        (films_csv, "00_inventory.py"),
        (scenes_csv, "02_scenes.py"),
        (utterances_csv, "03_utterances.py"),
    ):
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
    utterances = pd.read_csv(utterances_csv)
    settings = face_settings(cfg)
    if args.no_cluster:
        settings = dataclasses.replace(settings, cluster_threshold=0.0)

    faces_dir = out_dir / "faces"
    combined_csv = out_dir / "faces.csv"
    actors_csv = out_dir / "actors.csv"
    meta = load_meta(faces_dir / "faces_meta.json")
    pending: list[pd.Series] = []
    if args.recluster:
        print("mode recluster: membaca CSV+NPY, tanpa dekode video")
    else:
        for _, film in films.iterrows():
            out_path = faces_dir / f"{film['film_id']}.csv"
            if out_path.exists() and not args.force:
                print(f"{film['film_id']}: {out_path.name} sudah ada, dilewati (pakai --force)")
                continue
            pending.append(film)

    if args.scenes:
        print(
            "PERINGATAN: --scenes hanya untuk uji cepat. Keluaran tidak lengkap "
            "dan jangan dipakai untuk dataset."
        )

    combined_rows: list[pd.DataFrame] = []
    actor_rows_all: list[dict] = []

    if pending:
        warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
        warnings.filterwarnings("ignore", message=".*estimate.*deprecated.*")
        print(
            f"model {settings.model_pack} · sampel {settings.sample_fps:.1f} fps · "
            f"lebar {settings.frame_width} px · ambang wajah {settings.min_face_px} px"
        )
        analyzer, provider = face.build_analyzer(settings)
        print(f"provider onnxruntime: {provider}")
        if provider == "CPUExecutionProvider":
            print(
                "  PERINGATAN: GPU tidak terpakai. Satu film bisa memakan "
                "1-2 jam, bukan 5-10 menit."
            )
        for film in tqdm(pending, desc="film", unit="film"):
            film_id = str(film["film_id"])
            print()
            print(f"── {film_id}  {film.get('title', '')}")
            film_utts = utterances[utterances["film_id"] == film_id].sort_values(
                "start_s", kind="stable"
            )
            if film_utts.empty:
                print("   tidak ada ujaran untuk film ini, dilewati")
                continue
            frame, matrix, actor_rows, stats = process_film(
                analyzer,
                film,
                film_utts,
                scenes[scenes["film_id"] == film_id],
                settings,
                args.scenes,
            )
            if frame.empty:
                print("   tidak ada frame yang bisa dianalisis")
                continue

            faces_dir.mkdir(parents=True, exist_ok=True)
            frame.to_csv(faces_dir / f"{film_id}.csv", index=False)
            np.save(faces_dir / f"{film_id}.npy", matrix)
            actor_rows_all.extend(actor_rows)

            print(
                f"   ujaran {stats['n_utterances']} · frame {stats['n_frames']} · "
                f"wajah cukup {stats['face_ok_share']:.0%} · "
                f"embedding {stats['emb_coverage']:.0%} · "
                f"aktor {stats['n_actors']} · {stats['seconds']:.0f} s"
            )
            meta[film_id] = {
                **stats,
                "provider": provider,
                "model_pack": settings.model_pack,
                "sample_fps": settings.sample_fps,
                "config_sha256": cfg.sha256,
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            save_meta(
                faces_dir / "faces_meta.json",
                meta,
                generated_by="scripts/04_face_screen.py",
            )

    if args.recluster:
        for _, film in films.iterrows():
            film_id = str(film["film_id"])
            csv_path = faces_dir / f"{film_id}.csv"
            npy_path = faces_dir / f"{film_id}.npy"
            if not csv_path.exists() or not npy_path.exists():
                print(f"{film_id}: {csv_path.name}/{npy_path.name} tidak ada, dilewati")
                continue
            frame = pd.read_csv(csv_path)
            matrix = np.load(npy_path)
            if len(frame) != len(matrix):
                print(f"{film_id}: jumlah baris CSV dan NPY tidak cocok, dilewati")
                continue
            rows = frame.to_dict("records")
            actor_rows_all.extend(
                assign_clusters(
                    rows, matrix, settings, film_id, str(film.get("lead_actors") or "")
                )
            )
            pd.DataFrame(rows)[COLUMNS].to_csv(csv_path, index=False)
            print(f"{film_id}: {len(rows)} ujaran dikelompokkan ulang")

    for _, film in films.iterrows():
        path = faces_dir / f"{film['film_id']}.csv"
        if path.exists():
            combined_rows.append(pd.read_csv(path))

    print()
    if combined_rows:
        merged = pd.concat(combined_rows, ignore_index=True)
        frame = merge_rows(combined_csv, merged.to_dict("records"))
        print_summary(frame)
        print(f"wajah → {combined_csv}")
        if actor_rows_all or actors_csv.exists():
            actors = merge_actors(actors_csv, actor_rows_all)
            print(f"aktor → {actors_csv} ({len(actors)} klaster, isi actor_label manual)")
    elif combined_csv.exists():
        print_summary(pd.read_csv(combined_csv))
    else:
        print("Tidak ada hasil wajah.")
    return 0


def merge_actors(path: Path, rows: list[dict]) -> pd.DataFrame:
    """Gabungkan aktor tanpa menyisakan klaster lama dari film yang diulang.

    ``merge_rows`` hanya membuang baris dengan ``actor_key`` yang sama. Kalau
    jumlah klaster berubah setelah dijalankan ulang, kunci lama (A04, A05, ...)
    tidak pernah muncul lagi dan akan tertinggal sebagai hantu.
    """
    fresh = pd.DataFrame(rows)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=ACTOR_COLUMNS)
    if not fresh.empty and "film_id" in old.columns:
        old = old[~old["film_id"].astype(str).isin(fresh["film_id"].astype(str))]
    combined = pd.concat([old, fresh], ignore_index=True) if not fresh.empty else old
    if "actor_key" in combined.columns:
        combined = combined.sort_values("actor_key", kind="stable").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def print_summary(frame: pd.DataFrame) -> None:
    print("=" * 82)
    print(
        f"{'film':<9}{'ujaran':>8}{'wajah cukup':>13}{'median':>9}"
        f"{'embedding':>11}{'multiwajah':>12}{'aktor':>7}"
    )
    print("-" * 82)
    for film_id, group in frame.groupby("film_id", sort=True):
        actors = group.loc[group["cluster"] > 0, "actor_key"].nunique()
        print(
            f"{film_id:<9}{len(group):>8}{group['face_ok'].mean():>12.0%}"
            f"{group['face_ratio'].median():>9.2f}"
            f"{(group['emb_frames'] > 0).mean():>11.0%}"
            f"{group['multi_face_share'].mean():>12.2f}{actors:>7}"
        )
    print("=" * 82)
    print(
        f"total {len(frame)} ujaran · wajah cukup {frame['face_ok'].mean():.0%} "
        f"· embedding {frame['emb_frames'].gt(0).mean():.0%}"
    )


if __name__ == "__main__":
    sys.exit(main())
