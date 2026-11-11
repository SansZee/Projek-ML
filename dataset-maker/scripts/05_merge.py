#!/usr/bin/env python3
"""05_merge.py — bungkus kiriman sendiri, atau gabungkan kiriman semua orang.

Tahap 00–04 berjalan di laptop masing-masing anggota. Skrip ini menutup jarak
antar-laptop, tetapi hanya untuk berkas kecil: tabel dan embedding wajah
(±60 MB untuk 20 film). Film, audio, dan klip tidak pernah ikut.

Dua mode:

  anggota       --pack <nama>     membungkus hasil sendiri jadi satu zip
  koordinator   --from <folder>   memeriksa lalu menulis data/merged/

Contoh:

  # di laptop anggota, setelah 04 selesai
  uv run python scripts/05_merge.py --pack kesan

  # di laptop koordinator
  uv run python scripts/05_merge.py --from kiriman/
  uv run python scripts/05_merge.py --from kiriman/kesan.zip kiriman/budi.zip

Kalau satu film datang dari dua anggota, penggabungan berhenti. Pilih salah
satu dengan ``--keep budi`` setelah kamu yakin kiriman mana yang dipakai.

Keluaran ``data/merged/`` selalu ditulis ulang dari nol, jadi aman dijalankan
berkali-kali saat kiriman baru berdatangan. Tidak ada berkas lokal yang
ditimpa: kolom ``path`` dibuang dan diganti ``owner``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from emosi import merge
from emosi.config import Config, load_config

REPORT = "merge_report.md"
META = "merge_meta.json"


def uv_lock_sha(root: Path) -> str | None:
    path = root / "uv.lock"
    return merge.sha256_file(path) if path.exists() else None


# -----------------------------------------------------------------------------
# Mode anggota
# -----------------------------------------------------------------------------


def run_pack(cfg: Config, owner: str, out: Path | None) -> int:
    data_dir = cfg.resolve("paths.data_dir")
    for name in ("films.csv", "utterances.csv", "faces.csv"):
        if not (data_dir / name).exists():
            print(f"{data_dir / name} tidak ada. Jalankan tahap 00–04 dulu.")
            return 1

    films = pd.read_csv(data_dir / "films.csv")
    missing = [
        film
        for film in films["film_id"].astype(str)
        if not (data_dir / "faces" / f"{film}.npy").exists()
    ]
    if missing:
        print(
            "embedding wajah belum ada untuk: "
            + ", ".join(missing)
            + ". Jalankan scripts/04_face_screen.py dulu."
        )
        return 1

    files = merge.pack_files(data_dir)
    manifest = merge.build_manifest(
        owner=owner,
        data_dir=data_dir,
        config_sha256=cfg.sha256,
        uv_lock_sha256=uv_lock_sha(cfg.root) or "",
    )
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    dest = Path(out).expanduser() if out else cfg.root / "kiriman" / f"{owner}-{stamp}.zip"
    merge.write_bundle(dest, cfg.root, files, manifest)

    size_mb = dest.stat().st_size / 1048576
    print("=" * 70)
    print(f"kiriman  {dest}")
    print(f"pemilik  {owner}")
    print(f"film     {', '.join(manifest['films'])}")
    print(f"ujaran   {manifest['n_utterances']}")
    print(f"isi      {len(files)} berkas · {size_mb:.1f} MB")
    print(f"config   {cfg.sha256[:16]}…")
    print("=" * 70)
    print("Kirim zip ini ke koordinator. Jangan kirim media/ atau data/ mentah.")
    return 0


# -----------------------------------------------------------------------------
# Mode koordinator
# -----------------------------------------------------------------------------


DATA_MARKER = "data/utterances.csv"


def is_bundle(path: Path) -> bool:
    return (path / "data" / "utterances.csv").exists() or (path / merge.MANIFEST).exists()


def zip_has_data(path: Path) -> bool:
    """Zip ini kiriman data, bukan mis. bungkusan preview dari 07_annotate."""
    try:
        with zipfile.ZipFile(path) as bundle:
            return any(
                name == DATA_MARKER or name.endswith("/" + DATA_MARKER)
                for name in bundle.namelist()
            )
    except zipfile.BadZipFile:
        return False


def expand(paths: list[str]) -> list[Path]:
    """Terima zip, folder kiriman, atau folder yang berisi banyak kiriman."""
    entries: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.exists():
            raise SystemExit(f"tidak ada: {path}")
        if path.is_file() or is_bundle(path):
            entries.append(path)
            continue
        children = sorted(
            child
            for child in path.iterdir()
            if child.is_dir() or child.suffix.lower() == ".zip"
        )
        if not children:
            raise SystemExit(f"{path}: tidak ada kiriman di dalamnya")
        for child in children:
            if child.is_dir():
                if is_bundle(child):
                    entries.append(child)
                else:
                    print(f"  lewati {child.name}: bukan kiriman data")
            elif zip_has_data(child):
                entries.append(child)
            else:
                print(f"  lewati {child.name}: bukan kiriman data")
    # Jangan gabungkan kiriman yang sama dua kali.
    seen: set[Path] = set()
    unique: list[Path] = []
    for entry in entries:
        resolved = entry.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(entry)
    return unique


def load_all(entries: list[Path]) -> list[merge.Contributor]:
    contributors: list[merge.Contributor] = []
    for entry in entries:
        contributors.append(merge.load_contributor(entry))
    names = [c.owner for c in contributors]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise SystemExit(
            "nama kontributor dipakai dua kali: "
            + ", ".join(repeated)
            + ". Ganti nama folder atau zip-nya."
        )
    return contributors


def contributor_stats(contributor: merge.Contributor) -> dict:
    faces = contributor.table("faces.csv")
    utterances = contributor.table("utterances.csv")
    return {
        "owner": contributor.owner,
        "films": len(contributor.table("films.csv")),
        "utterances": len(utterances),
        "face_ok": round(float(faces["face_ok"].mean()), 3) if "face_ok" in faces else None,
        "embedding": round(float((faces["emb_frames"] > 0).mean()), 3)
        if "emb_frames" in faces
        else None,
        "actors": int(contributor.table("actors.csv")["actor_key"].nunique())
        if "actor_key" in contributor.table("actors.csv")
        else 0,
        "packed_at": str(contributor.manifest.get("packed_at") or ""),
        "config_sha256": str(contributor.manifest.get("config_sha256") or ""),
        "uv_lock_sha256": str(contributor.manifest.get("uv_lock_sha256") or ""),
    }


def write_merged(merged: merge.Merged, out_dir: Path) -> None:
    """Tulis katalog gabungan. Folder keluaran selalu dibuat ulang dari nol."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    faces_dir = out_dir / "faces"
    faces_dir.mkdir(parents=True)

    merged.films.to_csv(out_dir / "films.csv", index=False)
    merged.utterances.to_csv(out_dir / "utterances.csv", index=False)
    merged.faces.to_csv(out_dir / "faces.csv", index=False)
    merged.actors.to_csv(out_dir / "actors.csv", index=False)
    merged.scenes.to_csv(out_dir / "scenes.csv", index=False)
    if not merged.reject_log.empty:
        merged.reject_log.to_csv(out_dir / "reject_log.csv", index=False)

    for film_id, matrix in sorted(merged.embeddings.items()):
        np.save(faces_dir / f"{film_id}.npy", matrix.astype(np.float32, copy=False))
        subset = merged.faces[merged.faces["film_id"].astype(str) == film_id]
        subset.to_csv(faces_dir / f"{film_id}.csv", index=False)


def render_report(
    *,
    cfg: Config,
    contributors: list[merge.Contributor],
    merged: merge.Merged,
    problems: list[merge.Problem],
    candidates: pd.DataFrame,
    actor_distance: float,
) -> str:
    lines: list[str] = []
    lines.append("# Laporan penggabungan kiriman")
    lines.append("")
    lines.append(f"- dibuat: {datetime.now(UTC).isoformat(timespec='seconds')}")
    lines.append(f"- config `{cfg.sha256[:16]}…`")
    lines.append(f"- kontributor: {', '.join(c.owner for c in contributors)}")
    lines.append(f"- film diterima: {len(merged.films)} · ujaran: {len(merged.utterances)}")
    lines.append("")

    lines.append("## Per kontributor")
    lines.append("")
    lines.append("| Kontributor | Film | Ujaran | Wajah cukup | Embedding | Aktor | Dikemas |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for contributor in contributors:
        stats = contributor_stats(contributor)
        face_ok = "-" if stats["face_ok"] is None else f"{stats['face_ok']:.0%}"
        embedding = "-" if stats["embedding"] is None else f"{stats['embedding']:.0%}"
        packed = stats["packed_at"][:10] or "-"
        lines.append(
            f"| {stats['owner']} | {stats['films']} | {stats['utterances']} "
            f"| {face_ok} | {embedding} | {stats['actors']} | {packed} |"
        )
    lines.append("")

    lines.append("## Film di katalog")
    lines.append("")
    lines.append("| film_id | judul | tahun | owner | ujaran | wajah cukup |")
    lines.append("|---|---|---|---|---:|---:|")
    utt_count = merged.utterances.groupby("film_id").size()
    face_share = merged.faces.groupby("film_id")["face_ok"].mean()
    for row in merged.films.itertuples():
        film_id = str(row.film_id)
        lines.append(
            f"| {film_id} | {getattr(row, 'title', '')} | {getattr(row, 'year', '')} "
            f"| {row.owner} | {int(utt_count.get(film_id, 0))} "
            f"| {float(face_share.get(film_id, 0)):.0%} |"
        )
    lines.append("")

    if merged.dropped:
        lines.append("## Tidak masuk katalog")
        lines.append("")
        lines.append(
            "- film tidak lolos audisi: " + ", ".join(merged.dropped)
        )
        lines.append("")

    lines.append("## Pemeriksaan")
    lines.append("")
    errors = [p for p in problems if p.level == "error"]
    warnings = [p for p in problems if p.level == "warning"]
    if not errors and not warnings:
        lines.append("- semua pemeriksaan bersih")
    for problem in errors + warnings:
        lines.append(f"- **{problem.level}** — {problem.message}")
    lines.append("")

    lines.append("## Kandidat aktor lintas film")
    lines.append("")
    lines.append(
        f"Pasangan dengan jarak kosinus ≤ {actor_distance}. Ini usulan, bukan "
        "kebenaran: periksa manual, lalu pakai `actor_label` yang persis sama "
        "untuk aktor yang sama di kedua film."
    )
    lines.append("")
    if candidates.empty:
        lines.append("Tidak ada pasangan yang cukup mirip.")
    else:
        lines.append("| actor_a | actor_b | jarak |")
        lines.append("|---|---|---:|")
        for row in candidates.itertuples():
            lines.append(
                f"| {row.actor_a} | {row.actor_b} | {row.cosine_distance:.3f} |"
            )
    lines.append("")
    return "\n".join(lines)


def run_merge(cfg: Config, entries: list[Path], keep: str | None, out_dir: Path,
              actor_distance: float) -> int:
    contributors = load_all(entries)
    contributors, duplicate_problems = merge.resolve_duplicates(contributors, keep=keep)

    problems = duplicate_problems + merge.validate(
        contributors,
        config_sha256=cfg.sha256,
        uv_lock_sha256=uv_lock_sha(cfg.root),
    )
    for problem in problems:
        print(f"  {problem}")

    if merge.has_errors(problems):
        print()
        print("Penggabungan dibatalkan. Tidak ada berkas yang ditulis.")
        return 1

    merged = merge.merge_contributors(contributors)
    centroids = merge.actor_centroids(merged.faces, merged.embeddings)
    candidates = merge.cross_film_candidates(centroids, distance=actor_distance)

    write_merged(merged, out_dir)
    report = render_report(
        cfg=cfg,
        contributors=contributors,
        merged=merged,
        problems=problems,
        candidates=candidates,
        actor_distance=actor_distance,
    )
    (out_dir / REPORT).write_text(report, encoding="utf-8")
    meta = {
        "generated_by": "scripts/05_merge.py",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_sha256": cfg.sha256,
        "uv_lock_sha256": uv_lock_sha(cfg.root),
        "actor_distance": actor_distance,
        "contributors": [contributor_stats(c) for c in contributors],
        "n_films": int(len(merged.films)),
        "n_utterances": int(len(merged.utterances)),
        "n_face_ok": int(merged.faces["face_ok"].sum()) if "face_ok" in merged.faces else 0,
        "dropped_films": merged.dropped,
    }
    (out_dir / META).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print_summary(merged, out_dir, candidates)
    return 0


def print_summary(merged: merge.Merged, out_dir: Path, candidates: pd.DataFrame) -> None:
    print()
    print("=" * 82)
    print(f"{'film':<9}{'owner':<12}{'ujaran':>8}{'wajah cukup':>13}{'embedding':>11}")
    print("-" * 82)
    face_share = merged.faces.groupby("film_id")["face_ok"].mean()
    emb_share = merged.faces.groupby("film_id")["emb_frames"].apply(
        lambda series: (series > 0).mean()
    )
    for row in merged.films.itertuples():
        film_id = str(row.film_id)
        print(
            f"{film_id:<9}{row.owner:<12}"
            f"{int((merged.utterances['film_id'] == film_id).sum()):>8}"
            f"{float(face_share.get(film_id, 0)):>12.0%}"
            f"{float(emb_share.get(film_id, 0)):>11.0%}"
        )
    print("=" * 82)
    print(
        f"total {len(merged.films)} film · {len(merged.utterances)} ujaran · "
        f"wajah cukup {merged.faces['face_ok'].mean():.0%} · "
        f"embedding {(merged.faces['emb_frames'] > 0).mean():.0%}"
    )
    print(f"kandidat aktor lintas film: {len(candidates)} pasangan (periksa manual)")
    print(f"katalog → {out_dir}")
    print(f"laporan → {out_dir / REPORT}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--pack", metavar="NAMA", default=None, help="mode anggota: bungkus hasil sendiri"
    )
    parser.add_argument("--out", default=None, help="path zip keluaran --pack")
    parser.add_argument(
        "--from",
        dest="sources",
        nargs="+",
        default=None,
        metavar="PATH",
        help="mode koordinator: zip atau folder kiriman",
    )
    parser.add_argument(
        "--keep",
        default=None,
        metavar="NAMA",
        help="kalau satu film datang dua kali, pakai punya NAMA ini",
    )
    parser.add_argument("--out-dir", default=None, help="default: data/merged")
    parser.add_argument(
        "--actor-distance",
        type=float,
        default=0.45,
        help="ambang jarak kosinus kandidat aktor lintas film (laporan saja)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.pack:
        return run_pack(cfg, args.pack, Path(args.out) if args.out else None)
    if not args.sources:
        parser.print_help()
        print()
        print("Pilih salah satu: --pack <nama> (anggota) atau --from <kiriman> (koordinator).")
        return 1

    data_dir = cfg.resolve("paths.data_dir")
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else data_dir / "merged"
    return run_merge(
        cfg,
        expand(args.sources),
        args.keep,
        out_dir,
        args.actor_distance,
    )


if __name__ == "__main__":
    sys.exit(main())
