#!/usr/bin/env python3
"""09_select.py — pilih 600 ujaran per kelas dari label final.

Dijalankan **sekali** oleh koordinator, setelah `08_agree.py` selesai (tidak
ada lagi `perlu_adjudikasi`). Kalau setiap orang menyeleksi sendiri, kuota
kelas habis di film yang sama.

Aturan:

  - hanya label kelas (bukan `ambigu` / `tidak_layak`);
  - `confidence >= selection.min_label_conf`;
  - maksimal `selection.max_film_share_per_class` bagian tiap kelas dari satu
    film — dengan 600/kelas dan 15%, berarti 90 ujaran per film per kelas;
  - pengambilan bergiliran antar film supaya sebarannya merata.

Dengan tiga film, plafonnya 5 × 3 × 90 = 1.350 ujaran. Kekurangan dilaporkan
per kelas; skrip tidak menurunkan batas 15% diam-diam.

Keluaran:

  data/selected.csv          satu baris per klip terpilih + clip_id
  data/selection_report.md   kuota per kelas dan kontribusi per film
  data/selection_meta.json   ambang dan jumlah

Contoh:

  uv run python scripts/09_select.py
  uv run python scripts/09_select.py --per-class 20 --out-dir /tmp/uji
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from emosi import select
from emosi.config import Config, load_config

SELECTED = "selected.csv"
REPORT = "selection_report.md"
META = "selection_meta.json"

POOL_COLUMNS = [
    "utterance_id",
    "owner",
    "title",
    "start_s",
    "end_s",
    "duration_s",
    "actor_key",
    "emb_frames",
    "qc_pass",
]


def load_inputs(labels_path: Path, pool_path: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    pool = pd.read_csv(pool_path)
    frame = labels.merge(
        pool[POOL_COLUMNS],
        on="utterance_id",
        how="left",
        validate="one_to_one",
    )
    missing = frame["duration_s"].isna()
    if missing.any():
        print(f"Peringatan: {int(missing.sum())} label tidak ada di pool.csv, dibuang.")
        frame = frame[~missing]
    if "qc_pass" in frame.columns:
        failed = ~frame["qc_pass"].fillna(False).astype(bool)
        if failed.any():
            print(f"Peringatan: {int(failed.sum())} label berasal dari ujaran gagal QC, dibuang.")
            frame = frame[~failed]
    return frame.reset_index(drop=True)


def render_report(
    *,
    cfg: Config,
    frame: pd.DataFrame,
    result: select.Selection,
    dropped: pd.Series,
    per_class: int,
    max_share: float,
    min_conf: float,
    classes: list[str],
    seed: int,
) -> str:
    counts = select.class_counts(result.rows, classes)
    lines: list[str] = []
    lines.append("# Laporan seleksi berkuota")
    lines.append("")
    lines.append(f"- dibuat: {datetime.now(UTC).isoformat(timespec='seconds')}")
    lines.append(f"- config `{cfg.sha256[:16]}…`")
    lines.append(f"- kandidat setelah saringan: {len(frame)}")
    lines.append(f"- terpilih: {len(result.rows)}")
    lines.append(f"- target: {per_class} per kelas · batas {max_share:.0%} per film per kelas")
    lines.append(f"- keyakinan minimal: {min_conf}")
    lines.append("")

    lines.append("## Per kelas")
    lines.append("")
    lines.append("| Kelas | Kandidat | Terpilih | Kurang | Film |")
    lines.append("|---|---:|---:|---:|---:|")
    for label in classes:
        available = int((frame["label"] == label).sum())
        films = int(
            result.rows.loc[result.rows["label"] == label, "film_id"].nunique()
        ) if len(result.rows) else 0
        lines.append(
            f"| {label} | {available} | {counts[label]} "
            f"| {result.shortfall.get(label, 0)} | {films} |"
        )
    lines.append("")
    if not result.complete:
        lines.append(
            "Kuota belum penuh. Penyebabnya batas per film: dengan "
            f"{max_share:.0%} × {per_class} = {int(per_class * max_share)} ujaran "
            "per film per kelas, jumlah film yang tersedia belum cukup. "
            "Tambahkan film, bukan menurunkan batas."
        )
        lines.append("")

    lines.append("## Kontribusi per film")
    lines.append("")
    if result.per_film.empty:
        lines.append("Tidak ada klip terpilih.")
    else:
        columns = ["film_id", *classes, "total"]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join("---" for _ in columns) + "|")
        for row in result.per_film.itertuples(index=False):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines.append("")

    lines.append("## Yang dibuang sebelum seleksi")
    lines.append("")
    if dropped.empty:
        lines.append("Tidak ada.")
    else:
        lines.append("| Alasan | Baris |")
        lines.append("|---|---:|")
        for reason, count in dropped.items():
            lines.append(f"| `{reason}` | {int(count)} |")
    lines.append("")
    lines.append(
        "`bukan_kelas` mencakup `ambigu` dan `tidak_layak`. "
        "`keyakinan_rendah` berarti di bawah ambang, bukan labelnya salah."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--labels", default=None, help="default data/labels_final.csv")
    parser.add_argument("--pool", default=None, help="default data/pool.csv")
    parser.add_argument("--out-dir", default=None, help="default paths.data_dir")
    parser.add_argument("--per-class", type=int, default=None, help="menimpa selection.per_class")
    parser.add_argument(
        "--max-share", type=float, default=None, help="menimpa selection.max_film_share_per_class"
    )
    parser.add_argument(
        "--min-conf", type=float, default=None, help="menimpa selection.min_label_conf"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    base_dir = cfg.resolve("paths.data_dir")
    data_dir = Path(args.out_dir).expanduser() if args.out_dir else base_dir
    labels_path = (
        Path(args.labels).expanduser() if args.labels else base_dir / "labels_final.csv"
    )
    pool_path = Path(args.pool).expanduser() if args.pool else base_dir / "pool.csv"

    for path, hint in ((labels_path, "08_agree.py"), (pool_path, "06_qc.py")):
        if not path.exists():
            print(f"{path} tidak ada. Jalankan scripts/{hint} dulu.")
            return 1

    classes = cfg.classes()
    per_class = int(
        args.per_class if args.per_class is not None else cfg.get("selection.per_class")
    )
    max_share = float(
        args.max_share
        if args.max_share is not None
        else cfg.get("selection.max_film_share_per_class")
    )
    min_conf = float(
        args.min_conf
        if args.min_conf is not None
        else cfg.get("selection.min_label_conf")
    )
    seed = int(cfg.get("project.seed"))

    frame = load_inputs(labels_path, pool_path)
    eligible, dropped = select.eligible(frame, classes=classes, min_conf=min_conf)
    if eligible.empty:
        print("Tidak ada label yang memenuhi syarat seleksi.")
        return 1

    result = select.select_rows(
        eligible, classes=classes, per_class=per_class, max_share=max_share
    )
    if result.rows.empty:
        print("Tidak ada klip yang bisa dipilih.")
        return 1

    data_dir.mkdir(parents=True, exist_ok=True)
    result.rows.to_csv(data_dir / SELECTED, index=False)
    (data_dir / REPORT).write_text(
        render_report(
            cfg=cfg,
            frame=eligible,
            result=result,
            dropped=dropped,
            per_class=per_class,
            max_share=max_share,
            min_conf=min_conf,
            classes=classes,
            seed=seed,
        ),
        encoding="utf-8",
    )
    counts = select.class_counts(result.rows, classes)
    meta = {
        "generated_by": "scripts/09_select.py",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_sha256": cfg.sha256,
        "labels": labels_path.name,
        "pool": pool_path.name,
        "seed": seed,
        "per_class": per_class,
        "max_share": max_share,
        "min_conf": min_conf,
        "n_candidates": int(len(eligible)),
        "n_selected": int(len(result.rows)),
        "class_counts": counts,
        "shortfall": result.shortfall,
        "complete": result.complete,
    }
    (data_dir / META).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print("=" * 72)
    print(f"{'kelas':<10}{'kandidat':>10}{'terpilih':>10}{'kurang':>9}{'film':>7}")
    print("-" * 72)
    for label in classes:
        available = int((eligible["label"] == label).sum())
        films = int(result.rows.loc[result.rows["label"] == label, "film_id"].nunique())
        print(
            f"{label:<10}{available:>10}{counts[label]:>10}"
            f"{result.shortfall.get(label, 0):>9}{films:>7}"
        )
    print("=" * 72)
    print(f"total {len(result.rows)} klip · {result.rows['film_id'].nunique()} film")
    if not result.complete:
        print(
            "Kuota belum penuh karena batas "
            f"{max_share:.0%} per film per kelas ({int(per_class * max_share)} ujaran)."
        )
        print("Tambahkan film; jangan turunkan batas.")
    print(f"terpilih → {data_dir / SELECTED}")
    print(f"laporan  → {data_dir / REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
