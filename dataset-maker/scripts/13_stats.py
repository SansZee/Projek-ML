#!/usr/bin/env python3
"""13_stats.py — statistik akhir dan dataset card.

Membaca hasil seleksi, split, katalog media, dan label final, lalu menulis
``data/dataset_card.md`` (untuk lampiran laporan) dan ``data/stats.json``.

Tidak ada angka yang dihitung ulang dari media; semuanya dari tabel, supaya
bisa dijalankan di laptop mana pun setelah data dikumpulkan.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from emosi import dataset_stats
from emosi.config import load_config

CARD = "dataset_card.md"
STATS = "stats.json"

LIMITATIONS = [
    "Genre film pada versi ini masih terbatas, sehingga kelas **takut** dan "
    "**marah** jumlahnya jauh di bawah kelas lain. Angka per kelas harus "
    "dibaca dengan itu.",
    "Label dibuat empat anotator dengan gold set dan tukar silang; kappa ada di "
    "`agreement_report.md`, bukan di kartu ini.",
    "Aktor baru terkelompok per film. Kalau `actor_label` belum diisi, split C "
    "hanya mencegah kebocoran di dalam satu film.",
    "Klip diambil dari film berbahasa Inggris; aksen dan kualitas audio "
    "mengikuti sumbernya, bukan diseimbangkan.",
    "Tidak ada data demografis aktor. Dataset ini tidak untuk analisis bias "
    "yang membutuhkan atribut tersebut.",
]


def markdown_table(columns: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", default=None, help="default paths.data_dir")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data_dir = (
        Path(args.data_dir).expanduser()
        if args.data_dir
        else cfg.resolve("paths.data_dir")
    )
    selected_path = data_dir / "selected.csv"
    if not selected_path.exists():
        print(f"{selected_path} tidak ada. Jalankan scripts/09_select.py dulu.")
        return 1

    selected = pd.read_csv(selected_path)
    classes = cfg.classes()
    def optional(name: str) -> pd.DataFrame:
        path = data_dir / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    clips = optional("clips.csv")
    splits = optional("splits.csv")
    actors = optional("actors.csv")
    labels = optional("labels_final.csv")
    pool = optional("pool.csv")

    by_class = dataset_stats.class_summary(selected, classes)
    by_film = dataset_stats.film_summary(selected)
    media = dataset_stats.media_summary(clips)
    split_tables = dataset_stats.split_summary(splits, classes) if not splits.empty else {}

    actors_filled = actors_total = 0
    if not actors.empty and "actor_label" in actors.columns:
        actors_filled = int((actors["actor_label"].astype(str).str.strip() != "").sum())
        actors_total = int(actors["actor_key"].nunique())

    label_status = (
        labels["status"].value_counts().to_dict() if not labels.empty and "status" in labels else {}
    )
    words_mean = None
    if not pool.empty and "n_words" in pool.columns:
        joined = selected.merge(
            pool[["utterance_id", "n_words"]], on="utterance_id", how="left"
        )
        words_mean = round(float(joined["n_words"].mean()), 2)

    lines: list[str] = []
    lines.append("# Dataset card — emosi film multimodal")
    lines.append("")
    lines.append(f"- dibuat: {datetime.now(UTC).isoformat(timespec='seconds')}")
    lines.append(f"- config `{cfg.sha256[:16]}…`")
    lines.append(f"- klip terpilih: **{len(selected)}**")
    lines.append(f"- film: {selected['film_id'].nunique()} · kelas: {len(classes)}")
    if words_mean is not None:
        lines.append(f"- rata-rata kata per ujaran: {words_mean}")
    lines.append("")

    lines.append("## Komposisi kelas")
    lines.append("")
    rows = [
        [
            row.label,
            row.n,
            f"{row.share:.1%}",
            f"{row.duration_mean:.2f}",
            f"{row.confidence_mean:.2f}",
            f"{row.agreement_mean:.2f}",
        ]
        for row in by_class.itertuples()
    ]
    lines.extend(
        markdown_table(
            ["Kelas", "Klip", "Porsi", "Durasi rata-rata (s)", "Keyakinan", "Kesepakatan"],
            rows,
        )
    )
    lines.append("")
    lines.append(
        "Rasio kelas terbanyak : tersedikit = **"
        f"{dataset_stats.balance(by_class['n'])}:1**. Dataset tidak seimbang; "
        "laporkan metrik per kelas, bukan hanya akurasi."
    )
    lines.append("")

    lines.append("## Kontribusi film")
    lines.append("")
    rows = [
        [row.film_id, row.title, row.n, f"{row.share:.1%}", row.classes]
        for row in by_film.itertuples()
    ]
    lines.extend(markdown_table(["film_id", "Judul", "Klip", "Porsi", "Kelas"], rows))
    lines.append("")

    if media:
        lines.append("## Media")
        lines.append("")
        lines.append(
            f"- {media['clips']} klip · {media['total_mb']:.0f} MB total\n"
            f"- rata-rata video {media['clip_mb']:.2f} MB · audio "
            f"{media['audio_kb']:.0f} KB · wajah {media['face_kb']:.0f} KB"
        )
        if media["clips"] < len(selected):
            lines.append("")
            lines.append(
                f"**Peringatan:** baru {media['clips']} dari {len(selected)} klip "
                "terpilih yang dipotong; media belum lengkap."
            )
        lines.append("")

    if split_tables:
        lines.append("## Split")
        lines.append("")
        for variant, table in split_tables.items():
            lines.append(f"**Varian {variant}**")
            lines.append("")
            rows = [
                [
                    label,
                    *[int(table.loc[split, label]) for split in dataset_stats.SPLITS],
                ]
                for label in classes
            ]
            rows.append(
                [
                    "total",
                    *[int(table.loc[split, "total"]) for split in dataset_stats.SPLITS],
                ]
            )
            lines.extend(markdown_table(["Kelas", "train", "val", "test"], rows))
            lines.append("")
        lines.append(
            "Tiga varian dilaporkan bersama; selisih A→B dan A→C adalah ukuran "
            "kebocoran film dan aktor, bukan kesalahan implementasi."
        )
        lines.append("")

    lines.append("## Kualitas label")
    lines.append("")
    if label_status:
        rows = [[status, count] for status, count in label_status.items()]
        lines.extend(markdown_table(["Status label", "Ujaran"], rows))
        lines.append("")
    lines.append(
        f"`actor_label` terisi untuk {actors_filled} dari {actors_total} klaster aktor."
    )
    if media and "status" in clips.columns:
        failed = int((clips["status"] == "gagal").sum())
        if failed:
            lines.append(f"{failed} klip gagal dipotong dan tidak masuk katalog.")
    lines.append("")

    lines.append("## Keterbatasan")
    lines.append("")
    lines.extend(f"- {item}" for item in LIMITATIONS)
    lines.append("")

    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / CARD).write_text("\n".join(lines), encoding="utf-8")

    stats = {
        "generated_by": "scripts/13_stats.py",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_sha256": cfg.sha256,
        "n_clips": int(len(selected)),
        "n_films": int(selected["film_id"].nunique()),
        "n_classes": len(classes),
        "class_counts": {row.label: int(row.n) for row in by_class.itertuples()},
        "balance": dataset_stats.balance(by_class["n"]),
        "duration_mean": round(float(selected["duration_s"].mean()), 2),
        "duration_median": round(float(selected["duration_s"].median()), 2),
        "media": media,
        "splits": {
            variant: {
                split: int(table.loc[split, "total"]) for split in dataset_stats.SPLITS
            }
            for variant, table in split_tables.items()
        },
        "label_status": {str(key): int(value) for key, value in label_status.items()},
        "actor_labels_filled": actors_filled,
        "actor_clusters": actors_total,
    }
    (data_dir / STATS).write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"klip {len(selected)} · film {stats['n_films']} · "
          f"rasio kelas {stats['balance']}:1")
    if media:
        print(f"media {media['total_mb']:.0f} MB")
    print(f"kartu  → {data_dir / CARD}")
    print(f"statistik → {data_dir / STATS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
