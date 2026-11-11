#!/usr/bin/env python3
"""12_audit.py — bandingkan label final dengan prediksi model.

Tahap ini **bukan** pelatih model. Repo pelatihan menghasilkan CSV prediksi
(`utterance_id, predicted, score`), lalu skrip ini mencari ujaran yang
labelnya berbeda dari prediksi model. Yang paling yakin salah didahulukan,
karena biasanya di situ letak label yang keliru atau kasus yang memang sulit.

Contoh:

  uv run python scripts/12_audit.py --predictions artifacts/pred_test.csv

Keluaran:

  data/disagreement.csv   ujaran yang beda, urut skor model
  data/audit_report.md    akurasi, matriks konfusi, rekap per kelas
  data/audit_meta.json    angka ringkas

Kalau prediksi belum ada, jalankan dulu pelatihan di repo model; skrip ini
tidak melatih apa pun.
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

DISAGREEMENT = "disagreement.csv"
REPORT = "audit_report.md"
META = "audit_meta.json"


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
    parser.add_argument("--predictions", required=True, help="CSV utterance_id,predicted,score")
    parser.add_argument("--data-dir", default=None, help="default paths.data_dir")
    parser.add_argument("--min-score", type=float, default=0.0, help="hanya tampilkan skor ≥ ini")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data_dir = (
        Path(args.data_dir).expanduser()
        if args.data_dir
        else cfg.resolve("paths.data_dir")
    )
    labels_path = data_dir / "labels_final.csv"
    predictions_path = Path(args.predictions).expanduser()
    for path, hint in (
        (labels_path, "08_agree.py"),
        (predictions_path, "repo pelatihan"),
    ):
        if not path.exists():
            print(f"{path} tidak ada. Jalankan {hint} dulu.")
            return 1

    labels = pd.read_csv(labels_path)
    predictions = pd.read_csv(predictions_path)
    try:
        merged = dataset_stats.compare(labels, predictions)
    except ValueError as error:
        print(f"Prediksi tidak valid: {error}")
        return 1
    if merged.empty:
        print("Tidak ada utterance_id yang cocok antara label dan prediksi.")
        return 1

    classes = cfg.classes()
    summary = dataset_stats.audit_summary(merged)
    matrix = dataset_stats.confusion(merged, classes)
    errors = merged[~merged["benar"]].sort_values("score", ascending=False)
    errors = errors[errors["score"] >= args.min_score]

    pool_path = data_dir / "pool.csv"
    text = {}
    if pool_path.exists():
        pool = pd.read_csv(pool_path)
        text = dict(
            zip(
                pool["utterance_id"].astype(str),
                pool["text"].astype(str),
                strict=True,
            )
        )

    disagreement = errors[
        [
            "utterance_id",
            "film_id",
            "label",
            "predicted",
            "score",
            "confidence",
            "n_raters",
            "status",
        ]
    ].copy()
    disagreement.insert(4, "text", disagreement["utterance_id"].astype(str).map(text))
    disagreement = disagreement.sort_values("score", ascending=False).reset_index(drop=True)
    disagreement.to_csv(data_dir / DISAGREEMENT, index=False)

    lines: list[str] = []
    lines.append("# Audit label vs prediksi model")
    lines.append("")
    lines.append(f"- dibuat: {datetime.now(UTC).isoformat(timespec='seconds')}")
    lines.append(f"- prediksi: `{predictions_path.name}`")
    lines.append(f"- ujaran dibandingkan: {summary['n']} dari {len(labels)} label final")
    lines.append(f"- akurasi: **{summary['akurasi']:.1%}** ({summary['benar']}/{summary['n']})")
    lines.append("")
    if summary["n"] < 0.5 * len(labels):
        lines.append(
            "**Peringatan:** prediksi hanya mencakup sebagian label final. "
            "Pastikan model dan label memakai split yang sama."
        )
        lines.append("")

    lines.append("## Per kelas")
    lines.append("")
    rows = []
    for label in classes:
        if label not in matrix.index:
            continue
        total = int(matrix.loc[label].sum())
        correct = int(matrix.loc[label, label]) if label in matrix.columns else 0
        accuracy = correct / total if total else 0.0
        rows.append([label, total, correct, f"{accuracy:.1%}"])
    lines.extend(markdown_table(["Kelas", "Ujaran", "Benar", "Akurasi"], rows))
    lines.append("")

    lines.append("## Matriks konfusi")
    lines.append("")
    header = ["label \\ prediksi", *classes]
    rows = [
        [label, *[int(matrix.loc[label, column]) for column in classes]]
        for label in classes
    ]
    lines.extend(markdown_table(header, rows))
    lines.append("")

    lines.append("## Yang perlu diperiksa")
    lines.append("")
    if disagreement.empty:
        lines.append("Tidak ada perbedaan pada ambang skor ini.")
    else:
        lines.append(
            f"{len(disagreement)} ujaran berbeda; "
            f"{len(errors)} dengan skor ≥ {args.min_score:g}."
        )
        lines.append("")
        preview = disagreement.head(15)
        rows = [
            [
                row.utterance_id,
                row.label,
                row.predicted,
                f"{row.score:.2f}",
                str(row.text or "")[:60],
            ]
            for row in preview.itertuples()
        ]
        lines.extend(markdown_table(["utterance_id", "label", "prediksi", "skor", "teks"], rows))
        lines.append("")
        lines.append("Daftar lengkap ada di `data/disagreement.csv`.")
    lines.append("")
    lines.append(
        "Model yang salah tidak otomatis berarti labelnya salah: bisa jadi "
        "modelnya belum belajar kasus itu. Periksa manual sebelum mengubah label."
    )
    lines.append("")
    (data_dir / REPORT).write_text("\n".join(lines), encoding="utf-8")

    meta = {
        "generated_by": "scripts/12_audit.py",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_sha256": cfg.sha256,
        "predictions": predictions_path.name,
        "n_compared": summary["n"],
        "accuracy": summary["akurasi"],
        "n_errors": int(len(errors)),
        "min_score": args.min_score,
    }
    (data_dir / META).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"dibandingkan {summary['n']} ujaran · akurasi {summary['akurasi']:.1%} "
        f"· {len(errors)} beda"
    )
    print(f"daftar  → {data_dir / DISAGREEMENT}")
    print(f"laporan → {data_dir / REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
