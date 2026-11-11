#!/usr/bin/env python3
"""08_agree.py — hitung kesepakatan antar-anotator dan siapkan adjudikasi.

Membaca semua ``data/labels_*.csv`` (satu berkas per anotator, dari laptop
masing-masing), menggabungkannya dengan kunci ``(annotator, utterance_id)``,
lalu:

  1. menghitung **Cohen's kappa** untuk setiap pasangan penilai dan
     **Fleiss' kappa** untuk ujaran yang dinilai tiga orang atau lebih;
  2. memutuskan label akhir per ujaran dengan suara mayoritas;
  3. menulis daftar ujaran yang tidak punya mayoritas ke
     ``data/adjudication.csv`` untuk diputuskan manusia;
  4. menulis ``data/labels_final.csv`` yang dibaca tahap seleksi.

Kappa dihitung dari label mentah, sebelum adjudikasi: kalau dihitung setelah
keputusan koordinator, angkanya akan naik sendiri dan tidak jujur.

Adjudikasi:

  uv run python scripts/08_agree.py                    # hitung + tulis daftar
  # isi kolom decision pada data/adjudication.csv
  uv run python scripts/08_agree.py                    # terapkan keputusan

Keputusan yang sudah diisi disimpan kembali di ``adjudication.csv``, jadi
riwayatnya tidak hilang saat skrip dijalankan ulang. Kalau label mentah
berubah dan keputusan lama ingin dibuang, pakai ``--reset-decisions``.

Keluaran:

  data/labels_final.csv     satu baris per ujaran yang sudah diputuskan
  data/adjudication.csv     ujaran tanpa mayoritas + kolom decision
  data/agreement_report.md  ringkasan untuk laporan
  data/agreement_meta.json  angka kappa dan jumlah butir
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from emosi import agree
from emosi.config import Config, load_config

FINAL = "labels_final.csv"
ADJUDICATION = "adjudication.csv"
REPORT = "agreement_report.md"
META = "agreement_meta.json"


def label_files(data_dir: Path) -> list[Path]:
    """Semua berkas label per anotator, kecuali hasil tahap ini sendiri."""
    return [
        path
        for path in sorted(data_dir.glob("labels_*.csv"))
        if path.name != FINAL
    ]


def read_all(paths: list[Path]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            print(f"  {path.name}: kosong, dilewati")
            continue
        if "annotator" not in frame.columns or frame.empty:
            print(f"  {path.name}: bukan berkas label, dilewati")
            continue
        frames.append(frame)
        print(f"  {path.name}: {len(frame)} label")
    return frames


def markdown_table(columns: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def coverage(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for person, group in labels.groupby("annotator", sort=True):
        roles = group["role"].value_counts().to_dict()
        rows.append(
            {
                "anotator": person,
                "label": len(group),
                "ujaran": group["utterance_id"].nunique(),
                "milik": int(roles.get("milik", 0)),
                "gold": int(roles.get("gold", 0)),
                "silang": int(roles.get("silang", 0)),
            }
        )
    return pd.DataFrame(rows)


def kappa_rows(
    table: pd.DataFrame, *, min_items: int
) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table.itertuples():
        value = row.kappa
        if row.butir < min_items:
            tafsir = "butir kurang"
            shown = "—"
        elif value is None or (isinstance(value, float) and math.isnan(value)):
            tafsir = "tidak terdefinisi"
            shown = "—"
        else:
            tafsir = row.tafsir
            shown = f"{value:.3f}"
        rows.append([row.basis, row.metode, row.penilai, row.butir,
                     f"{row.setuju:.3f}" if row.butir >= min_items else "—",
                     shown, tafsir])
    return rows


def render_report(
    *,
    cfg: Config,
    labels: pd.DataFrame,
    votes: pd.DataFrame,
    table: pd.DataFrame,
    files: list[Path],
    min_items: int,
) -> str:
    counts = agree.status_counts(votes)
    pending = counts.get(agree.STATUS_PENDING, 0)
    multi = int((votes["n_raters"] >= 2).sum())
    lines: list[str] = []
    lines.append("# Laporan kesepakatan antar-anotator")
    lines.append("")
    lines.append(f"- dibuat: {datetime.now(UTC).isoformat(timespec='seconds')}")
    lines.append(f"- config `{cfg.sha256[:16]}…`")
    lines.append(f"- berkas label: {', '.join(path.name for path in files)}")
    lines.append(f"- label: {len(labels)} · anotator: {labels['annotator'].nunique()}")
    lines.append(f"- ujaran dinilai ≥2 orang: {multi} · ujaran: {len(votes)}")
    lines.append("")

    lines.append("## Cakupan per anotator")
    lines.append("")
    rows = [
        [row.anotator, row.label, row.ujaran, row.milik, row.gold, row.silang]
        for row in coverage(labels).itertuples()
    ]
    lines.extend(
        markdown_table(
            ["Anotator", "Label", "Ujaran", "Milik", "Gold", "Silang"], rows
        )
    )
    lines.append("")

    lines.append("## Ujaran yang dinilai bersama")
    lines.append("")
    matrix = agree.overlap_matrix(labels)
    people = list(matrix.index)
    rows = [
        [person] + [int(matrix.loc[person, other]) for other in people]
        for person in people
    ]
    lines.extend(markdown_table(["Anotator"] + [str(p) for p in people], rows))
    lines.append("")
    lines.append(
        "Angka di luar diagonal adalah jumlah ujaran yang dinilai kedua orang "
        "itu; itulah dasar kappa."
    )
    lines.append("")

    lines.append("## Kappa")
    lines.append("")
    if table.empty:
        lines.append(
            "Belum ada ujaran yang dinilai lebih dari satu orang. Kappa baru bisa "
            "dihitung setelah gold set dan tukar silang dilabeli."
        )
    else:
        lines.extend(
            markdown_table(
                ["Basis", "Metode", "Penilai", "Butir", "Setuju", "Kappa", "Tafsir"],
                kappa_rows(table, min_items=min_items),
            )
        )
        lines.append("")
        lines.append(
            f"Pasangan dengan kurang dari {min_items} ujaran bersama tidak diberi "
            "angka. Tafsir memakai Landis & Koch: <0,20 sedikit, 0,20–0,40 cukup, "
            "0,40–0,60 sedang, 0,60–0,80 baik, >0,80 sangat baik."
        )
    lines.append("")

    lines.append("## Keputusan label")
    lines.append("")
    order = (
        agree.STATUS_AGREE,
        agree.STATUS_MAJORITY,
        agree.STATUS_SINGLE,
        agree.STATUS_DECIDED,
        agree.STATUS_PENDING,
    )
    rows = [[status, counts.get(status, 0)] for status in order]
    lines.extend(markdown_table(["Status", "Ujaran"], rows))
    lines.append("")
    if pending:
        lines.append(
            f"**{pending} ujaran belum punya mayoritas.** Isi kolom `decision` di "
            "`data/adjudication.csv` (nilai: kelas, `ambigu`, atau `tidak_layak`), "
            "lalu jalankan skrip ini lagi. Ujaran itu belum masuk "
            "`labels_final.csv` sampai diputuskan."
        )
    else:
        lines.append("Semua ujaran sudah punya label akhir.")
    lines.append("")
    lines.append(
        "`tunggal` berarti hanya satu orang menilai ujaran itu; labelnya dipakai "
        "tetapi tidak punya bukti kesepakatan. `adjudikasi` berarti diputuskan "
        "manusia karena suara tidak mayoritas."
    )
    lines.append("")
    return "\n".join(lines)


def warn_invalid(labels: pd.DataFrame, allowed: set[str]) -> bool:
    unknown = agree.validate_labels(labels, allowed)
    if not unknown:
        return False
    print("Label tidak dikenal: " + ", ".join(unknown))
    print("Perbaiki label itu di berkas labels_<nama>.csv, lalu jalankan ulang.")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", default=None, help="default: paths.data_dir")
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=10,
        metavar="N",
        help="pasangan dengan kurang dari N ujaran bersama tidak diberi kappa",
    )
    parser.add_argument(
        "--reset-decisions",
        action="store_true",
        help="abaikan keputusan lama di adjudication.csv",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data_dir = (
        Path(args.data_dir).expanduser() if args.data_dir else cfg.resolve("paths.data_dir")
    )
    if not data_dir.exists():
        print(f"{data_dir} tidak ada.")
        return 1

    paths = label_files(data_dir)
    if not paths:
        print(
            f"Tidak ada labels_*.csv di {data_dir}. Kumpulkan label dari semua "
            "anotator dulu (berkas data/labels_<nama>.csv)."
        )
        return 1

    print(f"membaca {len(paths)} berkas label")
    labels = agree.load_labels(read_all(paths))

    allowed = set(cfg.classes()) | {
        str(cfg.get("labels.ambiguous_token")),
        str(cfg.get("labels.not_eligible_token")),
        "ambigu",
        "tidak_layak",
    }
    if warn_invalid(labels, allowed):
        return 1

    labels = agree.add_basis(labels)
    kappa_list, table = agree.agreement_table(labels, min_items=args.min_overlap)

    decisions: dict[str, dict] = {}
    adjudication_path = data_dir / ADJUDICATION
    if adjudication_path.exists() and not args.reset_decisions:
        decisions = agree.read_decisions(pd.read_csv(adjudication_path))
        if decisions:
            print(f"keputusan lama dipakai: {len(decisions)}")

    votes = agree.item_votes(labels)
    try:
        votes = agree.apply_decisions(votes, decisions, allowed=allowed)
    except agree.AgreementError as error:
        print(f"Adjudikasi gagal: {error}")
        return 1

    final = agree.final_rows(votes)
    adjudication = agree.adjudication_rows(votes)

    data_dir.mkdir(parents=True, exist_ok=True)
    final.to_csv(data_dir / FINAL, index=False)
    adjudication.to_csv(data_dir / ADJUDICATION, index=False)
    (data_dir / REPORT).write_text(
        render_report(
            cfg=cfg,
            labels=labels,
            votes=votes,
            table=table,
            files=paths,
            min_items=args.min_overlap,
        ),
        encoding="utf-8",
    )

    counts = agree.status_counts(votes)
    meta = {
        "generated_by": "scripts/08_agree.py",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_sha256": cfg.sha256,
        "files": [path.name for path in paths],
        "n_labels": int(len(labels)),
        "n_utterances": int(len(votes)),
        "n_multi_rater": int((votes["n_raters"] >= 2).sum()),
        "min_overlap": int(args.min_overlap),
        "kappa": [
            {
                "basis": row.basis,
                "method": row.kind,
                "raters": row.label,
                "n_items": row.n_items,
                "observed": None if math.isnan(row.observed) else round(row.observed, 4),
                "kappa": None if not row.ok else round(row.value, 4),
            }
            for row in kappa_list
        ],
        "status": counts,
        "n_pending": counts.get(agree.STATUS_PENDING, 0),
    }
    (data_dir / META).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print("=" * 78)
    print(f"{'anotator':<14}{'label':>8}{'ujaran':>9}{'milik':>8}{'gold':>7}{'silang':>8}")
    print("-" * 78)
    for row in coverage(labels).itertuples():
        print(
            f"{str(row.anotator):<14}{row.label:>8}{row.ujaran:>9}"
            f"{row.milik:>8}{row.gold:>7}{row.silang:>8}"
        )
    print("=" * 78)
    for row in kappa_rows(table, min_items=args.min_overlap):
        print(
            f"{row[0]:<8}{row[1]:<7}{row[2]:<22}{row[3]:>5} butir"
            f"  kappa {row[5]:>7}  {row[6]}"
        )
    print("-" * 78)
    print(
        " · ".join(f"{status} {count}" for status, count in counts.items())
    )
    pending = counts.get(agree.STATUS_PENDING, 0)
    if pending:
        print()
        print(f"{pending} ujaran perlu diputuskan → {data_dir / ADJUDICATION}")
        print(
            "Isi kolom decision (kelas / ambigu / tidak_layak), lalu jalankan "
            "skrip ini lagi."
        )
    print()
    print(f"label akhir → {data_dir / FINAL} ({len(final)} ujaran)")
    print(f"laporan     → {data_dir / REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
