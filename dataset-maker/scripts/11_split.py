#!/usr/bin/env python3
"""11_split.py — bagi klip terpilih jadi train/val/test, tiga varian.

  A  acak per ujaran (stratifikasi kelas) — angka optimistis
  B  per film — satu film hanya di satu split
  C  per aktor — satu aktor hanya di satu split

Ketiganya ditulis ke ``data/splits.csv`` supaya repo pelatihan bisa melaporkan
tiga angka, bukan satu. Selisih A→B dan A→C itulah ukuran kebocoran.

Untuk varian C, isi ``actor_label`` di ``data/actors.csv`` lebih dulu. Kalau
kosong, skrip memakai klaster per film (``ABT13-A01``); kebocoran aktor yang
sama di dua film baru tertutup kalau namanya diisi.

Contoh:

  uv run python scripts/11_split.py
  uv run python scripts/11_split.py --seed 7
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from emosi import split
from emosi.config import Config, load_config

SPLITS_CSV = "splits.csv"
REPORT = "split_report.md"
META = "split_meta.json"

SPLITS_COLUMNS = [
    "utterance_id",
    "clip_id",
    "film_id",
    "label",
    "split_A",
    "split_B",
    "split_C",
]


def actor_labels(data_dir: Path) -> tuple[dict[str, str], int, int]:
    path = data_dir / "actors.csv"
    if not path.exists():
        return {}, 0, 0
    actors = pd.read_csv(path)
    if "actor_label" not in actors.columns:
        return {}, 0, 0
    filled = actors[actors["actor_label"].astype(str).str.strip() != ""]
    mapping = {
        str(row.actor_key): str(row.actor_label).strip()
        for row in filled.itertuples()
    }
    return mapping, int(len(filled)), int(actors["actor_key"].nunique())


def markdown_table(columns: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def render_report(
    *,
    cfg: Config,
    frame: pd.DataFrame,
    variants: dict[str, split.Assignment],
    ratios: tuple[float, float, float],
    classes: list[str],
    seed: int,
    seeds: list[int],
    stability: dict[str, float],
    actors_filled: int,
    actors_total: int,
    keys: pd.Series,
) -> str:
    lines: list[str] = []
    lines.append("# Laporan pembagian data")
    lines.append("")
    lines.append(f"- dibuat: {datetime.now(UTC).isoformat(timespec='seconds')}")
    lines.append(f"- config `{cfg.sha256[:16]}…`")
    lines.append(f"- klip: {len(frame)} · kelas: {len(classes)}")
    lines.append(
        f"- target: train {ratios[0]:.0%} · val {ratios[1]:.0%} · test {ratios[2]:.0%}"
    )
    lines.append(f"- seed: {seed} · jumlah seed diuji: {len(seeds)}")
    lines.append("")

    lines.append("## Ringkasan varian")
    lines.append("")
    rows = []
    for name, variant in variants.items():
        shares = split.share(variant.split)
        films = split.leakage(variant.split, frame["film_id"])
        actor_leak = split.leakage(variant.split, keys)
        rows.append(
            [
                name,
                f"{shares['train']:.0%}",
                f"{shares['val']:.0%}",
                f"{shares['test']:.0%}",
                f"{films['leaked']}/{films['groups']}",
                f"{actor_leak['leaked']}/{actor_leak['groups']}",
            ]
        )
    lines.extend(
        markdown_table(
            ["Varian", "train", "val", "test", "Film bocor", "Aktor bocor"], rows
        )
    )
    lines.append("")
    lines.append(
        "Varian A memang membocorkan film dan aktor; itu gunanya pembanding. "
        "Pada B, film tidak boleh bocor. Pada C, aktor tidak boleh bocor."
    )
    lines.append("")

    lines.append("## Sebaran kelas")
    lines.append("")
    for name, variant in variants.items():
        table = split.class_table(variant.split, frame["label"], classes)
        lines.append(f"**{name}**")
        lines.append("")
        header = ["Kelas", "train", "val", "test"]
        rows = [
            [label, *[int(table.loc[label, column]) for column in split.SPLITS]]
            for label in classes
        ]
        lines.extend(markdown_table(header, rows))
        lines.append("")

    lines.append("## Stabilitas varian A")
    lines.append("")
    lines.append(
        f"Porsi train pada {len(seeds)} seed: {stability['min']:.1%}–"
        f"{stability['max']:.1%} (target {ratios[0]:.0%})."
    )
    lines.append("")

    lines.append("## Aktor")
    lines.append("")
    lines.append(
        f"`actor_label` terisi untuk {actors_filled} dari {actors_total} klaster."
    )
    if actors_filled < actors_total:
        lines.append("")
        lines.append(
            "Klaster tanpa nama dipakai apa adanya untuk split C, jadi aktor yang "
            "sama di dua film masih bisa bocor. Isi `data/actors.csv` sebelum "
            "angka split C dilaporkan."
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--selected", default=None, help="default data/selected.csv")
    parser.add_argument("--out-dir", default=None, help="default paths.data_dir")
    parser.add_argument("--seed", type=int, default=None, help="menimpa project.seed")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    base_dir = cfg.resolve("paths.data_dir")
    data_dir = Path(args.out_dir).expanduser() if args.out_dir else base_dir
    selected_path = (
        Path(args.selected).expanduser() if args.selected else base_dir / "selected.csv"
    )
    if not selected_path.exists():
        print(f"{selected_path} tidak ada. Jalankan scripts/09_select.py dulu.")
        return 1

    frame = pd.read_csv(selected_path)
    frame["label"] = frame["label"].astype(str)
    classes = cfg.classes()
    ratios = tuple(float(value) for value in cfg.get("split.ratios"))
    seed = int(args.seed if args.seed is not None else cfg.get("project.seed"))
    n_seeds = max(1, int(cfg.get("split.n_seeds")))
    seeds = [seed + index for index in range(n_seeds)]

    labels, filled, total = actor_labels(base_dir)
    if total and filled < total:
        print(
            f"Peringatan: actor_label terisi {filled}/{total} klaster. "
            "Split C memakai klaster per film untuk sisanya."
        )
    keys = split.group_keys(frame, actor_labels=labels)

    variant_a = split.Assignment(
        "A acak per ujaran", split.assign_random(frame, ratios=ratios, seed=seed)
    )
    variant_b = split.Assignment(
        "B per film", split.assign_groups(frame["film_id"], ratios=ratios)
    )
    variant_c = split.Assignment(
        "C per aktor", split.assign_groups(keys, ratios=ratios)
    )
    variants = {variant.name: variant for variant in (variant_a, variant_b, variant_c)}

    shares = [
        split.share(split.assign_random(frame, ratios=ratios, seed=other))
        for other in seeds
    ]
    stability = {
        "min": min(share["train"] for share in shares),
        "max": max(share["train"] for share in shares),
    }

    table = pd.DataFrame(
        {
            "utterance_id": frame["utterance_id"].astype(str),
            "clip_id": frame["clip_id"].astype(str),
            "film_id": frame["film_id"].astype(str),
            "label": frame["label"],
            "split_A": variant_a.split.values,
            "split_B": variant_b.split.values,
            "split_C": variant_c.split.values,
        }
    )[SPLITS_COLUMNS]
    table = table.sort_values(
        ["film_id", "utterance_id"], kind="stable"
    ).reset_index(drop=True)

    data_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(data_dir / SPLITS_CSV, index=False)
    (data_dir / REPORT).write_text(
        render_report(
            cfg=cfg,
            frame=frame,
            variants=variants,
            ratios=ratios,
            classes=classes,
            seed=seed,
            seeds=seeds,
            stability=stability,
            actors_filled=filled,
            actors_total=total,
            keys=keys,
        ),
        encoding="utf-8",
    )
    meta = {
        "generated_by": "scripts/11_split.py",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_sha256": cfg.sha256,
        "seed": seed,
        "seeds": seeds,
        "ratios": list(ratios),
        "n_clips": int(len(table)),
        "shares": {name: split.share(variant.split) for name, variant in variants.items()},
        "leakage": {
            name: {
                "films": split.leakage(variant.split, frame["film_id"]),
                "actors": split.leakage(variant.split, keys),
            }
            for name, variant in variants.items()
        },
        "actor_labels_filled": filled,
        "actor_clusters": total,
    }
    (data_dir / META).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print("=" * 78)
    print(
        f"{'varian':<22}{'train':>8}{'val':>7}{'test':>7}"
        f"{'film bocor':>12}{'aktor bocor':>13}"
    )
    print("-" * 78)
    for name, variant in variants.items():
        shares_here = split.share(variant.split)
        films = split.leakage(variant.split, frame["film_id"])
        actors = split.leakage(variant.split, keys)
        film_text = f"{films['leaked']}/{films['groups']}"
        actor_text = f"{actors['leaked']}/{actors['groups']}"
        print(
            f"{name:<22}{shares_here['train']:>7.0%}{shares_here['val']:>7.0%}"
            f"{shares_here['test']:>7.0%}{film_text:>12}{actor_text:>13}"
        )
    print("=" * 78)
    print(
        f"stabilitas A (train) pada {len(seeds)} seed: "
        f"{stability['min']:.1%}–{stability['max']:.1%}"
    )
    print(f"split  → {data_dir / SPLITS_CSV}")
    print(f"laporan → {data_dir / REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
