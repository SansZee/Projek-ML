#!/usr/bin/env python3
"""10b_klip.py — bungkus klip untuk dikirim, lalu gabungkan di satu laptop.

``10_cut.py`` berjalan di laptop pemilik film, jadi hasilnya terpencar. Dua
mode di sini menutup celahnya:

  pemilik      --pack <nama>   bungkus klip yang ada di laptop ini
  koordinator  --gabung        ekstrak semua zip dan susun katalog final

Contoh:

  # di laptop anggota, setelah 10_cut.py selesai
  uv run python scripts/10b_klip.py --pack kesan

  # di laptop koordinator, setelah zip dari semua orang terkumpul
  uv run python scripts/10b_klip.py --gabung

Zip berisi ``clips/``, ``audio/``, ``faces/``, ``clips.csv`` (hanya baris
milik pengirim) dan ``manifest.json``. Tanpa kompresi: mp4 dan wav sudah
padat, jadi membungkus 1 GB selesai dalam hitungan detik.

``--gabung`` menolak zip yang tumpang tindih, film yang salah pemilik, berkas
hilang, atau ukuran yang tidak cocok. Setelah selesai, ``data/clips.csv``
berisi seluruh klip terpilih: ``ok`` untuk yang terkumpul, ``hilang`` untuk
yang belum dikirim siapa pun.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from emosi import clips
from emosi.config import load_config

SELECTED = "selected.csv"
FILMS = "films.csv"
POOL = "pool.csv"
CATALOG = "clips.csv"


def local_films(data_dir: Path) -> set[str]:
    path = data_dir / FILMS
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["film_id"].astype(str))


def run_pack(cfg, args) -> int:
    data_dir = cfg.resolve("paths.data_dir")
    media_dir = cfg.resolve("paths.media_dir")
    catalog_path = data_dir / CATALOG
    if not catalog_path.exists():
        print(f"{catalog_path} tidak ada. Jalankan 10_cut.py dulu di laptop ini.")
        return 1
    catalog = pd.read_csv(catalog_path)
    films = local_films(data_dir)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    dest = Path("kiriman") / f"clips-{args.pack}-{stamp}.zip"
    try:
        count, size = clips.pack(
            catalog, media_dir, dest, owner=args.pack, films=films or None
        )
    except clips.ClipError as error:
        print(f"Gagal membungkus: {error}")
        return 1
    print(f"{count} klip · {size / 1048576:.0f} MB → {dest}")
    print("Kirim zip itu ke koordinator; ia menjalankan --gabung.")
    return 0


def run_merge(cfg, args) -> int:
    data_dir = cfg.resolve("paths.data_dir")
    media_dir = cfg.resolve("paths.media_dir")
    selected_path = data_dir / SELECTED
    if not selected_path.exists():
        print(f"{selected_path} tidak ada. Jalankan 09_select.py dulu.")
        return 1
    zips = [Path(item).expanduser() for item in args.zip] if args.zip else sorted(
        Path("kiriman").glob("clips-*.zip")
    )
    if not zips:
        print("Tidak ada zip klip di kiriman/. Taruh berkas clips-<nama>-<tanggal>.zip.")
        return 1

    films = None
    pool_path = data_dir / POOL
    if pool_path.exists():
        pool = pd.read_csv(pool_path)
        if {"film_id", "owner"} <= set(pool.columns):
            films = pool[["film_id", "owner"]].drop_duplicates("film_id")
    try:
        merged, report = clips.merge_bundles(zips, media_dir, films=films)
    except clips.ClipError as error:
        print(f"Gagal menggabungkan: {error}")
        return 1

    selected = pd.read_csv(selected_path)
    catalog, missing = clips.final_catalog(merged, selected)
    catalog.to_csv(data_dir / CATALOG, index=False)

    print(f"{'zip':<44}{'pengirim':<10}{'klip':>6}{'MB':>8}")
    print("-" * 68)
    for row in report:
        print(f"{row['zip']:<44}{row['owner']:<10}{row['klip']:>6}{row['mb']:>8.1f}")
    print("-" * 68)
    total_mb = sum(row["mb"] for row in report)
    print(f"terkumpul {len(merged)} klip · {total_mb:.0f} MB")
    print(f"terpilih {len(selected)} klip · hilang {len(missing)}")
    if len(missing):
        owners = pd.read_csv(pool_path)[["film_id", "owner"]].drop_duplicates("film_id")
        grouped = (
            missing.merge(owners, on="film_id", how="left")
            .groupby(["owner", "film_id"])
            .size()
        )
        print("\nBelum dikirim:")
        for (owner, film), count in grouped.items():
            print(f"  {owner}: {film} ({count} klip)")
    print(f"\nkatalog → {data_dir / CATALOG}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pack", metavar="NAMA", help="bungkus klip milik NAMA")
    parser.add_argument("--gabung", action="store_true", help="gabungkan zip dari semua orang")
    parser.add_argument(
        "--zip", action="append", default=None, help="path zip tertentu (boleh diulang)"
    )
    args = parser.parse_args(argv)

    if not args.pack and not args.gabung:
        parser.print_help()
        return 1
    cfg = load_config(args.config)
    if args.pack:
        return run_pack(cfg, args)
    return run_merge(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
