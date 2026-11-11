#!/usr/bin/env python3
"""06_qc.py — saring katalog gabungan jadi kolam kandidat yang layak dilabeli.

Masukan ``data/merged/`` (hasil ``05_merge.py``), keluaran ``data/pool.csv``.
Tahap ini tidak membuka media sama sekali: semua angka yang dibutuhkan
(``face_ratio``, ``snr_db``, ``clip_ratio``) sudah diukur tahap 00 dan 04.

Dua lapis penyaringan:

  per ujaran   teks tidak kosong, ``n_words ≥ qc.min_words``,
               ``face_ratio ≥ qc.face_ratio_min``
  per film     ``snr_db ≥ qc.snr_min_db``, ``clip_ratio ≤ qc.clip_ratio_max``

Ambang wajah di sini sengaja lebih longgar (0,50) daripada ambang tahap 04
(0,60). Tahap 04 menyaring cepat supaya embedding tidak dihitung dari frame
buruk; tahap 06 memutuskan apa yang benar-benar dilabeli, dan wajah 50% frame
masih cukup untuk dinilai manusia.

Selain itu skrip menyusun rencana adegan: pelabelan berjalan per adegan utuh,
jadi yang dibatasi adalah adegan, bukan ujaran lepas. Batas per film diturunkan
dari ``selection`` (450 ujaran untuk 600/kelas dan 15% per film), dan film
horor/thriller mendapat 1,5× karena di situ kelas takut dan marah paling
banyak. Semua ujaran yang lolos tetap ditulis ke ``pool.csv`` walau adeganya
tidak terpilih, supaya rencana bisa diubah tanpa mengulang penggabungan.

Keluaran:

  data/pool.csv          satu baris per ujaran + qc_pass, qc_reason, label_plan
  data/pool_scenes.csv   adegan yang dipilih beserta jumlah ujaran layaknya
  data/qc_report.md      ringkasan untuk dibaca manusia
  data/qc_meta.json      ambang, batas, dan hash config yang dipakai

Contoh:

  uv run python scripts/06_qc.py
  uv run python scripts/06_qc.py --cap 300
  uv run python scripts/06_qc.py --no-cap            # rencanakan semua adegan
  uv run python scripts/06_qc.py --merged-dir data/uji/merged
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from emosi import qc
from emosi.config import Config, load_config

REPORT = "qc_report.md"
META = "qc_meta.json"
POOL = "pool.csv"
SCENES = "pool_scenes.csv"


def settings_from(cfg: Config) -> qc.QcSettings:
    return qc.QcSettings(
        face_ratio_min=float(cfg.get("qc.face_ratio_min")),
        snr_min_db=float(cfg.get("qc.snr_min_db")),
        clip_ratio_max=float(cfg.get("qc.clip_ratio_max")),
        min_words=int(cfg.get("qc.min_words")),
    )


def caps_from(
    cfg: Config,
    films: pd.DataFrame,
    *,
    override: int | None,
    rare_multiplier: float,
    no_cap: bool,
) -> dict[str, int] | None:
    if no_cap:
        return None
    per_class = int(cfg.get("selection.per_class"))
    max_share = float(cfg.get("selection.max_film_share_per_class"))
    n_classes = len(cfg.classes())
    caps: dict[str, int] = {}
    for film in films.itertuples():
        if override is not None:
            caps[str(film.film_id)] = int(override)
        else:
            caps[str(film.film_id)] = qc.film_cap(
                per_class=per_class,
                max_share=max_share,
                n_classes=n_classes,
                genre=getattr(film, "genre", ""),
                rare_multiplier=rare_multiplier,
            )
    return caps


def merged_config(merged_dir: Path) -> str | None:
    path = merged_dir / "merge_meta.json"
    if not path.exists():
        return None
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("config_sha256") or "")
    except json.JSONDecodeError:
        return None


def render_report(
    *,
    cfg: Config,
    pool: pd.DataFrame,
    plan: pd.DataFrame,
    caps: dict[str, int] | None,
    settings: qc.QcSettings,
    stale_config: str | None,
    rare_films: list[str],
) -> str:
    summary = qc.cap_summary(pool, caps)
    reasons = qc.reason_counts(pool)
    lines: list[str] = []
    lines.append("# Laporan penyaringan kolam kandidat")
    lines.append("")
    lines.append(f"- dibuat: {datetime.now(UTC).isoformat(timespec='seconds')}")
    lines.append(f"- config `{cfg.sha256[:16]}…`")
    lines.append(f"- ujaran diperiksa: {len(pool)} · lolos QC: {int(pool['qc_pass'].sum())}")
    lines.append(f"- ujaran masuk rencana label: {int(pool['label_plan'].sum())}")
    if caps is None:
        lines.append("- batas per film: tidak dipakai (semua adegan)")
    else:
        unique = sorted(set(caps.values()))
        lines.append(f"- batas per film: {', '.join(str(value) for value in unique)} ujaran")
    lines.append("")

    lines.append("## Ambang")
    lines.append("")
    lines.append("| Pemeriksaan | Ambang |")
    lines.append("|---|---|")
    lines.append(f"| `face_ratio` minimum | {settings.face_ratio_min:.2f} |")
    lines.append(f"| `snr_db` minimum (per film) | {settings.snr_min_db:g} |")
    lines.append(f"| `clip_ratio` maksimum (per film) | {settings.clip_ratio_max:g} |")
    lines.append(f"| jumlah kata minimum | {settings.min_words} |")
    lines.append("")

    if stale_config:
        lines.append("## Peringatan")
        lines.append("")
        lines.append(
            f"Katalog gabungan dibuat dengan config `{stale_config[:16]}…`, bukan "
            f"`{cfg.sha256[:16]}…`. Ambang di laporan ini yang berlaku, tetapi "
            "embedding dan face ratio mungkin berasal dari config lama."
        )
        lines.append("")

    lines.append("## Per film")
    lines.append("")
    lines.append(
        "| film_id | ujaran | lolos QC | masuk rencana | batas | adegan dipilih |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    selected_scenes = plan[plan["selected"]].groupby("film_id").size()
    for row in summary.itertuples():
        cap = "-" if caps is None else str(caps.get(str(row.film_id), "-"))
        chosen = int(selected_scenes.get(row.film_id, 0))
        lines.append(
            f"| {row.film_id} | {row.utterances} | {row.qc_pass} "
            f"| {row.label_plan} | {cap} | {chosen} |"
        )
    lines.append("")

    lines.append("## Alasan gagal")
    lines.append("")
    if reasons.empty:
        lines.append("Tidak ada ujaran yang gagal.")
    else:
        lines.append("| Alasan | Ujaran |")
        lines.append("|---|---:|")
        for reason, count in reasons.items():
            lines.append(f"| `{reason}` | {int(count)} |")
        lines.append("")
        lines.append(
            "Satu ujaran bisa gagal karena lebih dari satu hal, jadi jumlahnya "
            "tidak sama dengan jumlah ujaran gagal."
        )
    lines.append("")

    lines.append("## Catatan")
    lines.append("")
    if caps is not None:
        over = summary[summary["label_plan"] > summary["cap"]]
        if over.empty:
            lines.append(
                "- Tidak ada film yang melewati batas: satu adegan terakhir masih "
                "muat di setiap film."
            )
        else:
            lines.append(
                "- Film yang sedikit melewati batas karena adegan selalu diambil "
                "utuh: " + ", ".join(
                    f"{row.film_id} ({row.label_plan} > {int(row.cap)})"
                    for row in over.itertuples()
                )
                + "."
            )
        if rare_films:
            lines.append(
                "- Batas lebih longgar untuk film genre langka: "
                + ", ".join(rare_films)
                + "."
            )
    lines.append(
        "- `label_plan` = lolos QC **dan** adeganya terpilih. Adegan tetap "
        "diputar utuh di aplikasi pelabelan, tetapi ujaran yang gagal QC tidak "
        "perlu diklik karena tidak akan lolos seleksi."
    )
    lines.append(
        "- Ujaran yang lolos QC tetapi adeganya tidak terpilih tetap ada di "
        "`pool.csv`, jadi rencana bisa diubah tanpa mengulang tahap 05."
    )
    lines.append(
        "- Ujaran tanpa embedding (`emb_frames = 0`) tetap bisa dilabeli untuk "
        "audio dan teks, tetapi tidak bisa dipakai split C."
    )
    lines.append("")
    return "\n".join(lines)


def print_summary(pool: pd.DataFrame, plan: pd.DataFrame, caps: dict[str, int] | None) -> None:
    summary = qc.cap_summary(pool, caps)
    selected_scenes = plan[plan["selected"]].groupby("film_id").size()
    print()
    print("=" * 88)
    print(
        f"{'film':<9}{'ujaran':>8}{'lolos QC':>10}{'rencana':>9}{'batas':>8}"
        f"{'adegan':>8}{'embedding':>11}"
    )
    print("-" * 88)
    for row in summary.itertuples():
        cap = "-" if caps is None else str(caps.get(str(row.film_id), "-"))
        print(
            f"{str(row.film_id):<9}{row.utterances:>8}{row.qc_pass:>10}"
            f"{row.label_plan:>9}{cap:>8}"
            f"{int(selected_scenes.get(row.film_id, 0)):>8}{row.emb_share:>10.0%}"
        )
    print("=" * 88)
    print(
        f"total {len(pool)} ujaran · lolos QC {int(pool['qc_pass'].sum())} "
        f"({pool['qc_pass'].mean():.0%}) · masuk rencana {int(pool['label_plan'].sum())} "
        f"· adegan {int(plan['selected'].sum())}/{len(plan)}"
    )
    reasons = qc.reason_counts(pool)
    if not reasons.empty:
        print("gagal: " + " · ".join(f"{name} {count}" for name, count in reasons.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--merged-dir", default=None, help="default: data/merged"
    )
    parser.add_argument(
        "--out-dir", default=None, help="default: paths.data_dir (data/)"
    )
    parser.add_argument(
        "--cap", type=int, default=None, metavar="N",
        help="batas ujaran per film untuk semua film (mengalahkan hitungan config)",
    )
    parser.add_argument(
        "--rare-multiplier", type=float, default=1.5, metavar="X",
        help="pengali batas untuk film horor/thriller (default 1.5)",
    )
    parser.add_argument(
        "--no-cap", action="store_true", help="rencanakan semua adegan, tanpa batas"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data_dir = cfg.resolve("paths.data_dir")
    merged_dir = (
        Path(args.merged_dir).expanduser() if args.merged_dir else data_dir / "merged"
    )
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else data_dir

    required = ("films.csv", "utterances.csv", "faces.csv", "scenes.csv")
    missing = [name for name in required if not (merged_dir / name).exists()]
    if missing:
        print(f"{merged_dir} belum lengkap ({', '.join(missing)}).")
        print("Jalankan scripts/05_merge.py --from kiriman/ dulu.")
        return 1

    films = pd.read_csv(merged_dir / "films.csv")
    utterances = pd.read_csv(merged_dir / "utterances.csv")
    faces = pd.read_csv(merged_dir / "faces.csv")
    scenes = pd.read_csv(merged_dir / "scenes.csv")

    accepted_films = films
    if "pass_audition" in films.columns:
        accepted_films = films[
            films["pass_audition"].astype(str).str.lower().isin(("true", "1", "yes"))
        ]
    accepted = set(accepted_films["film_id"].astype(str))
    if not set(utterances["film_id"].astype(str)) <= accepted:
        print("Peringatan: ada ujaran dari film yang tidak lolos audisi; dibuang.")
        utterances = utterances[utterances["film_id"].astype(str).isin(accepted)]

    settings = settings_from(cfg)
    pool = qc.evaluate(accepted_films, utterances, faces, settings)

    caps = caps_from(
        cfg,
        accepted_films,
        override=args.cap,
        rare_multiplier=args.rare_multiplier,
        no_cap=args.no_cap,
    )
    plan = qc.plan_scenes(pool, scenes, caps)
    pool = qc.apply_plan(pool, plan)

    out_dir.mkdir(parents=True, exist_ok=True)
    pool.to_csv(out_dir / POOL, index=False)
    plan.to_csv(out_dir / SCENES, index=False)

    stale = merged_config(merged_dir)
    stale = stale if stale and stale != cfg.sha256 else None
    rare_films = (
        []
        if args.no_cap
        else [
            str(film.film_id)
            for film in accepted_films.itertuples()
            if qc.is_rare_genre(getattr(film, "genre", ""))
        ]
    )
    report = render_report(
        cfg=cfg,
        pool=pool,
        plan=plan,
        caps=caps,
        settings=settings,
        stale_config=stale,
        rare_films=rare_films,
    )
    (out_dir / REPORT).write_text(report, encoding="utf-8")

    meta = {
        "generated_by": "scripts/06_qc.py",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_sha256": cfg.sha256,
        "merged_config_sha256": merged_config(merged_dir),
        "settings": {
            "face_ratio_min": settings.face_ratio_min,
            "snr_min_db": settings.snr_min_db,
            "clip_ratio_max": settings.clip_ratio_max,
            "min_words": settings.min_words,
        },
        "caps": caps,
        "rare_multiplier": args.rare_multiplier,
        "n_utterances": int(len(pool)),
        "n_qc_pass": int(pool["qc_pass"].sum()),
        "n_label_plan": int(pool["label_plan"].sum()),
        "n_scenes": int(len(plan)),
        "n_scenes_selected": int(plan["selected"].sum()),
    }
    (out_dir / META).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if stale:
        print(
            f"PERINGATAN: katalog dibuat dengan config {stale[:12]}…, "
            f"sekarang {cfg.sha256[:12]}…"
        )
    print_summary(pool, plan, caps)
    print(f"kolam    → {out_dir / POOL}")
    print(f"adegan   → {out_dir / SCENES}")
    print(f"laporan  → {out_dir / REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
