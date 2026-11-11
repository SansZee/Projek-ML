#!/usr/bin/env python3
"""07_annotate.py — penugasan, preview adegan, dan aplikasi pelabelan lokal.

Pelabelan berjalan **di laptop pemilik film**, karena media tidak pernah
dipindahkan. Tiga peran skrip ini:

  koordinator   --assign        membagi tugas ke semua anotator
  anggota       --prepare       memotong preview adegan yang ada di laptopnya
  anggota       --prepare --zip membungkus preview untuk dikirim ke orang lain
  anotator      streamlit run   aplikasi pelabelan

Alur lengkap:

  # 1. koordinator, setelah 06_qc.py
  uv run python scripts/07_annotate.py --assign --people kesan,budi,ani,dina

  # 2. tiap laptop memotong preview adegan yang bisa dibuat di sana
  uv run python scripts/07_annotate.py --prepare

  # 3. kirim preview adegan orang lain (gold + tukar silang)
  uv run python scripts/07_annotate.py --prepare --zip budi

  # 4. anotator membuka aplikasi
  uv run python scripts/07_annotate.py            # atau:
  uv run streamlit run scripts/07_annotate.py

Aplikasi menulis ``data/labels_<nama>.csv`` dan menambah baris setiap kali
satu label diklik. Sesi bisa ditutup kapan saja: yang sudah dilabeli tidak
ditanya lagi.

Tugas setiap orang:

  milik    seluruh adegan terpilih di film yang ada di laptopnya
  gold     dua adegan yang dinilai semua orang, untuk kappa global
  silang   150 ujaran milik orang lain, satu set per pemilik film

Butuh extra ``label``: ``uv sync --locked --extra label``.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from emosi import annotate
from emosi.config import Config, load_config
from emosi.media import require_tools, run_ffmpeg

ASSIGNMENTS = "assignments.csv"
POOL = "pool.csv"
POOL_SCENES = "pool_scenes.csv"


# -----------------------------------------------------------------------------
# Penugasan (koordinator)
# -----------------------------------------------------------------------------


def split_people(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def run_assign(cfg: Config, args) -> int:
    data_dir = cfg.resolve("paths.data_dir")
    pool_path = Path(args.pool).expanduser() if args.pool else data_dir / POOL
    if not pool_path.exists():
        print(f"{pool_path} tidak ada. Jalankan scripts/06_qc.py dulu.")
        return 1

    people = split_people(args.people)
    if len(people) < 2:
        print("Isi --people dengan minimal dua nama, mis. --people kesan,budi,ani,dina")
        return 1

    pool = pd.read_csv(pool_path)
    try:
        assignments = annotate.build_assignments(
            pool, people=people, gold=args.gold, cross=args.cross, seed=args.seed
        )
    except annotate.AssignmentError as error:
        print(f"Penugasan gagal: {error}")
        return 1

    out = data_dir / ASSIGNMENTS
    assignments.to_csv(out, index=False)

    table = annotate.workload(assignments).set_index("annotator")
    print("=" * 74)
    print(f"{'anotator':<14}{'milik':>8}{'gold':>8}{'silang':>8}{'total':>8}")
    print("-" * 74)
    for person, row in table.iterrows():
        print(
            f"{str(person):<14}{int(row[annotate.ROLE_OWN]):>8}"
            f"{int(row[annotate.ROLE_GOLD]):>8}{int(row[annotate.ROLE_CROSS]):>8}"
            f"{int(row['total']):>8}"
        )
    print("=" * 74)
    print(f"total {len(assignments)} tugas · {assignments['scene_id'].nunique()} adegan")

    gold = sorted(assignments.loc[assignments["role"] == annotate.ROLE_GOLD, "scene_id"].unique())
    print("gold set: " + (", ".join(gold) if gold else "-"))
    if table[annotate.ROLE_CROSS].min() == 0:
        print(
            "Catatan: ada anotator tanpa tugas silang karena semua film ada di satu "
            "laptop. Kappa tetap bisa dihitung dari gold set dan dari penilaian "
            "bersama atas ujaran pemilik film."
        )

    needs = annotate.scene_needs(assignments)
    per_person = {
        person: len(annotate.scenes_for(needs, person)) for person in table.index
    }
    print("adegan yang harus diputar: " + ", ".join(
        f"{person} {count}" for person, count in per_person.items()
    ))
    print(f"penugasan → {out}")
    print()
    print("Kirim data/pool.csv, data/pool_scenes.csv, dan data/assignments.csv ke")
    print("semua anotator. Ketiganya kecil (di bawah 1 MB).")
    return 0


# -----------------------------------------------------------------------------
# Preview (per laptop)
# -----------------------------------------------------------------------------


def local_films(data_dir: Path) -> dict[str, Path]:
    path = data_dir / "films.csv"
    if not path.exists():
        return {}
    films = pd.read_csv(path)
    if "path" not in films.columns:
        return {}
    return {
        str(row.film_id): Path(str(row.path))
        for row in films.itertuples()
        if str(getattr(row, "path", "")).strip()
    }


def scene_times(data_dir: Path) -> dict[str, tuple[float, float]]:
    path = data_dir / POOL_SCENES
    if not path.exists():
        return {}
    scenes = pd.read_csv(path)
    return {
        str(row.scene_id): (float(row.start_s), float(row.end_s))
        for row in scenes.itertuples()
    }


def preview_path(media_dir: Path, scene_id: str) -> Path:
    return media_dir / "preview" / f"{scene_id}.mp4"


def run_prepare(cfg: Config, args) -> int:
    require_tools("ffmpeg", "ffprobe")
    data_dir = cfg.resolve("paths.data_dir")
    preview_dir = cfg.resolve("paths.media_dir") / "preview"

    for name in (ASSIGNMENTS, POOL_SCENES):
        if not (data_dir / name).exists():
            print(f"{data_dir / name} tidak ada. Koordinator harus mengirim {name}.")
            return 1

    assignments = pd.read_csv(data_dir / ASSIGNMENTS)
    all_needs = annotate.scene_needs(assignments)
    needs = all_needs
    times = scene_times(data_dir)
    films = local_films(data_dir)
    if args.annotator:
        wanted = set(annotate.scenes_for(all_needs, args.annotator))
        needs = all_needs[all_needs["scene_id"].astype(str).isin(wanted)]

    jobs: list[tuple[str, str, Path, float, float]] = []
    unavailable: list[pd.Series] = []
    for row in needs.itertuples():
        film_id = str(row.film_id)
        scene_id = str(row.scene_id)
        if film_id not in films or not films[film_id].exists():
            unavailable.append(pd.Series(row._asdict()))
            continue
        if scene_id not in times:
            print(f"{scene_id}: tidak ada di {POOL_SCENES}, dilewati")
            continue
        start_s, end_s = times[scene_id]
        jobs.append((film_id, scene_id, films[film_id], start_s, end_s))

    if args.limit:
        jobs = jobs[: args.limit]

    if not jobs:
        print("Tidak ada adegan yang bisa dipotong di laptop ini.")
    else:
        print(
            f"memotong {len(jobs)} preview · {args.scale}p · crf {args.crf} → "
            f"{preview_dir}"
        )
        for _, scene_id, source, start_s, end_s in tqdm(
            jobs, desc="adegan", unit="adegan"
        ):
            dest = preview_path(cfg.resolve("paths.media_dir"), scene_id)
            if dest.exists() and not args.force:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            run_ffmpeg(
                annotate.preview_command(
                    source, dest, start_s, end_s, scale=args.scale, crf=args.crf
                )
            )

    if unavailable:
        print()
        print("Film berikut tidak ada di laptop ini, jadi preview-nya harus dikirim")
        print("pemiliknya (mereka menjalankan --prepare --zip <namamu>):")
        grouped = pd.DataFrame(unavailable).groupby(["film_id", "owner"]).agg(
            adegan=("scene_id", "nunique"),
            penilai=(
                "annotators",
                lambda values: ",".join(
                    sorted(set(",".join(values).split(",")))
                ),
            ),
        )
        for (film_id, owner), row in grouped.iterrows():
            print(
                f"  {film_id} milik {owner}: {int(row['adegan'])} adegan "
                f"(dibutuhkan {row['penilai']})"
            )

    if args.zip:
        # --zip selalu memakai seluruh penugasan, bukan yang sudah disaring
        # --annotator: yang dikirim adalah preview untuk orang lain.
        return run_zip(cfg, args, all_needs, preview_dir)

    sizes = [preview_path(cfg.resolve("paths.media_dir"), scene).stat().st_size
             for _, scene, _, _, _ in jobs
             if preview_path(cfg.resolve("paths.media_dir"), scene).exists()]
    if sizes:
        print(f"total preview di laptop ini: {sum(sizes) / 1048576:.0f} MB")
    print()
    print("Untuk mengirim preview ke anotator lain:")
    print("  uv run python scripts/07_annotate.py --prepare --zip <nama>")
    return 0


def run_zip(
    cfg: Config,
    args,
    needs: pd.DataFrame,
    preview_dir: Path,
) -> int:
    wanted = annotate.scenes_for(needs, args.zip)
    existing = [preview_dir / f"{scene}.mp4" for scene in wanted]
    files = [path for path in existing if path.exists()]
    missing = [path.stem for path in existing if not path.exists()]
    if not files:
        print(f"Tidak ada preview untuk {args.zip} di laptop ini.")
        return 1

    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    dest = cfg.root / "kiriman" / f"preview-untuk-{args.zip}-{stamp}.zip"
    annotate.write_preview_zip(dest, files, annotator=args.zip, scenes=wanted)
    size_mb = dest.stat().st_size / 1048576
    print(f"kiriman preview → {dest} ({size_mb:.0f} MB, {len(files)} adegan)")
    if missing:
        print("belum ada di laptop ini: " + ", ".join(sorted(missing)))
    print(
        f"Kirim zip itu ke {args.zip}; ia mengekstraknya sehingga isi "
        "preview/ masuk ke media/preview/."
    )
    return 0


# -----------------------------------------------------------------------------
# Aplikasi Streamlit
# -----------------------------------------------------------------------------


def streamlit_running() -> bool:
    try:
        from streamlit import runtime
    except ImportError:
        return False
    return bool(runtime.exists())


def mmss(seconds: float) -> str:
    total = int(max(0.0, seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def run_app() -> None:  # noqa: C901 - alur UI memang panjang
    import streamlit as st

    from emosi import annotate as ann

    st.set_page_config(page_title="Pelabelan emosi", layout="wide")
    cfg = load_config(os.environ.get("EMOSI_CONFIG", "config.yaml"))
    data_dir = cfg.resolve("paths.data_dir")
    preview_dir = cfg.resolve("paths.media_dir") / "preview"

    pool_path = data_dir / POOL
    assignment_path = data_dir / ASSIGNMENTS
    missing = [path.name for path in (pool_path, assignment_path) if not path.exists()]
    if missing:
        st.error(
            f"{', '.join(missing)} belum ada di `{data_dir}`. Minta koordinator "
            "mengirim `pool.csv`, `pool_scenes.csv`, dan `assignments.csv`."
        )
        st.stop()

    pool = pd.read_csv(pool_path)
    assignments = pd.read_csv(assignment_path)
    classes = cfg.classes()
    ambiguous = str(cfg.get("labels.ambiguous_token"))
    not_eligible = str(cfg.get("labels.not_eligible_token"))
    confidence_min = int(cfg.get("labels.confidence_min"))
    confidence_max = int(cfg.get("labels.confidence_max"))

    st.title("Pelabelan emosi")
    with st.expander("Aturan sesi (baca sekali)", expanded=False):
        st.markdown(
            "- **Kalibrasi dulu** bersama tim: sepakati contoh tiap kelas. Boleh "
            "berdiskusi di tahap ini.\n"
            "- Saat melabeli sendiri, **jangan berdiskusi**. Kappa hanya berarti "
            "kalau penilaian benar-benar independen.\n"
            "- Nilai emosi yang **terlihat dan terdengar**, bukan isi cerita. "
            "Kalau ragu, pakai `ambigu` dan turunkan keyakinan.\n"
            "- Pakai `tidak_layak` kalau wajah tertutup, suara tertimbun musik, "
            "atau teks subtitle tidak cocok dengan ucapannya.\n"
            "- Label bisa diubah selama sesi. Setelah adjudikasi, label dikunci."
        )

    people = sorted(assignments["annotator"].astype(str).unique())
    # Default ke pemilik film di laptop ini: orang yang punya tugas "milik".
    # Di laptop tanpa film, semua orang setara, jadi pakai urutan nama.
    owners = set(
        assignments.loc[assignments["role"] == ann.ROLE_OWN, "annotator"].astype(str)
    )
    wanted = os.environ.get("EMOSI_ANNOTATOR")
    default = (
        wanted
        if wanted in people
        else next((person for person in people if person in owners), people[0])
    )
    annotator = st.sidebar.selectbox(
        "Anotator", people, index=people.index(default), key="annotator_pick"
    )
    if st.session_state.get("_annotator") != annotator:
        st.session_state["_annotator"] = annotator
        st.session_state.pop("scene_pick", None)

    labels_path = data_dir / f"labels_{annotator}.csv"
    labels = ann.read_labels(labels_path)
    mine_labels = labels[labels["annotator"].astype(str) == annotator]
    done_ids = set(mine_labels["utterance_id"].astype(str))
    label_by_id = {
        str(row.utterance_id): row for row in mine_labels.itertuples()
    }

    mine = assignments[assignments["annotator"].astype(str) == annotator].merge(
        pool[
            [
                "utterance_id",
                "text",
                "start_s",
                "end_s",
                "duration_s",
                "index",
                "title",
            ]
        ],
        on="utterance_id",
        how="left",
        validate="one_to_one",
    )
    if mine.empty:
        st.warning("Tidak ada tugas untuk nama ini.")
        st.stop()
    if mine["text"].isna().any():
        st.warning("Ada tugas yang ujarannya tidak ada di pool.csv; tugas itu dilewati.")
        mine = mine[mine["text"].notna()]

    done_total, all_total = ann.progress(labels, assignments, annotator)
    st.sidebar.progress(min(1.0, done_total / max(1, all_total)))
    st.sidebar.caption(f"{done_total} / {all_total} ujaran selesai")
    roles = mine.groupby("role").size().to_dict()
    st.sidebar.caption(
        " · ".join(f"{role} {count}" for role, count in sorted(roles.items()))
    )

    scenes = (
        mine.groupby("scene_id", sort=False)
        .agg(
            film_id=("film_id", "first"),
            title=("title", "first"),
            start=("start_s", "min"),
            end=("end_s", "max"),
            n=("utterance_id", "size"),
        )
        .reset_index()
        .sort_values(["film_id", "start"], kind="stable")
        .reset_index(drop=True)
    )
    scene_done = {
        str(scene): len(
            set(group["utterance_id"].astype(str)) & done_ids
        )
        for scene, group in mine.groupby("scene_id")
    }
    scene_ids = [str(value) for value in scenes["scene_id"]]

    if st.session_state.get("scene_pick") not in scene_ids:
        unfinished = [
            scene for scene in scene_ids if scene_done.get(scene, 0) < len(
                mine[mine["scene_id"].astype(str) == scene]
            )
        ]
        st.session_state["scene_pick"] = (unfinished or scene_ids)[0]

    # Perpindahan adegan disimpan dulu, lalu diterapkan sebelum widget dibuat.
    # Streamlit menolak mengubah state widget yang sudah dibuat di run ini.
    next_scene = st.session_state.pop("_next_scene", None)
    if next_scene in scene_ids:
        st.session_state["scene_pick"] = next_scene

    def scene_label(scene_id: str) -> str:
        row = scenes[scenes["scene_id"].astype(str) == scene_id].iloc[0]
        total = int(row["n"])
        done = scene_done.get(scene_id, 0)
        mark = "selesai" if done >= total else f"{done}/{total}"
        return f"{scene_id} · {mmss(row['start'])} · {mark}"

    scene_id = st.sidebar.selectbox(
        "Adegan", scene_ids, key="scene_pick", format_func=scene_label
    )
    scene = scenes[scenes["scene_id"].astype(str) == str(scene_id)].iloc[0]
    scene_rows = (
        mine[mine["scene_id"].astype(str) == str(scene_id)]
        .sort_values("index", kind="stable")
        .reset_index(drop=True)
    )

    st.subheader(f"{scene['title']} · {scene_id}")
    scene_index = scene_ids.index(str(scene_id)) + 1
    st.caption(
        f"adegan {scene_index}/{len(scene_ids)} · "
        f"{mmss(scene['start'])}–{mmss(scene['end'])} · "
        f"{len(scene_rows)} ujaran"
    )

    preview = preview_dir / f"{scene_id}.mp4"
    if not preview.exists():
        st.error(
            f"Preview `{preview.name}` belum ada. Jalankan di laptop ini:\n\n"
            f"```\nuv run python scripts/07_annotate.py --prepare --annotator {annotator}\n```\n\n"
            "Kalau adegan ini milik film orang lain, minta zip preview dari pemiliknya."
        )
        st.stop()

    pick_key = f"utt_{scene_id}"
    advance = st.session_state.pop("_next", None)
    if advance and advance.get("scene") == str(scene_id):
        st.session_state[pick_key] = advance["utterance"]

    if pick_key not in st.session_state or st.session_state[pick_key] not in set(
        scene_rows["utterance_id"].astype(str)
    ):
        unlabeled = [
            str(value)
            for value in scene_rows["utterance_id"]
            if str(value) not in done_ids
        ]
        st.session_state[pick_key] = (unlabeled or list(scene_rows["utterance_id"]))[0]

    def utterance_label(utterance_id: str) -> str:
        row = scene_rows[scene_rows["utterance_id"].astype(str) == utterance_id].iloc[0]
        mark = "✓" if utterance_id in done_ids else "·"
        offset = mmss(float(row["start_s"]) - float(scene["start"]))
        text = str(row["text"])
        return f"{mark} {offset} · {text[:70]}"

    utterance_id = st.selectbox(
        "Ujaran", list(scene_rows["utterance_id"]), key=pick_key,
        format_func=utterance_label,
    )
    current = scene_rows[scene_rows["utterance_id"].astype(str) == str(utterance_id)].iloc[0]

    offset = max(0.0, float(current["start_s"]) - float(scene["start"]))
    duration = max(0.5, float(current["end_s"]) - float(current["start_s"]))
    whole_scene = st.toggle("Mulai dari awal adegan", value=False)
    if whole_scene:
        st.video(str(preview))
    else:
        st.video(
            str(preview),
            start_time=int(offset),
            end_time=int(offset + duration) + 1,
        )

    existing = label_by_id.get(str(utterance_id))
    if existing is not None:
        st.info(
            f"Sudah dilabeli: **{existing.label}** "
            f"(keyakinan {existing.confidence})"
        )
    st.markdown(f"### {current['text']}")
    st.caption(
        f"{mmss(current['start_s'])}–{mmss(current['end_s'])} · "
        f"{int(current['duration_s'])} detik · {current['role']}"
    )

    confidence = st.radio(
        "Keyakinan (1 ragu – 5 yakin)",
        list(range(confidence_min, confidence_max + 1)),
        horizontal=True,
        index=2 if confidence_min <= 3 <= confidence_max else 0,
        key="confidence",
    )
    notes = st.text_input("Catatan (opsional)", key=f"notes_{utterance_id}")

    def save(label: str) -> None:
        ann.upsert_labels(
            labels_path,
            [
                ann.label_row(
                    annotator=annotator,
                    utterance=current,
                    label=label,
                    confidence=int(confidence),
                    role=str(current["role"]),
                    notes=str(notes),
                )
            ],
        )
        remaining = [
            str(value)
            for value in scene_rows["utterance_id"]
            if str(value) != str(utterance_id) and str(value) not in done_ids
        ]
        if remaining:
            st.session_state["_next"] = {
                "scene": str(scene_id),
                "utterance": remaining[0],
            }
        st.rerun()

    columns = st.columns(len(classes))
    for column, name in zip(columns, classes, strict=True):
        if column.button(name, key=f"label_{name}_{utterance_id}", use_container_width=True):
            save(name)

    special = st.columns(3)
    if special[0].button(ambiguous, key=f"label_amb_{utterance_id}"):
        save(ambiguous)
    if special[1].button(not_eligible, key=f"label_no_{utterance_id}"):
        save(not_eligible)
    if special[2].button("lewati", key=f"skip_{utterance_id}"):
        remaining = [
            str(value)
            for value in scene_rows["utterance_id"]
            if str(value) != str(utterance_id) and str(value) not in done_ids
        ]
        if remaining:
            st.session_state["_next"] = {
                "scene": str(scene_id),
                "utterance": remaining[0],
            }
        st.rerun()

    if all(str(value) in done_ids for value in scene_rows["utterance_id"]):
        st.success("Adegan ini selesai.")
        next_scenes = [s for s in scene_ids if s != str(scene_id)]
        if next_scenes:
            current_index = scene_ids.index(str(scene_id))
            target = next_scenes[0]
            for candidate in scene_ids[current_index + 1 :]:
                if scene_done.get(candidate, 0) < len(
                    mine[mine["scene_id"].astype(str) == candidate]
                ):
                    target = candidate
                    break
            if st.button("Adegan berikutnya"):
                st.session_state["_next_scene"] = target
                st.rerun()

    st.sidebar.caption(f"label tersimpan: `{labels_path.name}`")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    if streamlit_running():
        run_app()
        return 0

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--assign", action="store_true", help="buat data/assignments.csv")
    parser.add_argument(
        "--people", default=None, help="daftar anotator, mis. kesan,budi,ani,dina"
    )
    parser.add_argument("--gold", type=int, default=2, help="jumlah adegan gold set")
    parser.add_argument("--cross", type=int, default=150, help="ujaran tukar silang per orang")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool", default=None, help="path pool.csv (default data/pool.csv)")
    parser.add_argument(
        "--prepare", action="store_true", help="potong preview adegan yang ada di laptop ini"
    )
    parser.add_argument(
        "--annotator", default=None, help="batasi --prepare ke adegan satu anotator"
    )
    parser.add_argument(
        "--zip", default=None, metavar="NAMA", help="bungkus preview yang dibutuhkan NAMA"
    )
    parser.add_argument("--scale", type=int, default=480, help="tinggi preview (piksel)")
    parser.add_argument(
        "--crf", type=int, default=30, help="kualitas x264, makin besar makin kecil"
    )
    parser.add_argument("--force", action="store_true", help="potong ulang preview yang ada")
    parser.add_argument("--limit", type=int, default=None, help="batasi jumlah adegan")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.assign:
        return run_assign(cfg, args)
    if args.prepare or args.zip:
        if args.zip and not args.prepare:
            args.prepare = True
        return run_prepare(cfg, args)

    parser.print_help()
    print()
    print("Untuk membuka aplikasi pelabelan:")
    print("  uv run streamlit run scripts/07_annotate.py")
    return 1


if __name__ == "__main__":
    sys.exit(main())
