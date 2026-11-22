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

``--prepare`` menghasilkan dua hal: ``media/preview/<scene>.mp4`` (480p) dan
``media/face_preview/<utterance>.jpg`` — tiga potongan wajah pada 25/50/75%
ujaran, diambil dari preview dengan aturan yang sama seperti ``10_cut.py``.
Anotator memakainya untuk memastikan wajah yang dinilai memang penutur.

Tugas setiap orang:

  milik    blok utama: adegan terpilih, bisa dari film sendiri atau orang lain
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

from emosi import annotate, face_preview
from emosi.config import Config, load_config
from emosi.media import probe_media, require_tools, run_ffmpeg

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
            pool,
            people=people,
            gold=args.gold,
            cross=args.cross,
            seed=args.seed,
            balance=args.balance,
            existing=annotate.labeled_scenes(data_dir) if args.balance else None,
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
    print("semua anotator. Ketiganya di bawah 4 MB.")
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


def build_face_strips(
    cfg: Config,
    scene_ids: set[str],
    times: dict[str, tuple[float, float]],
    *,
    force: bool = False,
) -> None:
    """Strip wajah 3 titik per ujaran, untuk ditampilkan saat melabeli.

    Diambil dari preview 480p, bukan film asli: anotator pemilik film lain
    hanya punya preview, dan yang perlu dikenali di sini wajahnya, bukan
    detailnya. Potongan final untuk model tetap dibuat ``10_cut.py`` dari film.
    """
    from emosi import cut, face, face_preview

    data_dir = cfg.resolve("paths.data_dir")
    media_dir = cfg.resolve("paths.media_dir")
    pool_path = data_dir / POOL
    if not pool_path.exists() or not scene_ids:
        return

    pool = pd.read_csv(pool_path)
    rows = pool[pool["scene_id"].astype(str).isin(scene_ids)]
    if rows.empty:
        return

    out_dir = media_dir / "face_preview"
    settings = face.settings_from_config(cfg)
    cut_settings = cut.CutSettings()
    analyzer, provider = face.build_analyzer(settings)

    print(
        f"\nstrip wajah · {len(rows)} ujaran · {settings.sample_fps:g} fps · "
        f"{provider} → {out_dir}"
    )
    made = skipped = missing = 0
    for scene_id, group in tqdm(
        rows.groupby("scene_id", sort=False), desc="adegan", unit="adegan"
    ):
        preview = preview_path(media_dir, str(scene_id))
        if not preview.exists():
            missing += len(group)
            continue
        info = probe_media(preview)
        scene_start = float(times.get(str(scene_id), (0.0, 0.0))[0])
        for row in group.itertuples():
            dest = face_preview.strip_path(out_dir, str(row.utterance_id))
            if dest.exists() and not force:
                skipped += 1
                continue
            start = max(0.0, float(row.start_s) - scene_start)
            end = max(start + 0.05, float(row.end_s) - scene_start)
            tiles = face_preview.strip_tiles(
                analyzer,
                preview,
                start,
                end,
                info.width,
                info.height,
                settings=settings,
                margin=cut_settings.face_margin,
                size=cut_settings.face_size,
            )
            face_preview.save_strip(dest, face_preview.compose_strip(tiles))
            made += 1
    print(f"  {made} dibuat · {skipped} sudah ada · {missing} tanpa preview")


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

    if not args.no_faces:
        build_face_strips(
            cfg,
            {str(value) for value in needs["scene_id"]},
            times,
            force=args.force,
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


CLASS_COLORS = {
    "senang": "#B45309",
    "sedih": "#1D4ED8",
    "marah": "#B91C1C",
    "takut": "#6D28D9",
    "netral": "#334155",
    "terkejut": "#0E7490",
    "jijik": "#15803D",
}

BADGE_COLORS = {
    "senang": "orange",
    "sedih": "blue",
    "marah": "red",
    "takut": "violet",
    "netral": "gray",
    "terkejut": "primary",
    "jijik": "green",
}


def build_app_css() -> str:
    class_rules = "\n".join(
        f'  [class*="st-key-label-{name}-"] button {{ background: {color}; color: #fff; }}\n'
        f'  [class*="st-key-label-{name}-"] button p {{ color: #fff; }}'
        for name, color in CLASS_COLORS.items()
    )
    return f"""
<style>
  [data-testid="stMainBlockContainer"], .block-container {{
    max-width: 1600px; padding-top: 1.4rem; padding-bottom: 3.5rem;
  }}
  [data-testid="stHeader"] {{ background: transparent; }}
  #MainMenu, footer {{ visibility: hidden; }}

  .emosi-legend {{ display: grid; gap: .3rem; margin: .3rem 0 .2rem; }}
  .emosi-legend div {{ font-size: .85rem; color: #334155; }}
  .emosi-legend i {{
    display: inline-block; width: .65rem; height: .65rem; border-radius: 999px;
    margin-right: .5rem; vertical-align: baseline;
  }}

  [class*="st-key-label-"] button {{
    min-height: 3.1rem; border-radius: .7rem; border: 1px solid transparent;
    padding: .3rem .4rem; gap: .3rem;
    transition: background-color .15s ease, box-shadow .15s ease;
  }}
  [class*="st-key-label-"] button p {{
    font-weight: 650; font-size: .85rem; white-space: nowrap;
  }}
  [class*="st-key-label-"] button:hover {{ filter: brightness(.9); }}
  [class*="st-key-label-"] button:focus-visible {{
    outline: 2px solid #0F172A; outline-offset: 2px;
  }}
{class_rules}

  [class*="st-key-special-"] button {{ min-height: 2.5rem; border-radius: .6rem; }}
  [class*="st-key-special-ambiguous-"] button {{
    border: 1px solid #0D9488; color: #0F766E; background: transparent;
  }}
  [class*="st-key-special-noteligible-"] button {{
    border: 1px solid #B91C1C; color: #991B1B; background: transparent;
  }}
  [class*="st-key-special-lewati"] button {{ color: #475569; }}
  [class*="st-key-step_"] button p,
  [class*="st-key-replay_utt_"] button p {{
    font-size: .85rem; white-space: nowrap;
  }}

  @media (prefers-reduced-motion: reduce) {{
    [class*="st-key-label-"] button {{ transition: none; }}
  }}
</style>
"""


ALL_FILMS = "__all__"


def _read_static_tables(
    pool_path: Path, assignment_path: Path, scene_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[float, float]]]:
    import streamlit as st

    @st.cache_data(show_spinner=False)
    def _cached(
        pool_str: str,
        assign_str: str,
        scene_str: str,
        _mtimes: tuple[float, float, float],
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[float, float]]]:
        pool = pd.read_csv(pool_str)
        assignments = pd.read_csv(assign_str)
        scenes = pd.read_csv(scene_str)
        bounds = {
            str(row.scene_id): (float(row.start_s), float(row.end_s))
            for row in scenes.itertuples()
        }
        return pool, assignments, bounds

    mtimes = (
        pool_path.stat().st_mtime,
        assignment_path.stat().st_mtime,
        scene_path.stat().st_mtime,
    )
    return _cached(str(pool_path), str(assignment_path), str(scene_path), mtimes)


def run_app() -> None:  # noqa: C901 - alur UI memang panjang
    import streamlit as st

    from emosi import annotate as ann

    st.set_page_config(page_title="Pelabelan emosi", layout="wide")
    st.markdown(build_app_css(), unsafe_allow_html=True)
    if toast_msg := st.session_state.pop("_scene_toast", None):
        st.toast(toast_msg)

    cfg = load_config(os.environ.get("EMOSI_CONFIG", "config.yaml"))
    data_dir = cfg.resolve("paths.data_dir")
    preview_dir = cfg.resolve("paths.media_dir") / "preview"
    face_dir = cfg.resolve("paths.media_dir") / "face_preview"

    pool_path = data_dir / POOL
    assignment_path = data_dir / ASSIGNMENTS
    scene_path = data_dir / POOL_SCENES
    missing = [
        path.name for path in (pool_path, assignment_path, scene_path) if not path.exists()
    ]
    if missing:
        st.error(
            f"{', '.join(missing)} belum ada di `{data_dir}`. Minta koordinator "
            "mengirim `pool.csv`, `pool_scenes.csv`, dan `assignments.csv`."
        )
        st.stop()

    pool, assignments, scene_bounds = _read_static_tables(
        pool_path, assignment_path, scene_path
    )
    classes = cfg.classes()
    ambiguous = str(cfg.get("labels.ambiguous_token"))
    not_eligible = str(cfg.get("labels.not_eligible_token"))
    confidence_min = int(cfg.get("labels.confidence_min"))
    confidence_max = int(cfg.get("labels.confidence_max"))

    st.sidebar.markdown("### Pelabelan emosi")
    st.sidebar.caption(f"`{cfg.get('project.name')}`")
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
    prev_annotator = st.session_state.get("_annotator")
    if prev_annotator != annotator:
        st.session_state["_annotator"] = annotator
        if prev_annotator is not None:
            st.session_state.pop("film_pick", None)
            st.session_state.pop("_active_film", None)
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
                "multi_face_share",
                "dominant_share",
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
    percent = int(round(100 * done_total / max(1, all_total)))
    st.sidebar.progress(
        min(1.0, done_total / max(1, all_total)),
        text=f"{done_total} / {all_total} ujaran · {percent}%",
    )
    roles = mine.groupby("role").size().to_dict()
    st.sidebar.caption(
        " · ".join(f"{role} {count}" for role, count in sorted(roles.items()))
    )
    st.sidebar.divider()
    st.sidebar.caption("Warna tombol")
    legend = "".join(
        f'<div><i style="background:{CLASS_COLORS.get(name, "#334155")}"></i>{name}</div>'
        for name in classes
    )
    st.sidebar.markdown(f'<div class="emosi-legend">{legend}</div>', unsafe_allow_html=True)
    video_penuh = st.sidebar.toggle(
        "Video penuh",
        key="video_penuh",
        help="Video selebar layar, teks pindah ke bawah. Untuk menilai wajah dari dekat.",
    )
    auto_next_scene = st.sidebar.toggle(
        "Lanjut adegan otomatis",
        value=True,
        key="auto_next_scene",
        help="Saat satu adegan selesai, otomatis lanjut ke adegan berikutnya di film yang sama.",
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
        str(scene): len(set(group["utterance_id"].astype(str)) & done_ids)
        for scene, group in mine.groupby("scene_id")
    }
    scene_totals = dict(zip(scenes["scene_id"].astype(str), scenes["n"].astype(int), strict=True))
    unfinished_set = {
        sid for sid, total in scene_totals.items() if scene_done.get(sid, 0) < total
    }
    scene_film_map = dict(
        zip(scenes["scene_id"].astype(str), scenes["film_id"].astype(str), strict=True)
    )

    films_df = (
        scenes.groupby("film_id", sort=False)
        .agg(
            title=("title", "first"),
            n_scenes=("scene_id", "size"),
            n_utts=("n", "sum"),
        )
        .reset_index()
        .sort_values("title", kind="stable")
        .reset_index(drop=True)
    )
    film_ids = [str(fid) for fid in films_df["film_id"]]
    film_options = [ALL_FILMS, *film_ids]
    film_titles = dict(zip(film_ids, films_df["title"].astype(str), strict=True))
    film_scene_counts = dict(zip(film_ids, films_df["n_scenes"].astype(int), strict=True))
    film_utt_counts = dict(zip(film_ids, films_df["n_utts"].astype(int), strict=True))
    film_done_counts = {
        str(fid): len(set(group["utterance_id"].astype(str)) & done_ids)
        for fid, group in mine.groupby("film_id")
    }

    # Label dropdown film/adegan/ujaran dijaga statis (tanpa angka progres berubah-ubah)
    # supaya Streamlit SelectboxSerde tidak gagal mencocokkan label lalu mengulang ke item lain.
    def film_label(fid: str) -> str:
        if fid == ALL_FILMS:
            return f"Semua film ({len(film_ids)} film · {len(scenes)} adegan)"
        return f"{film_titles[fid]} · {film_scene_counts[fid]} adegan"

    pending_film = st.session_state.pop("_next_film", None)
    if pending_film in film_options:
        st.session_state["film_pick"] = pending_film
    if st.session_state.get("film_pick") not in film_options:
        qp_film = st.query_params.get("film")
        qp_scene = st.query_params.get("scene")
        if qp_film in film_options:
            st.session_state["film_pick"] = qp_film
        elif qp_scene in scene_film_map:
            st.session_state["film_pick"] = scene_film_map[qp_scene]
        else:
            first_open_film = next(
                (str(sid) for sid in scenes["scene_id"].astype(str) if sid in unfinished_set),
                None,
            )
            st.session_state["film_pick"] = (
                scene_film_map[first_open_film] if first_open_film else film_ids[0]
            )

    film_pick = st.sidebar.selectbox(
        "Judul film", film_options, key="film_pick", format_func=film_label
    )
    prev_film = st.session_state.get("_active_film")
    if prev_film is not None and prev_film != film_pick:
        st.session_state.pop("scene_pick", None)
        st.session_state.pop("_next_scene", None)
    st.session_state["_active_film"] = film_pick

    if film_pick == ALL_FILMS:
        visible_scenes = scenes.reset_index(drop=True)
    else:
        visible_scenes = scenes[scenes["film_id"].astype(str) == film_pick].reset_index(
            drop=True
        )
        f_done = film_done_counts.get(film_pick, 0)
        f_tot = film_utt_counts.get(film_pick, 0)
        st.sidebar.caption(f"Film ini: {f_done} / {f_tot} ujaran")

    visible_ids = [str(value) for value in visible_scenes["scene_id"]]
    visible_meta = {
        str(row.scene_id): (pos, row)
        for pos, row in enumerate(visible_scenes.itertuples(), start=1)
    }

    next_scene = st.session_state.pop("_next_scene", None)
    if next_scene in visible_ids:
        st.session_state["scene_pick"] = next_scene
    if st.session_state.get("scene_pick") not in visible_ids:
        qp_scene = st.query_params.get("scene")
        if qp_scene in visible_ids:
            st.session_state["scene_pick"] = qp_scene
        else:
            unfinished_visible = [sid for sid in visible_ids if sid in unfinished_set]
            st.session_state["scene_pick"] = (unfinished_visible or visible_ids)[0]

    def scene_label(sid: str) -> str:
        pos, row = visible_meta[sid]
        if film_pick == ALL_FILMS:
            return f"{row.title} · {sid} · {mmss(row.start)} ({int(row.n)} ujaran)"
        return f"Adegan {pos}/{len(visible_ids)} · {mmss(row.start)} ({int(row.n)} ujaran) · {sid}"

    scene_id = st.sidebar.selectbox(
        "Adegan", visible_ids, key="scene_pick", format_func=scene_label
    )
    if (
        st.query_params.get("film") != film_pick
        or st.query_params.get("scene") != str(scene_id)
    ):
        st.query_params["film"] = film_pick
        st.query_params["scene"] = str(scene_id)

    scene = visible_scenes[visible_scenes["scene_id"].astype(str) == str(scene_id)].iloc[0]
    scene_rows = (
        mine[mine["scene_id"].astype(str) == str(scene_id)]
        .sort_values("index", kind="stable")
        .reset_index(drop=True)
    )
    scene_total = len(scene_rows)
    done_in_scene = scene_done.get(str(scene_id), 0)
    scene_finished = done_in_scene >= scene_total
    cut_start, cut_end = scene_bounds.get(
        str(scene_id), (float(scene["start"]), float(scene["end"]))
    )
    st.sidebar.progress(
        done_in_scene / max(1, scene_total),
        text=f"{done_in_scene} / {scene_total} di adegan ini",
    )

    scene_index = visible_ids.index(str(scene_id)) + 1
    st.markdown(f"#### {scene['title']}")
    st.caption(
        f"`{scene_id}` · adegan {scene_index}/{len(visible_ids)} · "
        f"{mmss(cut_start)}–{mmss(cut_end)} · {scene_total} ujaran"
    )
    if scene_finished:
        st.badge("adegan ini selesai", color="green")

    preview = preview_dir / f"{scene_id}.mp4"
    if not preview.exists():
        st.error(
            f"Preview `{preview.name}` belum ada. Jalankan di laptop ini:\n\n"
            f"```\nuv run python scripts/07_annotate.py --prepare --annotator {annotator}\n```\n\n"
            "Kalau adegan ini milik film orang lain, minta zip preview dari pemiliknya."
        )
        st.stop()

    scene_utt_ids = [str(value) for value in scene_rows["utterance_id"]]
    pick_key = f"utt_{scene_id}"
    advance = st.session_state.pop("_next", None)
    if advance and advance.get("scene") == str(scene_id):
        if advance["utterance"] in scene_utt_ids:
            st.session_state[pick_key] = advance["utterance"]

    if pick_key not in st.session_state or st.session_state[pick_key] not in set(scene_utt_ids):
        unlabeled = [uid for uid in scene_utt_ids if uid not in done_ids]
        st.session_state[pick_key] = (unlabeled or scene_utt_ids)[0]

    utt_meta = {
        str(row.utterance_id): (pos, row)
        for pos, row in enumerate(scene_rows.itertuples(), start=1)
    }

    def utterance_label(uid: str) -> str:
        pos, row = utt_meta[uid]
        offset_str = mmss(float(row.start_s) - cut_start)
        return f"{pos}/{scene_total} · {offset_str} · {str(row.text)[:65]}"

    utterance_id = st.selectbox(
        "Ujaran", scene_utt_ids, key=pick_key, format_func=utterance_label
    )
    current = scene_rows[scene_rows["utterance_id"].astype(str) == str(utterance_id)].iloc[0]

    offset = max(0.0, float(current["start_s"]) - cut_start)
    duration = max(0.5, float(current["end_s"]) - float(current["start_s"]))
    existing = label_by_id.get(str(utterance_id))
    replay_seq_key = f"_replay_seq_{scene_id}_{utterance_id}"

    def trigger_replay() -> None:
        st.session_state["_user_interacted"] = True
        st.session_state["whole_scene"] = False
        st.session_state[replay_seq_key] = (
            int(st.session_state.get(replay_seq_key, 0)) + 1
        )

    def move_step(direction: int) -> None:
        st.session_state["_user_interacted"] = True
        nxt = ann.step_utterance(
            scene_utt_ids, str(utterance_id), done_ids, direction=direction, prefer_unlabeled=False
        )
        if nxt is not None:
            st.session_state["_next"] = {"scene": str(scene_id), "utterance": nxt}
            return
        cur_pos = visible_ids.index(str(scene_id))
        if direction > 0 and cur_pos + 1 < len(visible_ids):
            target_scene = visible_ids[cur_pos + 1]
            st.session_state["_next_scene"] = target_scene
        elif direction < 0 and cur_pos > 0:
            prev_scene = visible_ids[cur_pos - 1]
            prev_utts = (
                mine[mine["scene_id"].astype(str) == prev_scene]
                .sort_values("index", kind="stable")["utterance_id"]
                .astype(str)
                .tolist()
            )
            st.session_state["_next_scene"] = prev_scene
            if prev_utts:
                st.session_state[f"utt_{prev_scene}"] = prev_utts[-1]

    def video_block() -> None:
        replay_seq = int(st.session_state.get(replay_seq_key, 0))
        autoplay = bool(
            not whole_scene
            and ((auto_play and st.session_state.get("_user_interacted", False)) or replay_seq > 0)
        )
        alt_tag = f"Preview {scene_id} · {utterance_id} · {replay_seq}"
        with st.container(key=f"video_slot_{scene_id}_{utterance_id}_{replay_seq}"):
            if whole_scene:
                st.video(str(preview), width="stretch", alt=alt_tag)
            else:
                start_time, end_time = ann.video_window(
                    offset, duration, cut_end - cut_start
                )
                st.video(
                    str(preview),
                    start_time=start_time,
                    end_time=end_time,
                    width="stretch",
                    autoplay=autoplay,
                    alt=alt_tag,
                )
        strip = face_preview.strip_path(face_dir, str(utterance_id))
        if strip.exists():
            st.image(str(strip), caption="Wajah yang dipakai model · 25/50/75%")
        else:
            st.caption(
                "Strip wajah belum ada — jalankan `07_annotate.py --prepare` "
                "di laptop ini."
            )
        stats = (
            current.get("multi_face_share"),
            current.get("dominant_share"),
        )
        if pd.notna(stats[0]) and pd.notna(stats[1]):
            multi, dominant = float(stats[0]), float(stats[1])
            detail = f"2+ wajah di {multi:.0%} frame · wajah dominan {dominant:.0%}"
            if multi > 0.35 or dominant < 0.5:
                st.warning(f"{detail} — pastikan yang dipotong memang penutur.")
            else:
                st.caption(detail)

    def text_block() -> None:
        if existing is not None:
            st.badge(
                f"Sudah dilabeli: {existing.label}",
                color=BADGE_COLORS.get(str(existing.label), "gray"),
            )
            st.caption(f"keyakinan {existing.confidence}")
        else:
            st.caption("Belum dilabeli")
        st.markdown(f"### {current['text']}")
        st.caption(
            f"{mmss(current['start_s'])}–{mmss(current['end_s'])} · "
            f"{int(current['duration_s'])} detik · {current['role']}"
        )

    def control_block() -> tuple[int, str]:
        chosen = st.radio(
            "Keyakinan (1 ragu – 5 yakin)",
            list(range(confidence_min, confidence_max + 1)),
            horizontal=True,
            index=2 if confidence_min <= 3 <= confidence_max else 0,
            key="confidence",
            help="1 = menebak, 3 = cukup yakin, 5 = sangat yakin.",
        )
        note = st.text_input("Catatan (opsional)", key=f"notes_{utterance_id}")
        return int(chosen), str(note)

    c_prev, c_rep, c_next, c_auto, c_whole = st.columns([1.08, 1.25, 1.08, 0.92, 0.97], gap="small")
    with c_prev:
        st.button(
            "← Sebelumnya",
            key=f"step_prev_{scene_id}_{utterance_id}",
            shortcut="Left",
            width="stretch",
            disabled=(scene_index == 1 and str(utterance_id) == scene_utt_ids[0]),
            on_click=move_step,
            args=(-1,),
        )
    with c_rep:
        st.button(
            "↺ Ulang potongan",
            key=f"replay_utt_{scene_id}_{utterance_id}",
            shortcut="U",
            width="stretch",
            help="Putar ulang video tepat di potongan ujaran ini (tombol U).",
            on_click=trigger_replay,
        )
    with c_next:
        st.button(
            "Berikutnya →",
            key=f"step_next_{scene_id}_{utterance_id}",
            shortcut="Right",
            width="stretch",
            disabled=(scene_index == len(visible_ids) and str(utterance_id) == scene_utt_ids[-1]),
            on_click=move_step,
            args=(1,),
        )
    with c_auto:
        auto_play = st.toggle(
            "Putar otomatis",
            value=True,
            key="auto_play",
            help="Langsung putar video setiap pindah atau selesai menilai ujaran.",
        )
    with c_whole:
        whole_scene = st.toggle("Dari awal adegan", key="whole_scene")

    if video_penuh:
        video_block()
        text_col, control_col = st.columns([1.6, 1], gap="large")
        with text_col:
            text_block()
        with control_col:
            confidence, notes = control_block()
    else:
        video_col, side_col = st.columns([2.4, 1], gap="large")
        with video_col:
            video_block()
        with side_col:
            text_block()
            confidence, notes = control_block()

    def save(label: str) -> None:
        st.session_state["_user_interacted"] = True
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
        done_after = done_ids | {str(utterance_id)}
        nxt = ann.step_utterance(
            scene_utt_ids,
            str(utterance_id),
            done_after,
            direction=1,
            prefer_unlabeled=True,
        )
        if nxt is not None:
            st.session_state["_next"] = {"scene": str(scene_id), "utterance": nxt}
        else:
            unfinished_after = set(unfinished_set)
            if set(scene_utt_ids) <= done_after:
                unfinished_after.discard(str(scene_id))
            nxt_scene = ann.next_scene_in_scope(
                visible_ids, str(scene_id), unfinished_after
            )
            if auto_next_scene and nxt_scene is not None:
                st.session_state["_next_scene"] = nxt_scene
                st.session_state["_scene_toast"] = (
                    f"Adegan {scene_id} selesai · lanjut ke {nxt_scene}"
                )
        st.rerun()

    ambiguous_shortcut = str(len(classes) + 1)
    st.caption(
        f"Keyboard: `←` sebelum · `U` ulang potongan · `→` berikut · "
        f"`1`–`{len(classes)}` emosi · `{ambiguous_shortcut}` ambigu · `0` tidak layak. "
        "Jika kursor masih di dropdown/catatan, klik area kosong sekali agar tombol aktif."
    )
    label_cols = st.columns(len(classes), gap="small")
    for index, (column, name) in enumerate(
        zip(label_cols, classes, strict=True), start=1
    ):
        if column.button(
            f"{index} · {name}",
            key=f"label-{name}-{utterance_id}",
            shortcut=str(index),
            width="stretch",
        ):
            save(name)

    special = st.columns(3, gap="small")
    if special[0].button(
        f"{ambiguous_shortcut} · {ambiguous}",
        key=f"special-ambiguous-{utterance_id}",
        shortcut=ambiguous_shortcut,
        width="stretch",
    ):
        save(ambiguous)
    if special[1].button(
        f"0 · {not_eligible.replace('_', ' ')}",
        key=f"special-noteligible-{utterance_id}",
        shortcut="0",
        width="stretch",
    ):
        save(not_eligible)
    if special[2].button(
        "Lewati ujaran ini",
        key=f"special-lewati-{utterance_id}",
        width="stretch",
    ):
        move_step(1)
        st.rerun()

    if scene_finished:
        nxt_in_film = ann.next_scene_in_scope(
            visible_ids, str(scene_id), unfinished_set
        )
        if nxt_in_film is not None:
            st.success("Adegan ini selesai.")
            if st.button("Adegan berikutnya", type="primary"):
                st.session_state["_next_scene"] = nxt_in_film
                st.rerun()
        elif film_pick != ALL_FILMS:
            st.success(f"Semua adegan film {scene['title']} sudah selesai.")
            other_unfinished_films = [
                fid
                for fid in film_ids
                if fid != film_pick and film_done_counts.get(fid, 0) < film_utt_counts.get(fid, 0)
            ]
            if other_unfinished_films and st.button(
                f"Lanjut ke film: {film_titles[other_unfinished_films[0]]}",
                type="primary",
            ):
                st.session_state["_next_film"] = other_unfinished_films[0]
                st.session_state.pop("scene_pick", None)
                st.rerun()
        else:
            st.success("Adegan ini selesai.")

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
    parser.add_argument(
        "--balance",
        action="store_true",
        help="bagi blok utama rata per adegan, bukan semua film ke pemiliknya",
    )
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
    parser.add_argument(
        "--no-faces",
        action="store_true",
        help="lewati strip wajah 3 titik (lebih cepat, tetapi anotator tak melihat wajah model)",
    )
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
