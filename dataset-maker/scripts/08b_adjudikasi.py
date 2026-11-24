#!/usr/bin/env python3
"""08b_adjudikasi.py — layar untuk memutuskan butir tanpa mayoritas.

Dijalankan setelah ``08_agree.py`` menulis ``data/adjudication.csv``:

  uv run streamlit run scripts/08b_adjudikasi.py

Hanya butir yang ditetapkan untuk satu orang (kolom ``adjudicator``) yang
ditampilkan. Tombol ``1``–``7`` memilih kelas, ``8`` ambigu, ``0`` tidak layak —
sama dengan aplikasi pelabelan. Setiap klik langsung menulis ke
``adjudication.csv``, jadi sesi bisa ditutup kapan saja.

Setelah semua butir terisi, jalankan ``08_agree.py`` lagi untuk menerapkannya.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from emosi import agree, face_preview
from emosi import annotate as ann
from emosi.config import load_config

ADJUDICATION = "adjudication.csv"
POOL = "pool.csv"
POOL_SCENES = "pool_scenes.csv"


def mmss(seconds: float) -> str:
    total = max(0, int(float(seconds)))
    return f"{total // 60}:{total % 60:02d}"


def load_items(data_dir: Path) -> pd.DataFrame:
    """Gabungkan adjudication.csv dengan kolom pool yang dibutuhkan layar."""
    frame = pd.read_csv(data_dir / ADJUDICATION)
    if "decision" not in frame.columns:
        frame["decision"] = ""
    frame["decision"] = frame["decision"].fillna("").astype(str).str.strip()
    for column in ("adjudicator", "notes", "suara"):
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)

    pool = pd.read_csv(data_dir / POOL)
    scenes = pd.read_csv(data_dir / POOL_SCENES).set_index("scene_id")
    merged = frame.merge(
        pool[
            [
                "utterance_id",
                "text",
                "start_s",
                "end_s",
                "duration_s",
                "title",
                "owner",
            ]
        ],
        on="utterance_id",
        how="left",
    )
    merged["scene_start"] = merged["scene_id"].astype(str).map(scenes["start_s"])
    merged["scene_end"] = merged["scene_id"].astype(str).map(scenes["end_s"])
    return merged


def main() -> None:
    import streamlit as st

    cfg = load_config(os.environ.get("EMOSI_CONFIG", "config.yaml"))
    data_dir = cfg.resolve("paths.data_dir")
    media = cfg.resolve("paths.media_dir")
    st.set_page_config(page_title="Adjudikasi emosi", layout="wide")

    required = {name: data_dir / name for name in (ADJUDICATION, POOL, POOL_SCENES)}
    missing = [path.name for path in required.values() if not path.exists()]
    if missing:
        st.error(
            f"`{', '.join(missing)}` belum ada. Jalankan `08_agree.py` dulu supaya "
            "`adjudication.csv` lahir."
        )
        st.stop()

    classes = [str(name) for name in cfg.classes()]
    ambiguous = str(cfg.get("labels.ambiguous_token"))
    not_eligible = str(cfg.get("labels.not_eligible_token"))
    items = load_items(data_dir)
    people = sorted({name for name in items["adjudicator"] if name})

    st.sidebar.markdown("### Adjudikasi emosi")
    adjudicator = st.sidebar.selectbox(
        "Adjudikator",
        people or [os.environ.get("EMOSI_ADJUDICATOR", "")],
        index=0,
        help="Hanya butir yang ditetapkan untuk nama ini yang muncul.",
    )
    mine = items[items["adjudicator"] == adjudicator] if adjudicator else items
    if mine.empty:
        st.warning(
            "Tidak ada butir untuk nama ini. Pilih nama lain, atau jalankan "
            "`08_agree.py` lagi kalau pembagiannya baru berubah."
        )
        st.stop()

    done = mine["decision"].ne("")
    st.sidebar.progress(
        float(done.mean()),
        text=f"{int(done.sum())} / {len(mine)} butir · {done.mean():.0%}",
    )
    st.sidebar.caption(
        "·".join(
            f"{name} {count}"
            for name, count in items["adjudicator"].value_counts().items()
        )
    )
    only_open = st.sidebar.toggle(
        "Hanya yang belum diputuskan",
        value=True,
        help="Matikan untuk meninjau atau mengubah keputusan yang sudah ada.",
    )
    visible = (mine[~done] if only_open else mine).reset_index(drop=True)
    if visible.empty:
        st.success(f"Semua {len(mine)} butir untuk {adjudicator} sudah diputuskan.")
        st.stop()

    position = int(st.session_state.get("adj_pos", 0))
    position = max(0, min(position, len(visible) - 1))
    row = visible.iloc[position]
    scene_id = str(row["scene_id"])
    utterance_id = str(row["utterance_id"])
    scene_start = float(row["scene_start"]) if pd.notna(row["scene_start"]) else 0.0
    scene_end = float(row["scene_end"]) if pd.notna(row["scene_end"]) else 0.0
    offset = max(0.0, float(row["start_s"]) - scene_start)
    duration = max(0.5, float(row["end_s"]) - float(row["start_s"]))

    st.markdown(f"#### {row.get('title', '')} · `{scene_id}`")
    st.caption(
        f"`{utterance_id}` · dinilai {int(row['n_raters'])} orang · "
        f"suara: {row['suara']} · untuk {adjudicator}"
    )

    notes_key = f"adj-notes-{utterance_id}"
    st.sidebar.text_input("Catatan butir ini", key=notes_key)

    video_col, side_col = st.columns([2.4, 1], gap="large")
    with video_col:
        preview = media / "preview" / f"{scene_id}.mp4"
        if preview.exists():
            start_time, end_time = ann.video_window(
                offset, duration, max(0.0, scene_end - scene_start)
            )
            st.video(
                str(preview),
                start_time=start_time,
                end_time=end_time,
                width="stretch",
            )
        else:
            st.error(
                f"Preview `{preview.name}` belum ada di laptop ini (film milik "
                f"{row.get('owner', '?')}). Minta zip-nya, atau putuskan tanpa video."
            )
        strip = face_preview.strip_path(media / "face_preview", utterance_id)
        if strip.exists():
            st.image(str(strip), caption="Wajah yang dipakai model · 25/50/75%")

    with side_col:
        st.markdown(f"### {row['text']}")
        st.caption(
            f"{mmss(float(row['start_s']))}–{mmss(float(row['end_s']))} · "
            f"{int(row['duration_s'])} detik"
        )
        if row["decision"]:
            st.badge(f"Sudah diputuskan: {row['decision']}", color="green")
        else:
            st.caption("Belum diputuskan")

    def decide(label: str) -> None:
        agree.save_decision(
            data_dir / ADJUDICATION,
            utterance_id,
            label,
            decided_by=adjudicator,
            notes=str(st.session_state.get(notes_key, "")),
        )
        st.rerun()

    def move(step: int) -> None:
        st.session_state["adj_pos"] = max(0, min(position + step, len(visible) - 1))
        st.rerun()

    nav = st.columns([1, 1, 4], gap="small")
    nav[0].button(
        "← Sebelumnya",
        key=f"adj-prev-{utterance_id}",
        shortcut="Left",
        width="stretch",
        disabled=position == 0,
        on_click=move,
        args=(-1,),
    )
    nav[1].button(
        "Berikutnya →",
        key=f"adj-next-{utterance_id}",
        shortcut="Right",
        width="stretch",
        disabled=position == len(visible) - 1,
        on_click=move,
        args=(1,),
    )
    nav[2].caption(
        f"Butir {position + 1}/{len(visible)} · klik tombol atau tekan angka. "
        "Keputusan tersimpan saat tombol ditekan."
    )

    for index, (column, name) in enumerate(
        zip(st.columns(len(classes), gap="small"), classes, strict=True), start=1
    ):
        if column.button(
            f"{index} · {name}",
            key=f"adj-{name}-{utterance_id}",
            shortcut=str(index),
            width="stretch",
        ):
            decide(name)

    special = st.columns(2, gap="small")
    if special[0].button(
        f"{len(classes) + 1} · {ambiguous}",
        key=f"adj-ambiguous-{utterance_id}",
        shortcut=str(len(classes) + 1),
        width="stretch",
    ):
        decide(ambiguous)
    if special[1].button(
        f"0 · {not_eligible.replace('_', ' ')}",
        key=f"adj-noteligible-{utterance_id}",
        shortcut="0",
        width="stretch",
    ):
        decide(not_eligible)


if __name__ == "__main__":
    main()
