from __future__ import annotations

import pandas as pd
import pytest

from emosi import qc

SETTINGS = qc.QcSettings(
    face_ratio_min=0.50,
    snr_min_db=15.0,
    clip_ratio_max=0.01,
    min_words=2,
)


def tables(
    *,
    genre: str = "romansa",
    snr: float = 22.0,
    clip: float = 0.0,
    texts: tuple[str, ...] = ("hello there", "how are you"),
    words: tuple[int, ...] = (2, 3),
    ratios: tuple[float, ...] = (0.7, 0.8),
    with_faces: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    films = pd.DataFrame(
        [
            {
                "film_id": "AAA01",
                "owner": "kesan",
                "genre": genre,
                "snr_db": snr,
                "clip_ratio": clip,
                "pass_audition": True,
            }
        ]
    )
    utterance_rows = []
    face_rows = []
    for index, (text, n_words, ratio) in enumerate(
        zip(texts, words, ratios, strict=True), start=1
    ):
        utterance_rows.append(
            {
                "film_id": "AAA01",
                "utterance_id": f"AAA01-U{index:05d}",
                "scene_id": "AAA01-S001",
                "index": index,
                "start_s": float(index),
                "end_s": float(index) + 2,
                "duration_s": 2.0,
                "text": text,
                "n_words": n_words,
                "speaker": "",
                "timing": "asr",
                "n_cues": 1,
            }
        )
        face_rows.append(
            {
                "film_id": "AAA01",
                "utterance_id": f"AAA01-U{index:05d}",
                "face_ratio": ratio,
                "emb_frames": 3,
                "actor_key": "AAA01-A01",
            }
        )
    faces = pd.DataFrame(face_rows)
    if not with_faces:
        faces = faces.iloc[0:0]
    return films, pd.DataFrame(utterance_rows), faces


def test_good_utterances_pass_and_keep_columns():
    pool = qc.evaluate(*tables(), SETTINGS)
    assert list(pool["qc_pass"]) == [True, True]
    assert list(pool["qc_reason"]) == ["", ""]
    assert list(pool["label_plan"]) == [False, False]
    assert set(qc.POOL_COLUMNS) == set(pool.columns)
    assert pool.loc[0, "owner"] == "kesan"


def test_each_reason_is_recorded():
    pool = qc.evaluate(
        *tables(texts=("", "hi"), words=(0, 1), ratios=(0.2, 0.1), snr=10.0, clip=0.5),
        SETTINGS,
    )
    first = set(pool.loc[0, "qc_reason"].split(";"))
    assert {"teks_kosong", "kata_sedikit:0", "wajah_jarang:0.20", "snr_rendah:10.0"} <= first
    assert "audio_klip:0.500" in first
    assert not pool["qc_pass"].any()


def test_lyric_lines_are_rejected():
    pool = qc.evaluate(
        *tables(texts=("# And life is like a song #", "how are you")), SETTINGS
    )
    assert "lirik" in pool.loc[0, "qc_reason"]
    assert not pool.loc[0, "qc_pass"]
    assert pool.loc[1, "qc_pass"]


def test_missing_face_row_is_its_own_reason():
    pool = qc.evaluate(*tables(with_faces=False), SETTINGS)
    assert all("tanpa_wajah" in reason for reason in pool["qc_reason"])
    assert not pool["qc_pass"].any()


@pytest.mark.parametrize(
    "genre,multiplier,expected",
    [
        ("romansa", 1.5, 450),
        ("horor", 1.5, 675),
        ("Horror", 1.5, 675),
        ("thriller psikologis", 1.5, 675),
        ("horor", 2.0, 900),
        ("romansa", 2.0, 450),
    ],
)
def test_film_cap_follows_selection_and_genre(genre, multiplier, expected):
    assert (
        qc.film_cap(
            per_class=600,
            max_share=0.15,
            n_classes=5,
            genre=genre,
            rare_multiplier=multiplier,
        )
        == expected
    )


def test_choose_scenes_takes_everything_under_cap():
    order = [f"S{i:03d}" for i in range(1, 6)]
    counts = {scene: 10 for scene in order}
    assert qc.choose_scenes(order, counts, 500) == order


def test_choose_scenes_spreads_over_the_whole_film():
    order = [f"S{i:03d}" for i in range(1, 11)]
    counts = {scene: 100 for scene in order}
    chosen = qc.choose_scenes(order, counts, 350)
    assert len(chosen) == 4
    assert chosen[0] == "S001"
    assert "S010" in chosen
    assert sum(counts[scene] for scene in chosen) >= 350


def test_choose_scenes_without_cap_keeps_order():
    order = ["S002", "S001"]
    assert qc.choose_scenes(order, {"S002": 1, "S001": 1}, None) == order


def test_plan_scenes_skips_scenes_without_passing_utterances():
    films, utterances, faces = tables()
    utterances.loc[1, "text"] = ""
    pool = qc.evaluate(films, utterances, faces, SETTINGS)
    scenes = pd.DataFrame(
        [
            {"film_id": "AAA01", "scene_id": "AAA01-S001", "index": 1,
             "start_s": 0.0, "end_s": 10.0},
            {"film_id": "AAA01", "scene_id": "AAA01-S002", "index": 2,
             "start_s": 10.0, "end_s": 20.0},
        ]
    )
    plan = qc.plan_scenes(pool, scenes, {"AAA01": 100})
    row = plan.set_index("scene_id")
    assert row.loc["AAA01-S001", "selected"]
    assert row.loc["AAA01-S001", "n_qc_pass"] == 1
    assert not row.loc["AAA01-S002", "selected"]
    assert row.loc["AAA01-S002", "n_qc_pass"] == 0


def test_apply_plan_marks_only_selected_scenes():
    films, utterances, faces = tables()
    utterances.loc[1, "scene_id"] = "AAA01-S002"
    pool = qc.evaluate(films, utterances, faces, SETTINGS)
    plan = pd.DataFrame(
        [
            {"film_id": "AAA01", "scene_id": "AAA01-S001", "selected": True,
             "n_qc_pass": 1},
            {"film_id": "AAA01", "scene_id": "AAA01-S002", "selected": False,
             "n_qc_pass": 1},
        ]
    )
    marked = qc.apply_plan(pool, plan)
    assert list(marked["label_plan"]) == [True, False]


def test_reason_counts_counts_every_reason():
    pool = qc.evaluate(
        *tables(texts=("", ""), words=(0, 1), ratios=(0.2, 0.9)), SETTINGS
    )
    counts = qc.reason_counts(pool)
    assert counts["teks_kosong"] == 2
    assert counts["kata_sedikit"] == 2
    assert counts["wajah_jarang"] == 1
