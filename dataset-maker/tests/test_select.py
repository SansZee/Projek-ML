from __future__ import annotations

import pandas as pd

from emosi import select

CLASSES = ["senang", "sedih", "marah", "takut", "netral"]


def make_labels(
    *,
    films: int = 3,
    per_class: int = 20,
    classes: tuple[str, ...] = ("senang", "sedih"),
    confidence: int = 4,
) -> pd.DataFrame:
    rows = []
    for film in range(1, films + 1):
        film_id = f"F{film:03d}"
        for label in classes:
            for index in range(per_class):
                rows.append(
                    {
                        "utterance_id": f"{film_id}-{label}-{index:03d}",
                        "film_id": film_id,
                        "scene_id": f"{film_id}-S001",
                        "owner": "kesan",
                        "title": f"Film {film}",
                        "label": label,
                        "confidence": confidence,
                        "status": "sepakat",
                        "n_raters": 2,
                        "agreement": 1.0,
                        "start_s": 100.0 + index,
                        "end_s": 103.0 + index,
                        "duration_s": 3.0,
                        "actor_key": f"{film_id}-A01",
                        "emb_frames": 3,
                    }
                )
    return pd.DataFrame(rows)


def test_eligible_drops_non_classes_and_low_confidence():
    frame = make_labels(classes=("senang",))
    frame.loc[0, "label"] = "ambigu"
    frame.loc[1, "label"] = "tidak_layak"
    frame.loc[2, "confidence"] = 2
    kept, dropped = select.eligible(frame, classes=["senang"], min_conf=3)
    assert len(kept) == len(frame) - 3
    assert dropped["bukan_kelas"] == 2
    assert dropped["keyakinan_rendah"] == 1


def test_selection_respects_per_film_cap():
    frame = make_labels(films=3, per_class=20, classes=("senang",))
    result = select.select_rows(
        frame, classes=["senang"], per_class=30, max_share=0.15
    )
    # 600 × 0,15 = 90, tetapi kuota ujinya 30 → batas 4 per film.
    assert len(result.rows) == 12
    assert result.rows.groupby("film_id").size().tolist() == [4, 4, 4]
    assert result.shortfall["senang"] == 18


def test_selection_round_robins_across_films():
    frame = make_labels(films=4, per_class=20, classes=("senang",))
    result = select.select_rows(
        frame, classes=["senang"], per_class=40, max_share=0.15
    )
    # Batas 6 per film; 40 diminta, 4 × 6 = 24 tersedia.
    assert result.rows.groupby("film_id").size().tolist() == [6, 6, 6, 6]


def test_higher_confidence_wins_inside_a_film():
    frame = make_labels(films=1, per_class=5, classes=("senang",), confidence=4)
    frame.loc[0, "confidence"] = 5
    frame.loc[0, "duration_s"] = 1.0
    result = select.select_rows(
        frame, classes=["senang"], per_class=1, max_share=1.0
    )
    assert result.rows.loc[0, "utterance_id"] == frame.loc[0, "utterance_id"]


def test_full_quota_when_films_are_enough():
    frame = make_labels(films=10, per_class=20, classes=("senang",))
    result = select.select_rows(
        frame, classes=["senang"], per_class=60, max_share=0.15
    )
    # 60 × 0,15 = 9 per film; 10 film menyediakan 90, cukup.
    assert len(result.rows) == 60
    assert result.complete
    assert result.rows.groupby("film_id").size().max() <= 9


def test_empty_class_is_reported_as_shortfall():
    frame = make_labels(films=2, classes=("senang",))
    result = select.select_rows(
        frame, classes=["senang", "sedih"], per_class=10, max_share=0.5
    )
    assert result.shortfall["sedih"] == 10
    assert set(result.rows["label"]) == {"senang"}


def test_selection_is_deterministic():
    frame = make_labels(films=3, per_class=10)
    first = select.select_rows(frame, classes=list(CLASSES), per_class=12, max_share=0.4)
    second = select.select_rows(frame, classes=list(CLASSES), per_class=12, max_share=0.4)
    assert first.rows.equals(second.rows)


def test_clip_id_follows_utterance_id_and_columns_are_fixed():
    frame = make_labels(films=2, classes=("senang",))
    result = select.select_rows(frame, classes=["senang"], per_class=4, max_share=1.0)
    assert list(result.rows.columns) == select.SELECTED_COLUMNS
    assert (result.rows["clip_id"] == result.rows["utterance_id"]).all()


def test_class_counts_covers_every_class():
    frame = make_labels(films=2, classes=("senang", "sedih"))
    result = select.select_rows(frame, classes=list(CLASSES), per_class=5, max_share=1.0)
    counts = select.class_counts(result.rows, list(CLASSES))
    assert counts["senang"] == 5
    assert counts["sedih"] == 5
    assert counts["takut"] == 0


def test_per_film_matrix_totals_match():
    frame = make_labels(films=3, classes=("senang", "sedih"))
    result = select.select_rows(frame, classes=["senang", "sedih"], per_class=10, max_share=1.0)
    assert result.per_film["total"].sum() == len(result.rows)
