from __future__ import annotations

import pandas as pd
import pytest

from emosi import split

RATIOS = (0.7, 0.15, 0.15)


def make_frame(films: int = 3, per_film: int = 30, labels: int = 2) -> pd.DataFrame:
    rows = []
    classes = ["senang", "sedih", "marah", "takut", "netral"][:labels]
    for film in range(1, films + 1):
        film_id = f"F{film:03d}"
        for index in range(per_film):
            rows.append(
                {
                    "utterance_id": f"{film_id}-U{index:03d}",
                    "clip_id": f"{film_id}-U{index:03d}",
                    "film_id": film_id,
                    "label": classes[index % len(classes)],
                    "actor_key": f"{film_id}-A01",
                    "emb_frames": 2,
                }
            )
    return pd.DataFrame(rows)


def test_random_split_is_stratified_per_class():
    frame = make_frame(films=2, per_film=100, labels=2)
    assignment = split.assign_random(frame, ratios=RATIOS, seed=1)
    for label in ("senang", "sedih"):
        part = assignment[frame["label"] == label]
        counts = part.value_counts()
        assert counts["train"] == pytest.approx(70, abs=1)
        assert counts["val"] == pytest.approx(15, abs=1)
        assert counts["test"] == pytest.approx(15, abs=1)


def test_random_split_is_deterministic():
    frame = make_frame()
    first = split.assign_random(frame, ratios=RATIOS, seed=3)
    second = split.assign_random(frame, ratios=RATIOS, seed=3)
    assert first.equals(second)


def test_film_split_never_splits_a_film():
    frame = make_frame(films=10, per_film=10, labels=5)
    assignment = split.assign_groups(frame["film_id"], ratios=RATIOS)
    per_film = pd.DataFrame({"film": frame["film_id"], "split": assignment})
    assert per_film.groupby("film")["split"].nunique().max() == 1
    assert split.leakage(assignment, frame["film_id"])["leaked"] == 0


def test_film_split_covers_all_three_splits():
    frame = make_frame(films=3, per_film=10, labels=5)
    assignment = split.assign_groups(frame["film_id"], ratios=RATIOS)
    assert set(assignment) == set(split.SPLITS)


def test_actor_split_keeps_an_actor_together():
    frame = make_frame(films=6, per_film=20, labels=5)
    keys = pd.Series([f"aktor:A{index % 4}" for index in range(len(frame))])
    assignment = split.assign_groups(keys, ratios=RATIOS)
    table = pd.DataFrame({"split": assignment, "key": keys})
    assert table.groupby("key")["split"].nunique().max() == 1


def test_group_keys_prefers_actor_label_then_cluster_then_utterance():
    frame = pd.DataFrame(
        [
            {"utterance_id": "u1", "actor_key": "F-A01", "emb_frames": 2},
            {"utterance_id": "u2", "actor_key": "F-A02", "emb_frames": 2},
            {"utterance_id": "u3", "actor_key": "", "emb_frames": 0},
        ]
    )
    keys = split.group_keys(frame, actor_labels={"F-A01": "Rachel"})
    assert list(keys) == ["aktor:Rachel", "klaster:F-A02", "ujaran:u3"]


def test_leakage_counts_groups_in_multiple_splits():
    assignment = pd.Series(["train", "val", "train", "test"])
    keys = pd.Series(["a", "a", "b", "c"])
    result = split.leakage(assignment, keys)
    assert result == {"groups": 3, "leaked": 1}


def test_share_sums_to_one():
    assignment = pd.Series(["train"] * 7 + ["val"] * 2 + ["test"] * 1)
    shares = split.share(assignment)
    assert shares == {"train": 0.7, "val": 0.2, "test": 0.1}


def test_class_table_has_a_row_per_class():
    frame = make_frame(films=2, per_film=10, labels=3)
    assignment = split.assign_random(frame, ratios=RATIOS, seed=0)
    table = split.class_table(assignment, frame["label"], ["senang", "sedih", "marah", "takut"])
    assert list(table.index) == ["senang", "sedih", "marah", "takut"]
    assert list(table.columns) == list(split.SPLITS)
    assert table.loc["takut"].sum() == 0
