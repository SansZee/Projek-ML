from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from emosi import annotate


def make_pool(
    films: list[tuple[str, str, str]] | None = None,
    *,
    scenes: int = 3,
    per_scene: int = 10,
) -> pd.DataFrame:
    films = films or [("AAA01", "kesan", "romansa"), ("BBB02", "budi", "horor")]
    rows = []
    for film_id, owner, genre in films:
        index = 0
        for scene in range(1, scenes + 1):
            scene_id = f"{film_id}-S{scene:03d}"
            for _ in range(per_scene):
                index += 1
                rows.append(
                    {
                        "film_id": film_id,
                        "owner": owner,
                        "title": f"Film {film_id}",
                        "genre": genre,
                        "scene_id": scene_id,
                        "utterance_id": f"{film_id}-U{index:05d}",
                        "index": index,
                        "label_plan": True,
                    }
                )
    return pd.DataFrame(rows)


def test_own_assignment_covers_every_planned_utterance():
    pool = make_pool()
    assignments = annotate.build_assignments(
        pool, people=["kesan", "budi"], gold=0, cross=5
    )
    own = assignments[assignments["role"] == annotate.ROLE_OWN]
    for film_id, owner in (("AAA01", "kesan"), ("BBB02", "budi")):
        expected = set(pool.loc[pool["film_id"] == film_id, "utterance_id"])
        got = set(own.loc[own["annotator"] == owner, "utterance_id"])
        assert got == expected


def test_gold_scenes_are_labeled_by_everyone_as_gold():
    pool = make_pool()
    assignments = annotate.build_assignments(
        pool, people=["kesan", "budi"], gold=1, cross=5
    )
    gold_scenes = set(assignments.loc[assignments["role"] == "gold", "scene_id"])
    assert len(gold_scenes) == 1
    scene = gold_scenes.pop()

    for person in ("kesan", "budi"):
        rows = assignments[
            (assignments["annotator"] == person) & (assignments["scene_id"] == scene)
        ]
        assert set(rows["role"]) == {annotate.ROLE_GOLD}

    assert not assignments[
        (assignments["role"] == annotate.ROLE_OWN)
        & (assignments["scene_id"] == scene)
    ].shape[0]


def test_balanced_assignments_split_scenes_not_people_fairly():
    pool = make_pool(
        films=[
            ("AAA01", "kesan", "romansa"),
            ("BBB02", "budi", "horor"),
            ("CCC03", "budi", "horor"),
            ("DDD04", "budi", "horor"),
        ],
        scenes=4,
        per_scene=10,
    )
    assignments = annotate.build_assignments(
        pool, people=["kesan", "budi"], gold=0, cross=5, balance=True
    )
    own = assignments[assignments["role"] == annotate.ROLE_OWN]

    assert own.groupby("scene_id")["annotator"].nunique().max() == 1
    load = own.groupby("annotator").size().to_dict()
    assert set(load) == {"kesan", "budi"}
    assert max(load.values()) - min(load.values()) <= 10
    assert own.groupby("scene_id")["owner"].first().nunique() == 2


def test_balanced_assignments_keep_labeled_scenes_with_their_labeler():
    pool = make_pool(
        films=[("AAA01", "kesan", "romansa"), ("BBB02", "budi", "horor")],
        scenes=4,
        per_scene=10,
    )
    scene = "BBB02-S004"
    assignments = annotate.build_assignments(
        pool,
        people=["kesan", "budi"],
        gold=0,
        cross=5,
        balance=True,
        existing={"kesan": {scene}},
    )
    own = assignments[
        (assignments["role"] == annotate.ROLE_OWN) & (assignments["scene_id"] == scene)
    ]
    assert set(own["annotator"]) == {"kesan"}
    assert set(own["owner"]) == {"budi"}


def test_balanced_assignments_prefer_owner_when_two_labeled_the_scene():
    pool = make_pool(
        films=[("AAA01", "kesan", "romansa"), ("BBB02", "budi", "horor")],
        scenes=3,
        per_scene=10,
    )
    assignments = annotate.build_assignments(
        pool,
        people=["kesan", "budi"],
        gold=0,
        cross=5,
        balance=True,
        existing={"budi": {"AAA01-S002"}, "kesan": {"AAA01-S002"}},
    )
    own = assignments[
        (assignments["role"] == annotate.ROLE_OWN)
        & (assignments["scene_id"] == "AAA01-S002")
    ]
    assert set(own["annotator"]) == {"kesan"}


def test_cross_rows_come_from_someone_else_and_have_fixed_size():
    pool = make_pool()
    assignments = annotate.build_assignments(
        pool, people=["kesan", "budi"], gold=0, cross=12
    )
    cross = assignments[assignments["role"] == annotate.ROLE_CROSS]
    assert set(cross["annotator"]) == {"kesan", "budi"}
    for person, group in cross.groupby("annotator"):
        assert (group["owner"] != person).all()
        assert len(group) == 12


def test_people_without_films_share_the_same_cross_set():
    pool = make_pool([("AAA01", "kesan", "romansa")])
    assignments = annotate.build_assignments(
        pool, people=["kesan", "budi", "ani"], gold=0, cross=7
    )
    budi = assignments[
        (assignments["annotator"] == "budi") & (assignments["role"] == "silang")
    ]
    ani = assignments[
        (assignments["annotator"] == "ani") & (assignments["role"] == "silang")
    ]
    assert list(budi["utterance_id"]) == list(ani["utterance_id"])
    assert len(budi) == 7
    assert assignments[
        (assignments["annotator"] == "kesan") & (assignments["role"] == "silang")
    ].empty


def test_cross_never_reuses_gold_utterances():
    pool = make_pool()
    assignments = annotate.build_assignments(
        pool, people=["kesan", "budi"], gold=1, cross=10
    )
    for _, group in assignments.groupby("annotator"):
        gold = set(group.loc[group["role"] == "gold", "utterance_id"])
        cross = set(group.loc[group["role"] == "silang", "utterance_id"])
        assert not gold & cross


def test_assignment_is_unique_per_annotator_and_utterance():
    pool = make_pool()
    assignments = annotate.build_assignments(
        pool, people=["kesan", "budi"], gold=1, cross=10
    )
    assert not assignments.duplicated(["annotator", "utterance_id"]).any()


def test_unknown_owner_is_rejected():
    pool = make_pool()
    with pytest.raises(annotate.AssignmentError, match="tidak ada di daftar"):
        annotate.build_assignments(pool, people=["kesan", "ani"], gold=0, cross=5)


def test_empty_pool_is_rejected():
    pool = make_pool()
    pool["label_plan"] = False
    with pytest.raises(annotate.AssignmentError, match="label_plan"):
        annotate.build_assignments(pool, people=["kesan", "budi"])


def test_pick_gold_prefers_rare_genre_and_distinct_films():
    pool = make_pool(
        [
            ("AAA01", "kesan", "romansa"),
            ("BBB02", "budi", "horor"),
            ("CCC03", "ani", "horor"),
        ]
    )
    scenes = annotate.pick_gold_scenes(pool, 2, seed=1)
    assert len(scenes) == 2
    films = {scene.split("-S")[0] for scene in scenes}
    assert len(films) == 2


def test_workload_totals_match_assignment():
    pool = make_pool()
    assignments = annotate.build_assignments(
        pool, people=["kesan", "budi"], gold=1, cross=6
    )
    table = annotate.workload(assignments).set_index("annotator")
    for person, group in assignments.groupby("annotator"):
        assert table.loc[person, "total"] == len(group)
    assert table["total"].sum() == len(assignments)


def test_scene_needs_lists_every_annotator():
    pool = make_pool()
    assignments = annotate.build_assignments(
        pool, people=["kesan", "budi"], gold=1, cross=6
    )
    needs = annotate.scene_needs(assignments)
    row = needs[needs["scene_id"] == needs.iloc[0]["scene_id"]].iloc[0]
    assert row["n_utterances"] > 0
    assert row["annotators"]
    assert annotate.scenes_for(needs, "kesan") == sorted(
        set(assignments.loc[assignments["annotator"] == "kesan", "scene_id"])
    )


def test_preview_command_keeps_audio_and_limits_size(tmp_path: Path):
    command = annotate.preview_command(
        Path("/film/AAA01.mkv"),
        tmp_path / "AAA01-S001.mp4",
        12.5,
        30.0,
        scale=480,
        crf=30,
    )
    joined = " ".join(command)
    assert command[0] == "ffmpeg"
    assert "-ss 12.500" in joined
    assert "-t 17.500" in joined
    assert "scale=-2:480" in joined
    assert "libx264" in joined
    assert "-c:a aac" in joined
    assert str(tmp_path / "AAA01-S001.mp4") == command[-1]


def test_upsert_replaces_the_same_utterance(tmp_path: Path):
    path = tmp_path / "labels_kesan.csv"
    row = {
        "annotator": "kesan",
        "utterance_id": "AAA01-U00001",
        "film_id": "AAA01",
        "scene_id": "AAA01-S001",
        "role": "milik",
        "label": "senang",
        "confidence": 4,
        "notes": "",
        "labeled_at": "2026-01-01T00:00:00+00:00",
    }
    annotate.upsert_labels(path, [row])
    annotate.upsert_labels(path, [{**row, "label": "sedih"}])
    frame = annotate.read_labels(path)
    assert len(frame) == 1
    assert frame.loc[0, "label"] == "sedih"


def test_upsert_keeps_labels_of_other_people(tmp_path: Path):
    path = tmp_path / "labels_kesan.csv"
    base = {
        "film_id": "AAA01",
        "scene_id": "AAA01-S001",
        "role": "milik",
        "label": "netral",
        "confidence": 3,
        "notes": "",
        "labeled_at": "2026-01-01T00:00:00+00:00",
    }
    annotate.upsert_labels(
        path,
        [
            {**base, "annotator": "kesan", "utterance_id": "AAA01-U00001"},
            {**base, "annotator": "budi", "utterance_id": "AAA01-U00001"},
        ],
    )
    assert set(annotate.read_labels(path)["annotator"]) == {"kesan", "budi"}


def test_progress_counts_only_this_annotator(tmp_path: Path):
    pool = make_pool()
    assignments = annotate.build_assignments(
        pool, people=["kesan", "budi"], gold=0, cross=6
    )
    mine = assignments[assignments["annotator"] == "kesan"]
    labels = pd.DataFrame(
        [
            {"annotator": "kesan", "utterance_id": mine["utterance_id"].iloc[0]},
            {"annotator": "budi", "utterance_id": mine["utterance_id"].iloc[1]},
        ]
    )
    done, total = annotate.progress(labels, assignments, "kesan")
    assert (done, total) == (1, len(mine))


def test_video_window_prevents_invalid_streamlit_end_time():
    # Kasus normal di tengah adegan
    assert annotate.video_window(14.575, 9.12, 148.732) == (14, 24)
    # Kasus nyata GBH14-U00079: offset dan batas adegan sama-sama di detik 148
    assert annotate.video_window(148.095, 3.72, 148.732) == (148, None)
    # Kasus offset melewati panjang adegan (POH06 / TSR94)
    assert annotate.video_window(142.370, 2.1, 141.600) == (141, None)
    # Adegan sangat pendek (< 1 detik)
    assert annotate.video_window(0.2, 1.5, 0.8) == (0, None)


def test_step_utterance_sequential_and_prefer_unlabeled():
    utts = ["u1", "u2", "u3", "u4"]
    done = {"u1", "u2"}
    assert annotate.step_utterance(
        utts, "u2", done, direction=1, prefer_unlabeled=True
    ) == "u3"
    assert annotate.step_utterance(
        utts, "u4", {"u1", "u3", "u4"}, direction=1, prefer_unlabeled=True
    ) == "u2"
    # Adegan sudah selesai semua -> tetap maju berurutan untuk review
    assert annotate.step_utterance(
        utts, "u2", set(utts), direction=1, prefer_unlabeled=True
    ) == "u3"
    assert annotate.step_utterance(
        utts, "u4", set(utts), direction=1, prefer_unlabeled=True
    ) is None
    assert annotate.step_utterance(utts, "u3", done, direction=-1) == "u2"
    assert annotate.step_utterance(utts, "u1", done, direction=-1) is None


def test_next_scene_in_scope_stays_within_film():
    film_scenes = ["S01", "S02", "S03", "S04"]
    unfinished = {"S01", "S03"}
    assert annotate.next_scene_in_scope(film_scenes, "S01", unfinished) == "S03"
    assert annotate.next_scene_in_scope(film_scenes, "S03", unfinished) == "S01"
    assert annotate.next_scene_in_scope(film_scenes, "S01", set()) is None
