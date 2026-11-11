from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from emosi import merge

FACE_COLUMNS = [
    "film_id",
    "utterance_id",
    "start_s",
    "face_ok",
    "emb_frames",
    "cluster",
    "actor_key",
    "emb_path",
    "emb_row",
]


def write_bundle(
    root: Path,
    owner: str,
    films: list[str],
    *,
    config_sha: str = "cfg",
    rejected: tuple[str, ...] = (),
    skip_embedding: tuple[str, ...] = (),
    corrupt_npy: str | None = None,
    skip_face: str | None = None,
) -> Path:
    """Bikin kiriman palsu yang bentuknya sama dengan hasil ``--pack``."""
    data = root / "data"
    (data / "faces").mkdir(parents=True, exist_ok=True)

    film_rows: list[dict] = []
    utterance_rows: list[dict] = []
    face_rows: list[dict] = []
    actor_rows: list[dict] = []
    scene_rows: list[dict] = []

    for film_id in films:
        film_rows.append(
            {
                "film_id": film_id,
                "title": f"Film {film_id}",
                "year": 2000,
                "path": f"/mesin/{owner}/{film_id}.mkv",
                "srt_path": f"/mesin/{owner}/{film_id}.srt",
                "pass_audition": film_id not in rejected,
            }
        )
        for index in (1, 2):
            utterance_id = f"{film_id}-U{index:05d}"
            utterance_rows.append(
                {
                    "film_id": film_id,
                    "utterance_id": utterance_id,
                    "scene_id": f"{film_id}-S001",
                    "index": index,
                    "start_s": index,
                    "end_s": index + 2,
                    "duration_s": 2,
                }
            )
            if film_id == skip_face:
                continue
            face_rows.append(
                {
                    "film_id": film_id,
                    "utterance_id": utterance_id,
                    "start_s": index,
                    "face_ok": True,
                    "emb_frames": 1,
                    "cluster": 1,
                    "actor_key": f"{film_id}-A01",
                    "emb_path": f"faces/{film_id}.npy",
                    "emb_row": index - 1,
                }
            )

        rows = 1 if film_id == corrupt_npy else 2
        matrix = np.zeros((rows, 8), dtype=np.float32)
        matrix[0, 0] = 1.0
        if rows > 1:
            matrix[1, 1] = 1.0
        if film_id not in skip_embedding:
            np.save(data / "faces" / f"{film_id}.npy", matrix)
        per_film = [row for row in face_rows if row["film_id"] == film_id]
        pd.DataFrame(per_film, columns=FACE_COLUMNS).to_csv(
            data / "faces" / f"{film_id}.csv", index=False
        )

        actor_rows.append(
            {
                "actor_key": f"{film_id}-A01",
                "film_id": film_id,
                "cluster": 1,
                "n_utterances": 2,
                "actor_label": "",
            }
        )
        scene_rows.append(
            {
                "film_id": film_id,
                "scene_id": f"{film_id}-S001",
                "index": 1,
                "start_s": 0,
                "end_s": 10,
            }
        )

    pd.DataFrame(film_rows).to_csv(data / "films.csv", index=False)
    pd.DataFrame(utterance_rows).to_csv(data / "utterances.csv", index=False)
    pd.DataFrame(face_rows, columns=FACE_COLUMNS).to_csv(data / "faces.csv", index=False)
    pd.DataFrame(actor_rows).to_csv(data / "actors.csv", index=False)
    pd.DataFrame(scene_rows).to_csv(data / "scenes.csv", index=False)

    manifest = {
        "format": merge.FORMAT,
        "owner": owner,
        "config_sha256": config_sha,
        "uv_lock_sha256": "lock",
        "films": films,
    }
    (root / merge.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return root


def levels(problems: list[merge.Problem]) -> set[str]:
    return {problem.level for problem in problems}


def test_neutralise_films_drops_machine_paths():
    films = pd.DataFrame(
        [{"film_id": "AAA01", "title": "A", "path": "/x/AAA01.mkv", "srt_path": "/x/AAA01.srt"}]
    )
    frame = merge.neutralise_films(films, "kesan")
    assert list(frame.columns) == ["owner", "film_id", "title"]
    assert frame.loc[0, "owner"] == "kesan"


def test_pack_files_collects_tables_and_embeddings(tmp_path):
    data = tmp_path / "data"
    (data / "faces").mkdir(parents=True)
    for name in ("films.csv", "utterances.csv", "faces.csv", "actors.csv", "scenes.csv"):
        (data / name).write_text("film_id\nAAA01\n", encoding="utf-8")
    (data / "faces" / "AAA01.npy").write_bytes(b"x")
    (data / "faces" / "AAA01.csv").write_text("film_id\nAAA01\n", encoding="utf-8")
    files = {path.name for path in merge.pack_files(data)}
    assert {"films.csv", "utterances.csv", "faces.csv", "actors.csv", "scenes.csv"} <= files
    assert {"AAA01.npy", "AAA01.csv"} <= files


def test_bundle_roundtrip_through_zip(tmp_path):
    root = write_bundle(tmp_path / "kiriman" / "kesan", "kesan", ["AAA01"])
    bundle = merge.write_bundle(
        tmp_path / "out" / "kesan.zip",
        tmp_path / "kiriman" / "kesan",
        merge.pack_files(root / "data"),
        json.loads((root / merge.MANIFEST).read_text(encoding="utf-8")),
    )
    contributor = merge.load_contributor(bundle)
    assert contributor.owner == "kesan"
    assert list(contributor.table("films.csv")["film_id"]) == ["AAA01"]
    assert len(contributor.embeddings["AAA01"]) == 2


def test_merge_two_contributors_keeps_owner(tmp_path):
    first = write_bundle(tmp_path / "k" / "kesan", "kesan", ["AAA01"])
    second = write_bundle(tmp_path / "k" / "budi", "budi", ["BBB02"])
    contributors = [merge.load_contributor(first), merge.load_contributor(second)]
    assert not merge.has_errors(merge.validate(contributors, config_sha256="cfg"))

    merged = merge.merge_contributors(contributors)
    assert list(merged.films["film_id"]) == ["AAA01", "BBB02"]
    assert list(merged.films["owner"]) == ["kesan", "budi"]
    assert "path" not in merged.films.columns
    assert len(merged.utterances) == 4
    assert set(merged.embeddings) == {"AAA01", "BBB02"}


def test_config_hash_mismatch_blocks_merge(tmp_path):
    contributor = merge.load_contributor(
        write_bundle(tmp_path / "kesan", "kesan", ["AAA01"], config_sha="beda")
    )
    problems = merge.validate([contributor], config_sha256="cfg")
    assert merge.has_errors(problems)
    assert any("config.yaml berbeda" in problem.message for problem in problems)


def test_duplicate_film_blocks_until_keep_is_given(tmp_path):
    first = merge.load_contributor(write_bundle(tmp_path / "k" / "kesan", "kesan", ["AAA01"]))
    second = merge.load_contributor(write_bundle(tmp_path / "k" / "budi", "budi", ["AAA01"]))

    _, problems = merge.resolve_duplicates([first, second])
    assert merge.has_errors(problems)

    cleaned, problems = merge.resolve_duplicates([first, second], keep="budi")
    assert not merge.has_errors(problems)
    assert cleaned[0].table("films.csv").empty
    assert list(cleaned[1].table("films.csv")["film_id"]) == ["AAA01"]


def test_embedding_row_count_mismatch_is_an_error(tmp_path):
    contributor = merge.load_contributor(
        write_bundle(tmp_path / "kesan", "kesan", ["AAA01"], corrupt_npy="AAA01")
    )
    problems = merge.validate([contributor], config_sha256="cfg")
    assert any("npy 1 baris" in problem.message for problem in problems)


def test_utterance_without_face_row_is_an_error(tmp_path):
    contributor = merge.load_contributor(
        write_bundle(tmp_path / "kesan", "kesan", ["AAA01"], skip_face="AAA01")
    )
    problems = merge.validate([contributor], config_sha256="cfg")
    assert any("tanpa baris wajah" in problem.message for problem in problems)


def test_embedding_problems_detects_bad_rows():
    faces = pd.DataFrame({"emb_row": [0, 1]})
    assert merge.embedding_problems(faces, 2) == []
    assert merge.embedding_problems(pd.DataFrame({"emb_row": [0, 0]}), 2)
    assert merge.embedding_problems(pd.DataFrame({"emb_row": [1, 2]}), 3)
    assert merge.embedding_problems(pd.DataFrame({"emb_row": []}), 0) == []


def test_rejected_film_is_dropped_with_its_rows(tmp_path):
    contributor = merge.load_contributor(
        write_bundle(tmp_path / "kesan", "kesan", ["AAA01", "BBB02"], rejected=("BBB02",))
    )
    merged = merge.merge_contributors([contributor])
    assert list(merged.films["film_id"]) == ["AAA01"]
    assert merged.dropped == ["BBB02"]
    assert set(merged.utterances["film_id"]) == {"AAA01"}
    assert set(merged.embeddings) == {"AAA01"}


def test_rejected_film_without_embedding_does_not_block_pack_or_merge(tmp_path):
    root = write_bundle(
        tmp_path / "budi",
        "budi",
        ["AAA01", "BBB02"],
        rejected=("BBB02",),
        skip_embedding=("BBB02",),
    )
    manifest = merge.build_manifest(
        owner="budi",
        data_dir=root / "data",
        config_sha256="cfg",
        uv_lock_sha256="lock",
    )
    assert manifest["films"] == ["AAA01"]

    contributor = merge.load_contributor(root)
    assert set(contributor.embeddings) == {"AAA01"}

    problems = merge.validate([contributor], config_sha256="cfg", uv_lock_sha256="lock")
    assert not merge.has_errors(problems), problems

    merged = merge.merge_contributors([contributor])
    assert list(merged.films["film_id"]) == ["AAA01"]
    assert merged.dropped == ["BBB02"]
    assert set(merged.embeddings) == {"AAA01"}


def test_cross_film_candidates_only_pairs_different_films():
    same = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    near = np.array([0.99, 0.1, 0.0], dtype=np.float32)
    near = (near / np.linalg.norm(near)).astype(np.float32)
    far = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    centroids = {
        "AAA01-A01": same,
        "AAA01-A02": near,
        "BBB02-A01": near,
        "BBB02-A02": far,
    }
    candidates = merge.cross_film_candidates(centroids, distance=0.45)
    pairs = {(row.actor_a, row.actor_b) for row in candidates.itertuples()}
    assert ("AAA01-A01", "BBB02-A01") in pairs
    assert ("AAA01-A01", "AAA01-A02") not in pairs
    assert ("AAA01-A01", "BBB02-A02") not in pairs


def test_actor_centroids_skip_empty_clusters(tmp_path):
    faces = pd.DataFrame(
        [
            {"film_id": "AAA01", "actor_key": "AAA01-A01", "emb_row": 0},
            {"film_id": "AAA01", "actor_key": "AAA01-A01", "emb_row": 1},
            {"film_id": "AAA01", "actor_key": "", "emb_row": 0},
        ]
    )
    matrix = np.eye(3, dtype=np.float32)
    centroids = merge.actor_centroids(faces, {"AAA01": matrix})
    assert list(centroids) == ["AAA01-A01"]
    assert np.linalg.norm(centroids["AAA01-A01"]) == pytest.approx(1.0, abs=1e-5)


def test_missing_required_table_raises(tmp_path):
    root = write_bundle(tmp_path / "kesan", "kesan", ["AAA01"])
    (root / "data" / "faces.csv").unlink()
    with pytest.raises(merge.MergeError, match="berkas wajib"):
        merge.load_contributor(root)
