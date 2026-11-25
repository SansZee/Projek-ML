from __future__ import annotations

import json
import zipfile

import pandas as pd
import pytest

from emosi import clips


def make_clip(media, clip_id, film_id="AAA01", status="ok", sizes=(120, 40, 12)):
    paths = {
        "clip_path": f"clips/{clip_id}.mp4",
        "audio_path": f"audio/{clip_id}.wav",
        "face_path": f"faces/{clip_id}.mp4",
    }
    for relative, size in zip(paths.values(), sizes, strict=True):
        path = media / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
    return {
        "clip_id": clip_id,
        "utterance_id": clip_id.replace("C", "U"),
        "film_id": film_id,
        "label": "senang",
        "confidence": 4.0,
        "duration_s": 3.0,
        **paths,
        "clip_bytes": sizes[0],
        "audio_bytes": sizes[1],
        "face_bytes": sizes[2],
        "status": status,
    }


def catalog(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=clips.CATALOG_COLUMNS)


def test_pack_keeps_only_own_films_and_writes_manifest(tmp_path):
    media = tmp_path / "media"
    rows = [
        make_clip(media, "C001", "AAA01"),
        make_clip(media, "C002", "BBB02"),
        make_clip(media, "C003", "AAA01", status="film_tidak_ada"),
    ]
    dest = tmp_path / "kiriman" / "clips-budi-2026-11-25.zip"

    count, size = clips.pack(
        catalog(rows), media, dest, owner="budi", films={"AAA01"}
    )

    assert count == 1
    assert size > 0
    with zipfile.ZipFile(dest) as bundle:
        names = set(bundle.namelist())
        assert {"clips/C001.mp4", "audio/C001.wav", "faces/C001.mp4"} <= names
        assert "clips/C002.mp4" not in names
        manifest = json.loads(bundle.read(clips.MANIFEST))
        assert manifest["owner"] == "budi"
        assert manifest["clips"] == 1
        packed = pd.read_csv(bundle.open(clips.CATALOG))
        assert list(packed["clip_id"]) == ["C001"]


def test_pack_refuses_missing_file(tmp_path):
    media = tmp_path / "media"
    rows = [make_clip(media, "C001")]
    (media / "clips/C001.mp4").unlink()

    with pytest.raises(clips.ClipError):
        clips.pack(catalog(rows), media, tmp_path / "x.zip", owner="budi")


def test_merge_extracts_and_marks_what_is_missing(tmp_path):
    media = tmp_path / "media"
    a = clips.pack(
        catalog([make_clip(media, "C001"), make_clip(media, "C002")]),
        media,
        tmp_path / "clips-ani-2026-11-25.zip",
        owner="ani",
    )
    assert a[0] == 2

    target = tmp_path / "gabung"
    merged, report = clips.merge_bundles(
        [tmp_path / "clips-ani-2026-11-25.zip"], target
    )
    assert len(merged) == 2
    assert (target / "clips/C001.mp4").exists()
    assert report[0]["owner"] == "ani"

    selected = pd.DataFrame(
        [
            {"clip_id": "C001", "utterance_id": "U001", "film_id": "AAA01",
             "label": "senang", "confidence": 4.0, "duration_s": 3.0},
            {"clip_id": "C002", "utterance_id": "U002", "film_id": "AAA02",
             "label": "sedih", "confidence": 4.0, "duration_s": 2.0},
            {"clip_id": "C009", "utterance_id": "U009", "film_id": "AAA03",
             "label": "marah", "confidence": 3.0, "duration_s": 4.0},
        ]
    )
    final, missing = clips.final_catalog(merged, selected)
    assert len(final) == 3
    assert list(missing["clip_id"]) == ["C009"]
    status = dict(zip(final["clip_id"], final["status"], strict=True))
    assert status["C001"] == "ok"
    assert status["C009"] == clips.MISSING_STATUS


def test_merge_rejects_the_same_clip_from_two_people(tmp_path):
    media = tmp_path / "media"
    row = make_clip(media, "C001")
    clips.pack(catalog([row]), media, tmp_path / "clips-ani-2026-11-25.zip", owner="ani")
    clips.pack(catalog([row]), media, tmp_path / "clips-budi-2026-11-25.zip", owner="budi")

    with pytest.raises(clips.ClipError, match="dua zip"):
        clips.merge_bundles(
            [tmp_path / "clips-ani-2026-11-25.zip", tmp_path / "clips-budi-2026-11-25.zip"],
            tmp_path / "gabung",
        )
