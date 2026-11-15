from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emosi import face, face_preview


def fake_frames(*args, **kwargs):
    for index in range(8):
        yield np.full((360, 640, 3), index * 8, dtype=np.uint8)


DETECTION = face.Detection(x1=100, y1=80, x2=260, y2=240, score=0.9)


def test_target_times_uses_quarters():
    assert face_preview.target_times(10.0, 14.0) == [11.0, 12.0, 13.0]


def test_target_times_keeps_order_when_clip_is_empty():
    times = face_preview.target_times(5.0, 5.0)

    assert times == sorted(times)
    assert all(moment > 5.0 for moment in times)


def test_compose_strip_fills_missing_tile_with_gray():
    tile = np.full((224, 224, 3), 200, dtype=np.uint8)

    strip = face_preview.compose_strip([tile, None, tile])

    assert strip.shape == (224, 672, 3)
    assert strip[0, 300, 0] == face_preview.EMPTY
    assert strip[0, 10, 0] == 200


def test_strip_tiles_crops_three_faces(monkeypatch):
    pytest.importorskip("cv2")
    monkeypatch.setattr(face, "iter_rgb_frames", fake_frames)
    monkeypatch.setattr(
        face, "analyze_frame", lambda analyzer, frame, min_face_px: [DETECTION]
    )

    tiles = face_preview.strip_tiles(
        None, Path("film.mkv"), 10.0, 14.0, 640, 360, settings=face.FaceSettings()
    )

    assert len(tiles) == 3
    assert all(tile is not None and tile.shape == (224, 224, 3) for tile in tiles)


def test_strip_tiles_marks_missing_face(monkeypatch):
    pytest.importorskip("cv2")
    monkeypatch.setattr(face, "iter_rgb_frames", fake_frames)
    monkeypatch.setattr(
        face, "analyze_frame", lambda analyzer, frame, min_face_px: []
    )

    tiles = face_preview.strip_tiles(
        None, Path("film.mkv"), 10.0, 14.0, 640, 360, settings=face.FaceSettings()
    )

    assert tiles == [None, None, None]
