from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from emosi.face import (
    Detection,
    FaceSettings,
    FrameFaces,
    cluster_embeddings,
    detections_from_result,
    dominant_face,
    frames_by_utterance,
    scaled_size,
    summarize_frames,
)


def settings(**overrides) -> FaceSettings:
    values = {
        "sample_fps": 2.0,
        "frame_width": 960,
        "min_face_px": 48,
        "dominance_min_ratio": 1.3,
        "min_face_frames": 1,
        "face_ratio_ok": 0.6,
    }
    values.update(overrides)
    return FaceSettings(**values)


def box(width: float, height: float, *, x: float = 0.0, y: float = 0.0) -> tuple[float, ...]:
    return (x, y, x + width, y + height)


def detection(width: float, height: float, *, score: float = 0.9, unit=None) -> Detection:
    return Detection(
        *box(width, height),
        score=score,
        embedding=None if unit is None else np.asarray(unit, dtype=np.float32),
    )


def test_scaled_size_even_and_no_upscale():
    assert scaled_size(1920, 800, 960) == (960, 400)
    assert scaled_size(1920, 814, 960) == (960, 408)
    assert scaled_size(640, 360, 960) == (640, 360)
    assert scaled_size(1920, 817, 960)[1] % 2 == 0
    with pytest.raises(ValueError):
        scaled_size(0, 800, 960)


def test_detections_from_result_filters_small_faces():
    result = [
        SimpleNamespace(bbox=box(120, 120), det_score=0.9, normed_embedding=np.ones(4)),
        SimpleNamespace(bbox=box(30, 30), det_score=0.8, normed_embedding=np.ones(4)),
    ]
    detections = detections_from_result(result, min_face_px=48)
    assert len(detections) == 1
    assert detections[0].embedding is not None
    assert detections[0].area == pytest.approx(120 * 120)


def test_dominant_face_prefers_largest_and_flags_ties():
    small = detection(80, 80)
    large = detection(160, 160)
    best, unambiguous, count = dominant_face([small, large], dominance_min_ratio=1.3)
    assert best is large
    assert unambiguous
    assert count == 2

    best, unambiguous, _ = dominant_face([detection(100, 100), detection(105, 105)])
    assert best is not None
    assert not unambiguous


def test_summarize_frames_counts_and_pools_embedding():
    first = [1.0, 0.0, 0.0]
    second = [0.8, 0.2, 0.0]
    frames = [
        FrameFaces(960, 400, [detection(200, 200, unit=first)]),
        FrameFaces(960, 400, []),
        FrameFaces(960, 400, [detection(200, 200, unit=second)]),
    ]
    summary = summarize_frames(frames, settings())
    assert summary.n_frames == 3
    assert summary.n_face_frames == 2
    assert summary.face_ratio == pytest.approx(2 / 3, abs=1e-4)
    assert summary.n_faces_max == 1
    assert summary.emb_frames == 2
    assert summary.embedding is not None
    assert np.linalg.norm(summary.embedding) == pytest.approx(1.0, abs=1e-5)
    assert summary.face_width_frac == pytest.approx(200 / 960, abs=1e-4)


def test_summarize_frames_skips_ambiguous_embedding_but_keeps_ratio():
    frames = [FrameFaces(960, 400, [detection(150, 150), detection(150, 150)])]
    summary = summarize_frames(frames, settings())
    assert summary.face_ratio == 1.0
    assert summary.multi_face_share == 1.0
    assert summary.dominant_share == 0.0
    assert summary.embedding is None


def test_summarize_empty_frames():
    summary = summarize_frames([], settings())
    assert summary.n_frames == 0
    assert summary.face_ratio == 0.0
    assert summary.embedding is None


def test_frames_by_utterance_handles_overlap_and_short_clip():
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    spans = [(0.0, 1.0), (0.5, 2.5), (13.0, 13.4)]
    buckets = frames_by_utterance(times, spans)
    assert buckets[0] == [0, 1]
    assert buckets[1] == [1, 2, 3, 4]
    # Klip 0,4 detik tidak kebagian frame: dipakai frame terdekat, bukan kosong.
    assert buckets[2] == [4]


def test_cluster_embeddings_groups_similar_faces():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(3, 8)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    vectors = np.stack(
        [
            base[0],
            base[0] + 0.02 * rng.normal(size=8),
            base[1],
            base[1] + 0.02 * rng.normal(size=8),
            base[2],
        ]
    ).astype(np.float32)
    labels = cluster_embeddings(vectors, threshold=0.2, min_size=2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    # Klaster tunggal di bawah min_size tidak dipaksa masuk klaster lain.
    assert labels[4] == -1


def test_cluster_embeddings_empty():
    labels = cluster_embeddings(np.zeros((0, 8), dtype=np.float32))
    assert labels.shape == (0,)


def test_cluster_embeddings_respects_order():
    rng = np.random.default_rng(1)
    a = rng.normal(size=8).astype(np.float32)
    b = rng.normal(size=8).astype(np.float32)
    vectors = np.stack([a, a + 0.01, b])
    labels = cluster_embeddings(vectors, threshold=0.2, min_size=2, order=[2, 0, 1])
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]
