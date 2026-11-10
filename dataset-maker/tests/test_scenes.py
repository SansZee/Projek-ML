from __future__ import annotations

import pytest

from emosi.scenes import (
    annotation_window,
    build_scenes,
    count_shots,
    dialogue_stats,
    timecode,
)
from emosi.subtitle import Cue


def assert_contiguous(spans: list[tuple[float, float]], start: float, end: float) -> None:
    assert spans[0][0] == pytest.approx(start, abs=1e-3)
    assert spans[-1][1] == pytest.approx(end, abs=1e-3)
    for (_, stop), (begin, _) in zip(spans, spans[1:], strict=False):
        assert begin == pytest.approx(stop, abs=1e-3)


def test_timecode_pads_hours():
    assert timecode(0) == "00:00:00"
    assert timecode(84.4) == "00:01:24"
    assert timecode(7017.6) == "01:56:58"


def test_film_shorter_than_max_becomes_one_scene():
    assert build_scenes([], 10.0, 200.0) == [(10.0, 200.0)]
    assert build_scenes([150.0], 0.0, 240.0) == [(0.0, 240.0)]


def test_no_cuts_falls_back_to_target_length():
    spans = build_scenes([], 0.0, 600.0, target_s=150.0, min_s=90.0, max_s=240.0)
    assert spans == [(0.0, 150.0), (150.0, 300.0), (300.0, 450.0), (450.0, 600.0)]


def test_boundaries_snap_to_nearest_cut():
    spans = build_scenes([152.0, 305.0], 0.0, 600.0)
    assert spans[0] == (0.0, 152.0)
    assert spans[1] == (152.0, 305.0)
    assert_contiguous(spans, 0.0, 600.0)


def test_every_scene_stays_within_min_and_max():
    cuts = [i * 7.3 for i in range(1, 200)]  # cut rapat, seperti film sungguhan
    spans = build_scenes(cuts, 0.0, 1234.5)
    assert_contiguous(spans, 0.0, 1234.5)
    assert len(spans) > 1
    for start, end in spans:
        assert end - start >= 90.0 - 1e-6
        assert end - start <= 240.0 + 1e-6


def test_short_tail_is_rebalanced_not_left_tiny():
    # Cut di 370 hanya menyisakan ekor 30 detik: harus digeser, bukan dibiarkan.
    spans = build_scenes([150.0, 370.0], 0.0, 400.0)
    assert_contiguous(spans, 0.0, 400.0)
    assert all(end - start >= 90.0 - 1e-6 for start, end in spans)
    assert all(end - start <= 240.0 + 1e-6 for start, end in spans)
    assert spans[-1][1] == pytest.approx(400.0)


def test_late_cut_is_passed_over_when_a_better_one_exists():
    # 370 menyisakan 30 detik, 300 menyisakan 100: pilih 300.
    spans = build_scenes([150.0, 300.0, 370.0], 0.0, 400.0)
    assert spans[-2][1] == pytest.approx(300.0)
    assert_contiguous(spans, 0.0, 400.0)


def test_cuts_outside_the_window_are_ignored():
    spans = build_scenes([-40.0, 150.0, 900.0], 0.0, 300.0)
    assert spans == [(0.0, 150.0), (150.0, 300.0)]


def test_empty_or_inverted_window_returns_nothing():
    assert build_scenes([], 100.0, 100.0) == []
    assert build_scenes([], 200.0, 100.0) == []


def test_annotation_window_trims_opening_and_credits():
    start, end = annotation_window(7402.0, 54.6, 7017.6)
    assert start == pytest.approx(34.6)
    assert end == pytest.approx(7037.6)


def test_annotation_window_keeps_full_film_without_subtitles():
    assert annotation_window(600.0, None, None) == (0.0, 600.0)


def test_annotation_window_never_extends_past_the_file():
    start, end = annotation_window(100.0, 90.0, 120.0)
    assert end == pytest.approx(100.0)
    assert start < end


def test_count_shots_counts_span_plus_one():
    assert count_shots([10.0, 20.0], 0.0, 30.0) == 3
    assert count_shots([10.0, 20.0], 10.0, 20.0) == 1


def test_dialogue_stats_clips_cues_to_scene():
    cues = [Cue(0.0, 5.0, "a"), Cue(8.0, 12.0, "b"), Cue(30.0, 40.0, "c")]
    n_cues, seconds = dialogue_stats(cues, 3.0, 20.0)
    assert n_cues == 2
    assert seconds == pytest.approx(2.0 + 4.0)
