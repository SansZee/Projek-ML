from __future__ import annotations

from pathlib import Path

import pytest

from emosi import cut

SETTINGS = cut.CutSettings(height=720, crf=28, audio_sr=16000, face_size=224)


def test_clip_command_keeps_audio_and_limits_height(tmp_path: Path):
    command = cut.clip_command(
        Path("/film/FDS09.mkv"), tmp_path / "C1.mp4", 12.5, 16.0, SETTINGS
    )
    joined = " ".join(command)
    assert command[0] == "ffmpeg"
    assert "-ss 12.500" in joined
    assert "-t 3.500" in joined
    assert "scale=-2:720" in joined
    assert "-c:a aac" in joined
    assert "-map 0:a:0?" in joined
    assert command[-1].endswith("C1.mp4")


def test_audio_command_is_mono_16k_wav(tmp_path: Path):
    command = cut.audio_command(
        Path("/film/FDS09.mkv"), tmp_path / "C1.wav", 0.0, 4.0, SETTINGS
    )
    joined = " ".join(command)
    assert "-vn" in joined
    assert "-ac 1" in joined
    assert "-ar 16000" in joined
    assert "pcm_s16le" in joined


def test_face_command_reads_raw_frames(tmp_path: Path):
    command = cut.face_command(tmp_path / "C1.mp4", fps=5.0, settings=SETTINGS)
    joined = " ".join(command)
    assert "-f rawvideo" in joined
    assert "-s 224x224" in joined
    assert "-r 5.000" in joined
    assert command[-1].endswith("C1.mp4")


def test_face_box_is_square_and_inside_the_frame():
    left, top, right, bottom = cut.face_box(400, 200, 500, 340, 1920, 800)
    assert right - left == bottom - top
    assert 0 <= left < right <= 1920
    assert 0 <= top < bottom <= 800
    # Wajah 100×140 dengan margin 1.35 → sisi 189.
    assert right - left == pytest.approx(189, abs=1)


def test_face_box_clamps_at_the_edge():
    left, top, right, bottom = cut.face_box(0, 0, 80, 80, 320, 240)
    assert (left, top) == (0, 0)
    assert right <= 320 and bottom <= 240


def test_center_box_covers_the_middle():
    left, top, right, bottom = cut.center_box(1920, 800)
    assert left == pytest.approx(720, abs=1)
    assert right - left == bottom - top


def test_smooth_box_moves_towards_the_new_box():
    blended = cut.smooth_box((0.0, 0.0, 10.0, 10.0), (10.0, 10.0, 20.0, 20.0), alpha=0.5)
    assert blended == pytest.approx((5.0, 5.0, 15.0, 15.0))


def test_smooth_box_without_history_returns_current():
    current = (1.0, 2.0, 3.0, 4.0)
    assert cut.smooth_box(None, current) == current


def test_duration_has_a_floor():
    assert cut.duration(10.0, 10.0) == pytest.approx(0.1)
