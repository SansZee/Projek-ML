"""Uji bagian asr yang tidak membutuhkan GPU atau faster-whisper.

Pemeriksaan sinkronisasi adalah satu-satunya bagian 01_asr.py yang angkanya
masuk laporan, jadi rumusnya harus diuji tanpa memuat model 3 GB.
"""

from __future__ import annotations

import pytest

from emosi.asr import (
    CheckSettings,
    WordIndex,
    block_error,
    build_blocks,
    check_sync,
    coverage,
    edit_distance,
    median_onset_offset,
    normalize_words,
    word_error_rate,
)
from emosi.subtitle import Cue


def make_words(n: int = 140, start: float = 10.0, step: float = 1.0) -> list[dict]:
    words = []
    for i in range(n):
        begin = start + i * step
        words.append({"word": f" w{int(begin)}", "start": begin, "end": begin + 0.4})
    return words


def make_cues(
    n: int = 12, shift: float = 0.0, start: float = 10.0, block: float = 5.0
) -> list[Cue]:
    """Cue 4,8 detik berisi 5 kata, berjarak 0,2 detik — menyerupai subtitle asli."""
    cues = []
    for i in range(n):
        cue_start = start + i * block + shift
        words = [f"w{int(start + i * block + k)}" for k in range(5)]
        cues.append(Cue(start=cue_start, end=cue_start + 4.8, text=" ".join(words)))
    return cues


def payload(words: list[dict]) -> dict:
    return {"segments": [{"words": words}]}


def settings(pad_s: float = 0.0) -> CheckSettings:
    return CheckSettings(
        block_s=30.0, max_blocks=12, onset_window_s=0.75, pad_s=pad_s
    )


def make_short_cues(count: int = 20, shift: float = 0.0) -> tuple[list[Cue], list[dict]]:
    """Cue pendek berisi tiga kata, jarak antarcue lebar: geseran jadi mudah diukur."""
    cues = []
    words = []
    for i in range(count):
        base = 10.0 + i * 8.0 + shift
        cues.append(Cue(start=base, end=base + 2.0, text=f"a{i} b{i} c{i}"))
        for k in range(3):
            start = base + k * 0.6
            words.append({"word": f" w{i}{k}", "start": start, "end": start + 0.3})
    return cues, words


def shift_cues(cues: list[Cue], delta: float) -> list[Cue]:
    """Geser waktu cue saja, tanpa menggeser kata — meniru subtitle yang tidak sinkron."""
    return [Cue(start=cue.start + delta, end=cue.end + delta, text=cue.text) for cue in cues]


def test_normalize_strips_punctuation_and_case():
    assert normalize_words("Don't GO there!") == ["don't", "go", "there"]


def test_wer_counts_substitution_insertion_deletion():
    assert edit_distance(["a", "b", "c"], ["a", "b", "d"]) == 1
    assert edit_distance(["a", "b"], ["a", "b", "c"]) == 1
    assert edit_distance(["a", "b", "c"], ["a", "c"]) == 1
    assert word_error_rate(["a", "b", "c"], ["a", "b", "d"]) == pytest.approx(1 / 3)
    assert word_error_rate(["a", "b"], ["a", "b", "c"]) == pytest.approx(0.5)


def test_word_index_returns_words_in_window_only():
    index = WordIndex.from_segments(payload(make_words(5))["segments"])
    assert index.between(11.0, 13.0) == [" w11", " w12"]


def test_blocks_group_consecutive_cues_and_stay_bounded():
    blocks = build_blocks(make_cues(40), block_s=30.0, max_blocks=6)
    assert 1 < len(blocks) <= 6
    assert all(block for block in blocks)
    # Blok harus berurutan waktu dan tidak tumpang tindih.
    starts = [block[0].start for block in blocks]
    assert starts == sorted(starts)


def test_aligned_subtitles_score_zero_correctly():
    cues = make_cues()
    index = WordIndex.from_segments(payload(make_words())["segments"])
    blocks = build_blocks(cues, 30.0, 12)
    wer, edits, total = block_error(blocks, index, pad_s=0.0)
    assert (wer, edits) == (0.0, 0)
    assert total == sum(len(cue.text.split()) for block in blocks for cue in block)


def test_aligned_subtitles_verdict_selaras():
    row = check_sync("TST00", make_cues(), payload(make_words()), settings())
    assert row["verdict"] == "selaras"
    assert row["wer"] == 0.0
    assert row["offset"] == 0.0


def test_onset_offset_is_zero_for_aligned_subtitles():
    cues, words = make_short_cues()
    offset, measured = median_onset_offset(
        cues, WordIndex.from_segments(payload(words)["segments"])
    )
    assert offset == pytest.approx(0.0, abs=0.05)
    assert measured == len(cues)


def test_onset_offset_detects_late_subtitles():
    cues, words = make_short_cues()
    index = WordIndex.from_segments(payload(words)["segments"])
    offset, _measured = median_onset_offset(shift_cues(cues, 0.2), index)
    assert offset == pytest.approx(0.2, abs=0.05)

    early, _measured = median_onset_offset(shift_cues(cues, -0.2), index)
    assert early == pytest.approx(-0.2, abs=0.05)


def test_onset_offset_returns_none_when_no_word_is_near():
    cues, words = make_short_cues()
    offset, measured = median_onset_offset(
        shift_cues(cues, 3.0), WordIndex.from_segments(payload(words)["segments"])
    )
    assert offset is None
    assert measured == 0


def test_shifted_subtitles_are_flagged_and_still_aligned_in_content():
    cues, words = make_short_cues()
    row = check_sync("TST00", shift_cues(cues, 2.0), payload(words), settings())
    assert row["offset"] is None
    assert "geseran tidak bisa diukur" in row["notes"]
    # Isi tetap cocok setelah digeser: yang bermasalah hanya waktunya.
    assert row["wer_at_zero"] > 0.3


def test_small_drift_is_not_flagged():
    cues, words = make_short_cues(count=90)
    for position, cue in enumerate(cues):
        shift = 0.05 if position < 30 else (0.1 if position < 60 else 0.15)
        cue.start += shift
        cue.end += shift
    row = check_sync("TST00", cues, payload(words), settings())
    assert row["drift_s"] is not None and row["drift_s"] < 0.2
    assert "geseran berubah" not in row["notes"]


def test_mismatched_text_is_rejected():
    cues = make_cues()
    for cue in cues:
        cue.text = "kata yang sama sekali berbeda tanpa hubungan"
    row = check_sync("TST00", cues, payload(make_words()), settings())
    assert row["verdict"] == "tidak_cocok"


def test_coverage_flags_missing_subtitles():
    cues = make_cues(n=2)
    assert coverage(cues, WordIndex.from_segments(payload(make_words())["segments"])) < 0.3


def test_empty_transcript_does_not_crash():
    row = check_sync("TST00", make_cues(), {"segments": []}, settings())
    assert row["verdict"] == "tidak_ada_data"
