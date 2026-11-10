from __future__ import annotations

from emosi.subtitle import clean_text, parse_srt_text, subtitle_stats

SAMPLE = """1
00:00:01,000 --> 00:00:03,500
TIM: I always knew we were a fairly odd family.

2
00:00:03,500 --> 00:00:05,000
<i>Too tall. Too skinny.</i>

3
00:00:06,000 --> 00:00:07,000
[MUSIC]

4
00:00:08,000 --> 00:00:10,000
Adapted by: Some Guy
"""


def test_parse_counts_only_dialogue_cues():
    cues, errors = parse_srt_text(SAMPLE)
    assert errors == 0
    assert len(cues) == 4
    assert cues[0].text == "I always knew we were a fairly odd family."
    assert cues[1].text == "Too tall. Too skinny."


def test_stats_detects_speaker_label_and_watermark():
    cues, errors = parse_srt_text(SAMPLE)
    stats = subtitle_stats(cues, film_duration_s=600.0, errors=errors)
    assert stats.n_cues == 4
    assert stats.n_speaker_labeled == 1
    assert stats.n_watermark == 1
    assert stats.first_s == 1.0
    assert stats.last_s == 10.0


def test_wpm_uses_span_between_first_and_last_cue():
    cues, _ = parse_srt_text(SAMPLE)
    stats = subtitle_stats(cues, film_duration_s=600.0)
    words = sum(len(c.text.split()) for c in cues)
    assert words > 0
    expected = words / (stats.last_s - stats.first_s) * 60
    assert abs(stats.wpm - expected) < 0.01


def test_clean_text_removes_non_dialog_lines():
    assert clean_text("[MUSIC]\nHello there.") == "Hello there."
    assert clean_text("(laughs)") == ""


def test_tail_to_end_reports_credits_gap():
    cues, _ = parse_srt_text(SAMPLE)
    stats = subtitle_stats(cues, film_duration_s=400.0)
    assert stats.tail_to_end_s == 390.0
