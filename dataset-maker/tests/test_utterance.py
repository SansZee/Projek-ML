from __future__ import annotations

import pytest

from emosi.asr import WordIndex
from emosi.subtitle import Cue
from emosi.utterance import (
    MergeSettings,
    build_utterances,
    clean_cue_text,
    ends_sentence,
    merge_cues,
    span_bounds,
)


def settings(**overrides) -> MergeSettings:
    values = {
        "merge_gap_s": 0.4,
        "short_gap_s": 1.0,
        "dur_min_s": 2.0,
        "dur_max_s": 10.0,
        "pad_head_s": 0.25,
        "pad_tail_s": 0.35,
        "asr_head_tol_s": 0.35,
        "asr_tail_tol_s": 0.0,
        "terminal_punct": (".", "?", "!", "…"),
    }
    values.update(overrides)
    return MergeSettings(**values)


def make_index(spans: list[tuple[float, float]]) -> WordIndex:
    segments = [
        {"words": [{"word": " w", "start": start, "end": end}]} for start, end in spans
    ]
    return WordIndex.from_segments(segments)


def cue(start: float, end: float, text: str, speaker: str | None = None) -> Cue:
    return Cue(start=start, end=end, text=text, speaker=speaker)


def test_ends_sentence_handles_ellipsis():
    assert ends_sentence("Hello.", (".", "?", "!"))
    assert ends_sentence("Kenapa?", (".", "?"))
    assert not ends_sentence("Hello there", (".", "?"))
    assert ends_sentence("Wait…", (".", "…"))


def test_clean_cue_text_removes_dialogue_dashes():
    assert clean_cue_text("- Hello there. - Hi.") == "Hello there. Hi."
    assert clean_cue_text("  two   spaces ") == "two spaces"


def test_clean_cue_text_drops_sdh_descriptions_and_labels():
    assert clean_cue_text("(SIGHS) What have you done?") == "What have you done?"
    assert clean_cue_text("Well... [door slams] okay.") == "Well... okay."
    assert clean_cue_text("MARY: Yeah. She is.") == "Yeah. She is."
    assert clean_cue_text("(POP MUSIC CONTINUES PLAYING)") == ""


def test_merges_incomplete_sentence_even_across_a_pause():
    cues = [cue(0.0, 1.5, "Think of the moment"), cue(2.2, 3.5, "and you are there.")]
    groups = merge_cues(cues, settings())
    assert len(groups) == 1
    assert [item.text for item in groups[0]] == ["Think of the moment", "and you are there."]


def test_keeps_complete_sentences_separate():
    cues = [cue(0.0, 3.0, "I love you."), cue(3.2, 6.0, "I know.")]
    groups = merge_cues(cues, settings())
    assert len(groups) == 2


def test_short_cue_is_rescued_with_a_wider_gap():
    cues = [cue(0.0, 1.0, "Hi."), cue(1.8, 3.5, "You came.")]
    assert len(merge_cues(cues, settings())) == 1
    assert len(merge_cues(cues, settings(short_gap_s=0.5))) == 2


def test_never_exceeds_max_duration():
    cues = [cue(0.0, 8.0, "A very long line without punctuation"), cue(8.2, 10.5, "more")]
    assert len(merge_cues(cues, settings(dur_max_s=10.0))) == 2


def test_speaker_change_blocks_merge():
    cues = [cue(0.0, 1.0, "Hi.", "ANA"), cue(1.1, 2.5, "Hey.", "BUDI")]
    assert len(merge_cues(cues, settings())) == 2


def test_empty_cues_are_dropped():
    assert merge_cues([cue(0.0, 1.0, "  "), cue(1.0, 2.0, "")], settings()) == []


def test_span_bounds_uses_first_and_last_word_inside_the_cue():
    index = make_index([(10.05, 10.4), (11.0, 11.4), (12.0, 12.35)])
    assert span_bounds(index, 10.0, 12.4) == pytest.approx((10.05, 12.35))


def test_span_bounds_ignores_words_starting_after_the_cue():
    # Kata di 12,2 milik kalimat berikutnya, bukan ujaran ini.
    index = make_index([(10.05, 10.4), (12.2, 12.6)])
    assert span_bounds(index, 10.0, 12.0) == pytest.approx((10.05, 10.4))


def test_span_bounds_returns_none_when_words_are_far():
    index = make_index([(30.0, 30.4)])
    assert span_bounds(index, 10.0, 12.0) is None


def test_build_utterances_assigns_scenes_and_pads_from_words():
    cues = [cue(1.0, 3.0, "Hello."), cue(11.0, 13.0, "Later.")]
    index = make_index([(1.05, 2.9), (11.05, 12.9)])
    scenes = [("F-S001", 0.0, 10.0), ("F-S002", 10.0, 20.0)]

    items = build_utterances("F", cues, scenes, index, settings())
    assert [item.utterance_id for item in items] == ["F-U00001", "F-U00002"]
    assert [item.scene_id for item in items] == ["F-S001", "F-S002"]
    assert items[0].timing == "asr"
    assert items[0].start_s == pytest.approx(0.8)
    assert items[0].end_s == pytest.approx(3.25)


def test_cue_crossing_a_scene_boundary_keeps_its_audio():
    cues = [cue(9.0, 12.0, "Crosses the cut.")]
    index = make_index([(9.05, 11.9)])
    scenes = [("F-S001", 0.0, 10.0), ("F-S002", 10.0, 20.0)]
    items = build_utterances("F", cues, scenes, index, settings())
    assert len(items) == 1
    assert items[0].scene_id == "F-S001"
    assert items[0].end_s == pytest.approx(12.25)


def test_sparse_asr_words_fall_back_to_cue_times():
    cues = [cue(1.0, 5.0, "A full line of dialogue.")]
    index = make_index([(4.6, 4.8)])
    items = build_utterances("F", cues, [("F-S001", 0.0, 10.0)], index, settings())
    assert items[0].timing == "cue"
    assert (items[0].start_s, items[0].end_s) == (1.0, 5.0)


def test_build_utterances_falls_back_to_cue_times():
    cues = [cue(5.0, 8.0, "No words here.")]
    items = build_utterances("F", cues, [("F-S001", 0.0, 10.0)], make_index([]), settings())
    assert items[0].timing == "cue"
    assert (items[0].start_s, items[0].end_s) == (5.0, 8.0)


def test_build_utterances_takes_shared_speaker_only():
    cues = [
        cue(1.0, 2.5, "Yes.", "ANA"),
        cue(2.7, 4.5, "Of course.", "ANA"),
        cue(4.7, 6.0, "No.", "BUDI"),
    ]
    index = make_index([(1.05, 2.4), (2.75, 4.4), (4.75, 5.9)])
    items = build_utterances("F", cues, [("F-S001", 0.0, 10.0)], index, settings())
    assert items[0].speaker == "ANA"
    assert items[0].text == "Yes. Of course."
    assert items[1].speaker == "BUDI"


def test_cues_before_and_between_scenes_are_skipped():
    cues = [cue(1.0, 2.0, "Before."), cue(50.0, 52.0, "Inside.")]
    index = make_index([(1.0, 2.0), (50.0, 52.0)])
    items = build_utterances("F", cues, [("F-S001", 40.0, 60.0)], index, settings())
    assert len(items) == 1
    assert items[0].text == "Inside."


def test_adjacent_utterances_do_not_overlap():
    # Padding 0,25 + 0,35 detik akan menimpa kalau ujung klip tidak dipotong.
    cues = [cue(1.0, 3.0, "First one."), cue(3.3, 5.5, "Second one.")]
    index = make_index([(1.05, 2.9), (3.35, 5.4)])
    items = build_utterances("F", cues, [("F-S001", 0.0, 10.0)], index, settings())
    assert len(items) == 2
    assert items[0].end_s == pytest.approx(items[1].start_s)
    assert items[0].duration_s == pytest.approx(items[0].end_s - items[0].start_s, abs=1e-3)


def test_simultaneous_speech_keeps_its_audio():
    # Dua baris tumpang tindih jauh: memotong ekor akan menghilangkan ucapan.
    cues = [cue(1.0, 5.0, "A long first line."), cue(1.2, 3.0, "Short overlap.")]
    index = make_index([(1.05, 4.9), (1.25, 2.9)])
    items = build_utterances("F", cues, [("F-S001", 0.0, 10.0)], index, settings())
    assert len(items) == 2
    # Keduanya tetap panjang; ekor klip pertama tidak dipotong sampai habis.
    assert items[0].duration_s > 2.0
    assert items[1].duration_s > 1.0
    assert items[0].end_s > items[1].start_s
