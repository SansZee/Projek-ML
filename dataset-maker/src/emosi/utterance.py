"""Pembentukan ujaran dari cue subtitle dan timestamp kata Whisper.

Prinsipnya: **teks dari subtitle, waktu dari transkripsi.**

- Teks subtitle sudah bersih dan ejaannya benar; transkripsi Whisper dipakai
  hanya karena ia tahu kapan tiap kata diucapkan.
- Cue subtitle berdurasi ~2 detik dan sering memotong kalimat di tengah, jadi
  beberapa cue digabung sampai ujaran layak dilabeli (2-10 detik).
- Kata ASR di luar rentang cue tidak pernah dipakai. Ini yang membuang teks
  karangan Whisper pada musik tanpa perlu menyalakan VAD.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass

from .asr import WordIndex
from .subtitle import Cue

DASH = re.compile(r"^\s*[-–—]\s*|\s+[-–—]\s+")
# Deskripsi bunyi gaya SDH, mis. "(PEOPLE SCREAMING)" atau "[door slams]".
BRACKETED = re.compile(r"[\(\[\{][^\)\]\}]*[\)\]\}]")
STRAY_BRACKET = re.compile(r"[()\[\]{}]")
# Label pembicara yang tertinggal di tengah cue gabungan, mis. "MARY: Yeah."
SPEAKER_ANY = re.compile(r"(?<![A-Za-z])[A-Z][A-Z0-9 '.\-]{1,20}:\s*")


@dataclass(frozen=True)
class MergeSettings:
    merge_gap_s: float = 0.4
    short_gap_s: float = 1.0
    dur_min_s: float = 2.0
    dur_max_s: float = 10.0
    pad_head_s: float = 0.25
    pad_tail_s: float = 0.35
    asr_head_tol_s: float = 0.35
    asr_tail_tol_s: float = 0.0
    trim_overlap_max_s: float = 0.7
    terminal_punct: tuple[str, ...] = (".", "?", "!", "...", "…")


@dataclass
class Utterance:
    film_id: str
    utterance_id: str
    scene_id: str
    index: int
    start_s: float
    end_s: float
    duration_s: float
    text: str
    n_words: int
    speaker: str | None
    timing: str
    n_cues: int
    cue_start_s: float
    cue_end_s: float


def clean_cue_text(text: str) -> str:
    """Rapikan teks cue: buang penanda dialog, deskripsi bunyi, dan label.

    Deskripsi bunyi seperti "(PEOPLE SCREAMING)" bukan ucapan; kalau ikut masuk
    teks, model teks belajar memetakan tulisan itu ke emosi.
    """
    text = BRACKETED.sub(" ", text)
    text = STRAY_BRACKET.sub(" ", text)
    text = DASH.sub(" ", text)
    text = SPEAKER_ANY.sub("", text)
    return " ".join(text.split())


def ends_sentence(text: str, punct: tuple[str, ...]) -> bool:
    return text.rstrip().endswith(tuple(punct))


def merge_cues(cues: list[Cue], settings: MergeSettings) -> list[list[Cue]]:
    """Gabungkan cue berurutan menjadi ujaran 2-10 detik.

    Cue digabung kalau jaraknya dekat **dan** salah satu dari ini benar:
    ujaran masih terlalu pendek, atau kalimatnya belum selesai. Kalimat yang
    sudah lengkap dibiarkan berdiri sendiri walaupun pendek, selama masih di
    atas ``dur_min_s``.
    """
    groups: list[list[Cue]] = []
    current: list[Cue] = []

    for cue in cues:
        if not cue.text.strip():
            continue
        if not current:
            current = [cue]
            continue

        previous = current[-1]
        span = previous.end - current[0].start
        gap = cue.start - previous.end
        too_short = span < settings.dur_min_s
        allowed_gap = settings.short_gap_s if too_short else settings.merge_gap_s
        same_speaker = not (
            previous.speaker and cue.speaker and previous.speaker != cue.speaker
        )

        join = (
            same_speaker
            and gap <= allowed_gap
            and cue.end - current[0].start <= settings.dur_max_s
            and (too_short or not ends_sentence(previous.text, settings.terminal_punct))
        )
        if join:
            current.append(cue)
        else:
            groups.append(current)
            current = [cue]

    if current:
        groups.append(current)
    return groups


def _nearest(values: list[float], target: float, window_s: float) -> float | None:
    lo = bisect_left(values, target - window_s)
    hi = bisect_left(values, target + window_s)
    if hi <= lo:
        return None
    return min(values[lo:hi], key=lambda value: abs(value - target))


def span_bounds(
    index: WordIndex,
    start_s: float,
    end_s: float,
    *,
    head_tol_s: float = 0.35,
    tail_tol_s: float = 0.0,
) -> tuple[float, float] | None:
    """Awal kata pertama dan akhir kata terakhir yang mulai di dalam cue.

    Toleransi hanya diberikan di kepala, karena subtitle mulai sedikit lebih
    awal daripada ucapan (±0,06 s terukur). Di ekor tidak ada toleransi: kata
    yang mulai setelah cue berakhir hampir selalu milik kalimat berikutnya.
    Mengukur "kata terdekat ke akhir cue" justru menyeret kata itu masuk —
    terukur pada About Time, 53% ujaran berakhir di kata milik baris
    berikutnya.

    ``None`` berarti tidak ada kata yang bisa dipakai; pemanggil mundur ke
    waktu cue.
    """
    if not index.starts or not index.ends:
        return None
    lo = bisect_left(index.starts, start_s - head_tol_s)
    hi = bisect_left(index.starts, end_s + tail_tol_s)
    if hi <= lo:
        return None
    first = index.starts[lo]
    last = max(index.ends[lo:hi])
    if last - first < 0.1:
        return None
    return first, last


def _unique_speaker(cues: list[Cue]) -> str | None:
    names = {cue.speaker for cue in cues if cue.speaker}
    return names.pop() if len(names) == 1 else None


def build_utterances(
    film_id: str,
    cues: list[Cue],
    scenes: list[tuple[str, float, float]],
    index: WordIndex,
    settings: MergeSettings,
) -> list[Utterance]:
    """Susun ujaran per adegan.

    ``scenes`` berisi ``(scene_id, start_s, end_s)``. Cue masuk ke adegan yang
    memuat waktu mulainya; sisa cue yang melewati batas adegan dipotong karena
    batas adegan selalu jatuh di cut gambar, tempat dialog jarang berlangsung.
    """
    utterances: list[Utterance] = []
    cursor = 0

    for scene_id, scene_start, scene_end in scenes:
        selected: list[Cue] = []
        while cursor < len(cues) and cues[cursor].start < scene_start:
            cursor += 1
        probe = cursor
        while probe < len(cues) and cues[probe].start < scene_end:
            selected.append(cues[probe])
            probe += 1
        cursor = probe
        if not selected:
            continue

        for group in merge_cues(selected, settings):
            cue_start = group[0].start
            cue_end = group[-1].end
            text = " ".join(
                part for cue in group if (part := clean_cue_text(cue.text))
            ).strip()
            if not text:
                continue

            words = span_bounds(
                index,
                cue_start,
                cue_end,
                head_tol_s=settings.asr_head_tol_s,
                tail_tol_s=settings.asr_tail_tol_s,
            )
            # Kalau Whisper hanya menangkap sepotong kecil dari cue (musik,
            # tumpang-tindih suara), waktu kata tidak bisa dipercaya: klip
            # 0,4 detik untuk kalimat 4 detik. Mundur ke waktu cue.
            if words is not None:
                cue_span = cue_end - cue_start
                word_span = words[1] - words[0]
                if word_span < max(0.5, 0.25 * cue_span):
                    words = None
            if words is None:
                start = cue_start
                end = cue_end
                timing = "cue"
            else:
                start = words[0] - settings.pad_head_s
                end = words[1] + settings.pad_tail_s
                timing = "asr"
            if end - start > settings.dur_max_s:
                end = start + settings.dur_max_s
            if end - start <= 0:
                start, end, timing = cue_start, cue_end, "cue"

            index_number = len(utterances) + 1
            item = Utterance(
                film_id=film_id,
                utterance_id=f"{film_id}-U{index_number:05d}",
                scene_id=scene_id,
                index=index_number,
                start_s=round(start, 3),
                end_s=round(end, 3),
                duration_s=round(end - start, 3),
                text=text,
                n_words=len(text.split()),
                speaker=_unique_speaker(group),
                timing=timing,
                n_cues=len(group),
                cue_start_s=round(cue_start, 3),
                cue_end_s=round(cue_end, 3),
            )

            # Padding membuat klip berdekatan saling menimpa. Ujung klip
            # sebelumnya dipotong tepat di awal klip berikutnya: awal yang
            # hilang hanya padding atau ekor kata yang ditarik Whisper,
            # sedangkan awal ujaran berikutnya tidak pernah digeser.
            previous = utterances[-1] if utterances else None
            if previous is not None and previous.scene_id == scene_id:
                overlap = previous.end_s - item.start_s
                if 0 < overlap <= settings.trim_overlap_max_s:
                    previous.end_s = item.start_s
                    previous.duration_s = round(previous.end_s - previous.start_s, 3)

            utterances.append(item)
    return utterances
