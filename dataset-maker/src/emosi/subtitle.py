"""Pembacaan subtitle dan statistik dialog.

Parser sengaja ditulis sendiri (bukan hanya pysubs2) supaya berkas SRT yang
cacat dari internet tetap bisa diukur tanpa melempar exception.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

TIMESTAMP = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)
TAG = re.compile(r"<[^>]+>|\{\\[^}]*\}")
NON_DIALOG = re.compile(r"^\s*[\[\(\{][^\]\)\}]*[\]\)\}]\s*$")
SPEAKER_LABEL = re.compile(r"^([A-Z][A-Z0-9 '.\-]{1,20}):\s*")
WATERMARK = re.compile(
    r"(subtitle|subs?\s+by|synced?\s+by|adapted\s+by|opensubtitles|yify|yts|addic7ed)",
    re.IGNORECASE,
)


@dataclass
class Cue:
    start: float
    end: float
    text: str
    speaker: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class SubtitleStats:
    n_cues: int = 0
    n_words: int = 0
    first_s: float = 0.0
    last_s: float = 0.0
    wpm: float = 0.0
    n_overlap: int = 0
    n_short: int = 0
    n_speaker_labeled: int = 0
    n_watermark: int = 0
    median_cue_s: float = 0.0
    max_gap_s: float = 0.0
    tail_to_end_s: float = 0.0
    parse_errors: int = 0
    warnings: list[str] = field(default_factory=list)


def _to_seconds(h: str, m: str, s: str, frac: str) -> float:
    # "5" berarti 500 ms, bukan 5 ms.
    millis = int(frac.ljust(3, "0")[:3])
    return int(h) * 3600 + int(m) * 60 + int(s) + millis / 1000.0


def clean_text(raw: str) -> str:
    """Buang tag HTML/ASS dan baris non-dialog, pertahankan label pembicara."""
    lines: list[str] = []
    for line in raw.splitlines():
        line = TAG.sub("", line).strip()
        if not line or NON_DIALOG.match(line):
            continue
        lines.append(line)
    return " ".join(lines).strip()


def split_speaker(text: str) -> tuple[str | None, str]:
    """Pisahkan label pembicara gaya SDH, mis. ``DUKE: Good night.``

    Subtitle SDH kadang memberi label pembicara. Informasi ini berguna untuk
    memvalidasi hasil diarisasi, jadi disimpan, bukan dibuang.
    """
    stripped = text.strip()
    match = SPEAKER_LABEL.match(stripped)
    if not match:
        return None, stripped
    return match.group(1).strip(), stripped[match.end() :].lstrip()


def parse_srt_text(text: str) -> tuple[list[Cue], int]:
    cues: list[Cue] = []
    errors = 0
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        if not block.strip():
            continue
        match = TIMESTAMP.search(block)
        if not match:
            errors += 1
            continue
        groups = match.groups()
        body = block[match.end() :].strip()
        speaker, text = split_speaker(clean_text(body))
        cues.append(
            Cue(
                start=_to_seconds(*groups[:4]),
                end=_to_seconds(*groups[4:]),
                text=text,
                speaker=speaker,
            )
        )
    cues.sort(key=lambda c: c.start)
    return cues, errors


def parse_srt(path: Path) -> tuple[list[Cue], int]:
    return parse_srt_text(path.read_text(encoding="utf-8", errors="replace"))


def subtitle_stats(cues: list[Cue], film_duration_s: float, errors: int = 0) -> SubtitleStats:
    stats = SubtitleStats(parse_errors=errors)
    if not cues:
        stats.warnings.append("tidak ada cue yang bisa dibaca")
        return stats

    stats.n_cues = len(cues)
    stats.first_s = round(cues[0].start, 3)
    stats.last_s = round(cues[-1].end, 3)
    stats.n_words = sum(len(c.text.split()) for c in cues)

    span = stats.last_s - stats.first_s
    stats.wpm = round(stats.n_words / span * 60, 2) if span > 0 else 0.0

    durations = sorted(c.duration for c in cues)
    stats.median_cue_s = round(durations[len(durations) // 2], 3)
    stats.n_short = sum(1 for c in cues if c.duration < 0.8)
    stats.n_speaker_labeled = sum(1 for c in cues if c.speaker)
    stats.n_watermark = sum(1 for c in cues if WATERMARK.search(c.text))

    stats.n_overlap = sum(1 for i in range(len(cues) - 1) if cues[i + 1].start < cues[i].end)
    gaps = [cues[i + 1].start - cues[i].end for i in range(len(cues) - 1)]
    stats.max_gap_s = round(max(gaps), 3) if gaps else 0.0

    if film_duration_s > 0:
        stats.tail_to_end_s = round(film_duration_s - stats.last_s, 3)

    if stats.n_watermark:
        stats.warnings.append(f"{stats.n_watermark} cue berisi watermark penerjemah")
    if stats.parse_errors:
        stats.warnings.append(f"{stats.parse_errors} blok tanpa timestamp valid")
    return stats
