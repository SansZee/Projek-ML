"""Pemotongan klip, audio, dan wajah (``10_cut.py``).

Tahap ini satu-satunya yang memindahkan media. Untuk setiap klip terpilih:

  clips/<clip_id>.mp4   720p, audio ikut, untuk model video
  audio/<clip_id>.wav   16 kHz mono, untuk model suara
  faces/<clip_id>.mp4   wajah dominan 224×224, untuk model wajah

Ukuran berkasnya dijaga supaya 3.000 klip tetap 3–6 GB: video 720p CRF 28,
audio PCM 16 kHz, wajah 224×224.

Bagian murni (perintah ffmpeg dan kotak wajah) dipisahkan supaya bisa diuji
tanpa GPU; pemotongan wajah sendiri memakai model dari tahap 04.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CutSettings:
    height: int = 720
    crf: int = 28
    audio_sr: int = 16000
    face_size: int = 224
    face_crf: int = 23
    face_margin: float = 1.35
    face_alpha: float = 0.65
    preset: str = "veryfast"


def duration(start_s: float, end_s: float) -> float:
    return max(0.1, float(end_s) - float(start_s))


def clip_command(
    source: Path, dest: Path, start_s: float, end_s: float, settings: CutSettings
) -> list[str]:
    """Video 720p + audio, cukup untuk model video dan sinkronisasi."""
    return [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-ss",
        f"{max(0.0, float(start_s)):.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration(start_s, end_s):.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        f"scale=-2:{int(settings.height)}",
        "-c:v",
        "libx264",
        "-preset",
        settings.preset,
        "-crf",
        str(int(settings.crf)),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ac",
        "1",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        str(dest),
    ]


def audio_command(
    source: Path, dest: Path, start_s: float, end_s: float, settings: CutSettings
) -> list[str]:
    """WAV mono 16 kHz; model suara hampir selalu memakai laju ini."""
    return [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-ss",
        f"{max(0.0, float(start_s)):.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration(start_s, end_s):.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(settings.audio_sr)),
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]


def face_command(dest: Path, *, fps: float, settings: CutSettings) -> list[str]:
    """ffmpeg yang menerima frame wajah 224×224 dari pipe."""
    size = int(settings.face_size)
    return [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{size}x{size}",
        "-r",
        f"{float(fps):.3f}",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        settings.preset,
        "-crf",
        str(int(settings.face_crf)),
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(dest),
    ]


def face_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    *,
    margin: float = 1.35,
) -> tuple[int, int, int, int]:
    """Kotak persegi di sekitar wajah, dengan margin, di dalam frame."""
    width = max(2, int(width))
    height = max(2, int(height))
    cx = (float(x1) + float(x2)) / 2.0
    cy = (float(y1) + float(y2)) / 2.0
    side = max(float(x2) - float(x1), float(y2) - float(y1)) * float(margin)
    side = max(8.0, min(side, float(min(width, height))))

    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))
    left = max(0, min(left, width - int(side)))
    top = max(0, min(top, height - int(side)))
    right = min(width, left + int(side))
    bottom = min(height, top + int(side))
    return left, top, right, bottom


def center_box(width: int, height: int, *, fraction: float = 0.6) -> tuple[int, int, int, int]:
    side = max(8.0, min(int(width), int(height)) * float(fraction))
    left = int((width - side) / 2)
    top = int((height - side) / 2)
    return left, top, left + int(side), top + int(side)


def smooth_box(
    previous: tuple[float, float, float, float] | None,
    current: tuple[float, float, float, float],
    *,
    alpha: float = 0.65,
) -> tuple[float, float, float, float]:
    """Redam goyangan kotak antar frame; ``alpha`` bagian untuk kotak lama."""
    if previous is None:
        return current
    weight = min(1.0, max(0.0, float(alpha)))
    return tuple(
        weight * float(old) + (1.0 - weight) * float(new)
        for old, new in zip(previous, current, strict=True)
    )


def run_ffmpeg(command: list[str]) -> None:
    proc = subprocess.run(command, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace")[-600:]
        raise RuntimeError(f"perintah gagal ({proc.returncode}): {' '.join(command)}\n{tail}")
