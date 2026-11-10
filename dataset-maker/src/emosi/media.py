"""Pembungkus ffprobe/ffmpeg: metadata, frame, dan audio."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

VIDEO_EXT = {".mkv", ".mp4", ".avi", ".m4v", ".mov", ".wmv", ".ts"}
SUB_EXT = {".srt", ".ass", ".ssa", ".vtt"}
TEXT_SUB_CODECS = {"subrip", "ass", "ssa", "mov_text", "webvtt", "text"}


class ToolMissing(RuntimeError):
    """Biner eksternal tidak tersedia."""


def require_tools(*names: str) -> None:
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise ToolMissing(
            "biner tidak ditemukan di PATH: " + ", ".join(missing) + ". Pasang ffmpeg dulu."
        )


def run_ffmpeg(cmd: list[str], *, binary: bool = False) -> bytes | str:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace")[-800:]
        raise RuntimeError(f"perintah gagal ({proc.returncode}): {' '.join(cmd)}\n{tail}")
    return proc.stdout if binary else proc.stdout.decode("utf-8", "replace")


def ffprobe_json(path: Path) -> dict:
    out = run_ffmpeg(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    return json.loads(str(out))


def parse_fps(rate: str | None) -> float:
    if not rate or "/" not in rate:
        return 0.0
    try:
        num, den = rate.split("/", 1)
        return float(num) / float(den) if float(den) else 0.0
    except ValueError:
        return 0.0


@dataclass
class MediaInfo:
    duration_s: float
    fps: float
    width: int
    height: int
    vcodec: str
    acodec: str
    audio_channels: int
    n_audio: int
    n_sub: int
    n_sub_text: int
    n_sub_bitmap: int
    sub_langs: str

    def as_dict(self) -> dict:
        return asdict(self)


def probe_media(path: Path) -> MediaInfo:
    data = ffprobe_json(path)
    fmt = data.get("format", {}) or {}
    streams = data.get("streams", []) or []

    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    sub_streams = [s for s in streams if s.get("codec_type") == "subtitle"]
    audio = audio_streams[0] if audio_streams else {}

    text_subs = [s for s in sub_streams if s.get("codec_name") in TEXT_SUB_CODECS]
    bitmap_subs = [s for s in sub_streams if s.get("codec_name") not in TEXT_SUB_CODECS]
    langs = sorted({(s.get("tags") or {}).get("language", "?") for s in sub_streams})

    duration = float(fmt.get("duration") or video.get("duration") or 0.0)

    return MediaInfo(
        duration_s=round(duration, 3),
        fps=round(parse_fps(video.get("r_frame_rate")), 4),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        vcodec=str(video.get("codec_name") or ""),
        acodec=str(audio.get("codec_name") or ""),
        audio_channels=int(audio.get("channels") or 0),
        n_audio=len(audio_streams),
        n_sub=len(sub_streams),
        n_sub_text=len(text_subs),
        n_sub_bitmap=len(bitmap_subs),
        sub_langs=",".join(langs),
    )
