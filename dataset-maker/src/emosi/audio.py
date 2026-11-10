"""Pengukuran audio: SNR, clipping, dan ekstraksi potongan."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .media import run_ffmpeg

DEFAULT_SR = 16000


def extract_audio(path: Path, start_s: float, dur_s: float, sr: int = DEFAULT_SR) -> np.ndarray:
    """Ambil potongan audio sebagai float32 mono, lewat pipe ffmpeg (tanpa berkas sementara)."""
    raw = run_ffmpeg(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{max(0.0, start_s):.3f}",
            "-t",
            f"{max(0.1, dur_s):.3f}",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-",
        ],
        binary=True,
    )
    return np.frombuffer(raw, dtype=np.float32)


def frame_rms(samples: np.ndarray, sr: int = DEFAULT_SR, frame_ms: int = 20) -> np.ndarray:
    hop = max(1, int(sr * frame_ms / 1000))
    n_frames = samples.size // hop
    if n_frames == 0:
        return np.empty(0, dtype=np.float32)
    frames = samples[: n_frames * hop].reshape(n_frames, hop)
    return np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1) + 1e-12)


def snr_db(samples: np.ndarray, sr: int = DEFAULT_SR) -> float:
    """SNR kasar: persentil-90 energi (bicara) dibanding persentil-10 (dasar bising).

    Bukan SNR sejati, tetapi cukup untuk memisahkan film dengan dialog bersih
    dari film yang musiknya menutupi ucapan.
    """
    rms = frame_rms(samples, sr)
    if rms.size < 20:
        return 0.0
    speech = float(np.percentile(rms, 90))
    noise = float(np.percentile(rms, 10))
    if noise <= 1e-9:
        return 99.0
    return round(20.0 * float(np.log10(max(speech, 1e-9) / noise)), 2)


def clip_ratio(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return round(float(np.mean(np.abs(samples) >= 0.999)), 5)
