"""Pengambilan frame dan deteksi wajah untuk audisi (bukan untuk dataset final).

Deteksi di sini memakai Haar cascade bawaan OpenCV: cepat, tanpa unduhan model,
dan cukup untuk mengukur *rasio frame yang mengandung wajah*. Untuk kotak wajah
presisi dan embedding, gunakan tahap 04_face_screen.py (RetinaFace/ArcFace).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .media import run_ffmpeg


def iter_gray_frames(
    path: Path,
    start_s: float,
    dur_s: float,
    fps: float = 1.0,
    width: int = 640,
    height: int = 360,
) -> Iterator[np.ndarray]:
    """Frame grayscale lewat pipe ffmpeg.

    Sengaja tidak memakai cv2.VideoCapture: HEVC 10-bit sering gagal didekode
    di sana dan gagalnya diam-diam (frame hitam).
    """
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
            "-vf",
            f"fps={fps},scale={width}:{height},format=gray",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        binary=True,
    )
    frame_size = width * height
    for offset in range(0, len(raw) - frame_size + 1, frame_size):
        frame = np.frombuffer(raw[offset : offset + frame_size], dtype=np.uint8)
        yield frame.reshape(height, width)


def load_haar_detector():
    import cv2

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"cascade wajah tidak bisa dimuat: {cascade_path}")
    return detector


def face_stats(
    path: Path,
    start_s: float,
    dur_s: float,
    fps: float = 1.0,
    width: int = 640,
    height: int = 360,
    min_size_px: int = 40,
) -> dict:
    detector = load_haar_detector()
    n_frames = 0
    n_face = 0
    n_multi = 0
    largest = 0

    for frame in iter_gray_frames(path, start_s, dur_s, fps=fps, width=width, height=height):
        n_frames += 1
        faces = detector.detectMultiScale(
            frame, scaleFactor=1.1, minNeighbors=5, minSize=(min_size_px, min_size_px)
        )
        detected = len(faces) if hasattr(faces, "__len__") else 0
        if detected:
            n_face += 1
            if detected > 1:
                n_multi += 1
            largest = max(largest, max(int(w) for (_, _, w, _h) in faces))

    return {
        "face_frames": n_face,
        "probe_frames": n_frames,
        "face_ratio": round(n_face / n_frames, 4) if n_frames else 0.0,
        "multi_face_ratio": round(n_multi / n_frames, 4) if n_frames else 0.0,
        "largest_face_px": largest,
    }
