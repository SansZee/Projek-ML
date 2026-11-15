"""Strip tiga potongan wajah untuk aplikasi pelabelan.

Anotator perlu melihat wajah yang benar-benar dipakai model, bukan wajah siapa
pun di adegan. Strip diambil pada 25/50/75% ujaran dengan aturan yang sama
dengan ``10_cut.py``: wajah terbesar di frame, margin kotak 1,35, hasil 224x224.

Tile abu-abu berarti tidak ada wajah terdeteksi pada titik itu — anotator harus
tahu bagian mana yang akan diterima model apa adanya.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from emosi import cut, face

TILE = 224
FRACTIONS = (0.25, 0.5, 0.75)
EMPTY = 60


def target_times(start_s: float, end_s: float) -> list[float]:
    """Tiga titik sampel di dalam satu ujaran."""
    start = float(start_s)
    end = max(start + 0.05, float(end_s))
    return [round(start + (end - start) * fraction, 3) for fraction in FRACTIONS]


def strip_path(directory: Path, utterance_id: str) -> Path:
    return Path(directory) / f"{utterance_id}.jpg"


def compose_strip(tiles: Sequence[np.ndarray | None], size: int = TILE) -> np.ndarray:
    """Gabung tiga tile jadi satu baris; tile kosong diisi abu-abu."""
    blank = np.full((size, size, 3), EMPTY, dtype=np.uint8)
    return np.hstack([blank if tile is None else tile for tile in tiles])


def strip_tiles(
    analyzer,
    source: Path,
    start_s: float,
    end_s: float,
    width: int,
    height: int,
    *,
    settings: face.FaceSettings,
    margin: float = 1.35,
    size: int = TILE,
) -> list[np.ndarray | None]:
    """Tiga crop wajah (RGB) pada 25/50/75% ujaran."""
    import cv2

    targets = target_times(start_s, end_s)
    duration = max(0.05, float(end_s) - float(start_s))
    fps = max(1.0, float(settings.sample_fps))

    nearest: list[tuple[float, np.ndarray] | None] = [None] * len(targets)
    for index, frame in enumerate(
        face.iter_rgb_frames(source, float(start_s), duration, fps, width, height)
    ):
        moment = float(start_s) + index / fps
        for slot, target in enumerate(targets):
            distance = abs(moment - target)
            if nearest[slot] is None or distance < nearest[slot][0]:
                nearest[slot] = (distance, frame)

    tiles: list[np.ndarray | None] = []
    for item in nearest:
        if item is None:
            tiles.append(None)
            continue
        frame = item[1]
        detections = face.analyze_frame(analyzer, frame, min_face_px=settings.min_face_px)
        if not detections:
            tiles.append(None)
            continue
        best = max(detections, key=lambda item_: item_.area)
        left, top, right, bottom = cut.face_box(
            best.x1, best.y1, best.x2, best.y2, width, height, margin=margin
        )
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            tiles.append(None)
            continue
        tiles.append(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA))
    return tiles


def save_strip(path: Path, strip_rgb: np.ndarray, *, quality: int = 90) -> None:
    """Tulis strip RGB sebagai JPEG; isi berkas ~40-60 KB."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(path),
        np.ascontiguousarray(strip_rgb[:, :, ::-1]),
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )
