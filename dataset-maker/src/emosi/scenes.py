"""Deteksi potongan gambar dan pengelompokannya menjadi adegan anotasi.

Dua tingkat yang berbeda dan sering tertukar:

- **Shot** (potongan gambar): satu rangkaian kamera tanpa cut. Film 2 jam punya
  1000-2000 shot, rata-rata beberapa detik.
- **Adegan anotasi**: 2-3 menit berisi banyak shot. Ini yang ditonton anotator
  dalam satu sesi, karena emosi butuh konteks cerita, bukan potongan 4 detik.

Deteksi shot memakai filter ``scdet`` bawaan ffmpeg: satu kali dekode, tanpa
menyimpan berkas antara, dan tidak bergantung pada backend OpenCV yang sudah
terbukti gagal membaca HEVC 10-bit pada berkas uji.
"""

from __future__ import annotations

import re
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

CUT_LINE = re.compile(r"lavfi\.scd\.time:\s*([0-9]+(?:\.[0-9]+)?)")


@dataclass
class Scene:
    index: int
    start_s: float
    end_s: float
    n_shots: int
    n_cues: int
    dialogue_s: float

    @property
    def duration_s(self) -> float:
        return round(self.end_s - self.start_s, 3)


def detect_cuts(
    path: Path,
    *,
    start_s: float = 0.0,
    end_s: float | None = None,
    threshold: float = 5.0,
    width: int = 480,
    progress: Callable[[float], None] | None = None,
) -> list[float]:
    """Waktu cut gambar di dalam [start_s, end_s], dalam detik absolut.

    ``threshold`` memakai skala 0-100 milik filter scdet (bukan skala
    ContentDetector). Pada About Time, ambang 5 menghasilkan 46 cut per 3 menit,
    setara ContentDetector 27 yang menghasilkan 49.
    """
    duration = (end_s - start_s) if end_s else None
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info", "-progress", "pipe:1"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.3f}"]
    if duration:
        cmd += ["-t", f"{max(0.1, duration):.3f}"]
    cmd += [
        "-i",
        str(path),
        "-an",
        "-sn",
        "-dn",
        "-vf",
        f"scale={width}:-2,scdet=threshold={threshold}",
        "-f",
        "null",
        "-",
    ]

    cuts: list[float] = []

    def read_cuts(stream) -> None:
        for line in stream:
            match = CUT_LINE.search(line)
            if match:
                cuts.append(round(float(match.group(1)) + start_s, 3))

    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        reader = threading.Thread(target=read_cuts, args=(proc.stderr,), daemon=True)
        reader.start()
        for line in proc.stdout:
            if progress is None or not duration:
                continue
            if line.startswith("out_time_us="):
                value = line.split("=", 1)[1].strip()
                if value.isdigit():
                    progress(min(1.0, int(value) / 1e6 / duration))
        proc.wait()
        reader.join(timeout=5)

    return sorted(set(cuts))


def timecode(seconds: float) -> str:
    """HH:MM:SS, untuk ditampilkan ke anotator dan dibaca manusia."""
    total = int(round(max(0.0, float(seconds))))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def annotation_window(
    duration_s: float,
    srt_first_s: float | None = None,
    srt_last_s: float | None = None,
    *,
    head_pad_s: float = 20.0,
    tail_pad_s: float = 20.0,
) -> tuple[float, float]:
    """Rentang film yang layak dijadikan adegan anotasi.

    Bagian tanpa dialog dibuang: logo studio, kredit pembuka, dan kredit
    penutup tidak bisa dilabeli emosinya. Tentunya hanya kalau subtitle memang
    tersedia; tanpa subtitle seluruh durasi dipakai.
    """
    duration = float(duration_s)
    end = duration
    if srt_last_s is not None and srt_last_s > 0:
        end = min(duration, float(srt_last_s) + tail_pad_s)

    start = 0.0
    if srt_first_s is not None and srt_first_s > 0:
        start = min(max(0.0, float(srt_first_s) - head_pad_s), max(0.0, end - 1.0))

    return round(start, 3), round(max(end, start + 1.0), 3)


def build_scenes(
    cuts: list[float],
    start_s: float,
    end_s: float,
    *,
    target_s: float = 150.0,
    min_s: float = 90.0,
    max_s: float = 240.0,
) -> list[tuple[float, float]]:
    """Gabungkan shot menjadi adegan 2-3 menit yang berhenti di cut gambar.

    Batas adegan dipilih dari cut terdekat dengan durasi target di dalam
    jendela [min_s, max_s]. Kalau di jendela itu tidak ada cut sama sekali
    (adegan panjang tanpa potongan), dipotong paksa di titik target.

    Adegan terakhir yang lebih pendek dari ``min_s`` tidak dibiarkan: batas
    sebelumnya digeser ke cut yang paling dekat dengan titik tengah rentang
    sisa, sehingga dua adegan terakhir sama-sama layak ditonton.
    """
    if end_s <= start_s:
        return []
    if end_s - start_s <= max_s:
        return [(round(start_s, 3), round(end_s, 3))]

    usable = sorted({cut for cut in cuts if start_s < cut < end_s})
    bounds = [float(start_s)]

    while end_s - bounds[-1] > max_s:
        cursor = bounds[-1]
        wanted = cursor + target_s
        window = [cut for cut in usable if cursor + min_s <= cut <= cursor + max_s]
        # Cut yang menyisakan ekor terlalu pendek dilewati kalau ada pilihan
        # lain: lebih baik memotong di titik target daripada membuat adegan
        # 30 detik di akhir.
        roomy = [cut for cut in window if end_s - cut >= min_s]
        choices = roomy or window
        if choices:
            nxt = min(choices, key=lambda cut: abs(cut - wanted))
        else:
            nxt = wanted
        # Jaga-jaga kalau config diubah menjadi target < min atau > max.
        bounds.append(min(max(nxt, cursor + min_s), cursor + max_s))

    bounds.append(float(end_s))
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    if len(spans) >= 2 and spans[-1][1] - spans[-1][0] < min_s:
        tail, previous = spans.pop(), spans.pop()
        combined = (previous[0], tail[1])
        # Tidak ada cut yang bisa dipakai: semua cut di dekat sini sudah
        # dipertimbangkan pada langkah sebelumnya, dan yang tersisa justru
        # membuat ekor pendek. Karena itu rentang dibelah di titik tengah.
        if combined[1] - combined[0] > 2 * min_s:
            middle = (combined[0] + combined[1]) / 2
            spans += [(combined[0], middle), (middle, combined[1])]
        else:
            # Config yang tidak lazim (max < 2*min): tidak ada cara membagi
            # tanpa melanggar salah satu batas, jadi sisa digabung saja.
            spans.append(combined)

    return [(round(a, 3), round(b, 3)) for a, b in spans]


def count_shots(cuts: list[float], start_s: float, end_s: float) -> int:
    return sum(1 for cut in cuts if start_s < cut < end_s) + 1


def dialogue_stats(cues, start_s: float, end_s: float) -> tuple[int, float]:
    """Jumlah cue dan total durasi dialog di dalam adegan."""
    n_cues = 0
    seconds = 0.0
    for cue in cues:
        overlap = min(cue.end, end_s) - max(cue.start, start_s)
        if overlap > 0:
            n_cues += 1
            seconds += overlap
    return n_cues, round(seconds, 2)
