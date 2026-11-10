"""Transkripsi film dan pemeriksaan sinkronisasi subtitle.

Dua pekerjaan terpisah, sengaja dalam satu modul:

1. **Transkripsi** — faster-whisper menghasilkan kata beserta timestamp.
   Waktu kata inilah yang nanti dipakai 03_utterances.py, bukan waktu cue
   subtitle, karena subtitle sering lebih awal/lebih lambat dari ucapannya.

2. **Pemeriksaan SRT** — membandingkan teks subtitle dengan hasil transkripsi
   untuk membuktikan subtitle memang sinkron dengan berkas film yang dipakai.
   Tanpa bukti ini, seluruh pemotongan berdasarkan waktu bisa bergeser tanpa
   ada yang menyadarinya.

Modul ini tidak mengimpor faster-whisper di tingkat atas: `from emosi import asr`
harus tetap bisa dipakai walaupun extra `asr` belum dipasang.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from bisect import bisect_left
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .media import run_ffmpeg
from .subtitle import Cue

PUNCT = re.compile(r"[^\w\s']")
SPACE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# teks dan jarak edit
# --------------------------------------------------------------------------- #
def normalize_words(text: str) -> list[str]:
    """Huruf kecil, tanpa tanda baca. Apostrof dipertahankan (don't -> don't)."""
    return SPACE.sub(" ", PUNCT.sub(" ", text.lower())).strip().split()


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Jarak Levenshtein pada tingkat kata (jumlah salah, sisip, hapus)."""
    if not reference:
        return len(hypothesis)

    prev = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, start=1):
        cur = [i] + [0] * len(hypothesis)
        for j, hyp in enumerate(hypothesis, start=1):
            cur[j] = min(
                prev[j] + 1,  # hapus
                cur[j - 1] + 1,  # sisip
                prev[j - 1] + (ref != hyp),  # cocok / ganti
            )
        prev = cur
    return prev[-1]


def word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    """WER standar. Ditulis sendiri supaya modul ini bisa diuji tanpa extra `asr`."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


# --------------------------------------------------------------------------- #
# ekstraksi audio
# --------------------------------------------------------------------------- #
def extract_speech_audio(
    video_path: Path,
    out_path: Path,
    *,
    sr: int = 16000,
    stream: int = 0,
    limit_s: float | None = None,
    force: bool = False,
) -> Path:
    """Ambil satu track audio sebagai FLAC mono 16 kHz.

    Ditulis ke berkas, tidak dipipe, karena Whisper membaca audio berkali-kali
    dan film 2 jam butuh dekode ulang dari mkv setiap kali.

    Track 5.1 diturunkan ke mono oleh ffmpeg dengan matriks downmix standar,
    sehingga channel tengah (dialog) tetap dominan.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        return out_path

    tmp = out_path.with_name(out_path.stem + ".tmp.flac")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(video_path)]
    if limit_s is not None:
        cmd += ["-t", f"{max(0.5, limit_s):.3f}"]
    cmd += [
        "-map",
        f"0:a:{stream}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-c:a",
        "flac",
        "-compression_level",
        "8",
        str(tmp),
    ]
    run_ffmpeg(cmd)
    tmp.replace(out_path)
    return out_path


# --------------------------------------------------------------------------- #
# indeks kata
# --------------------------------------------------------------------------- #
@dataclass
class WordIndex:
    """Kata hasil transkripsi, terurut, untuk pencarian cepat per rentang waktu."""

    starts: list[float]
    ends: list[float]
    words: list[str]

    @classmethod
    def from_segments(cls, segments: list[dict]) -> WordIndex:
        starts: list[float] = []
        ends: list[float] = []
        words: list[str] = []
        for segment in segments:
            for word in segment.get("words") or []:
                if word.get("start") is None:
                    continue
                start = float(word["start"])
                # Kata tanpa akhir (jarang) diberi durasi wajar supaya topeng
                # aktivitas tidak berlubang.
                end = float(word.get("end") or start + 0.2)
                starts.append(start)
                ends.append(max(end, start + 0.02))
                words.append(str(word.get("word", "")))
        return cls(starts=starts, ends=ends, words=words)

    def between(self, t0: float, t1: float) -> list[str]:
        """Kata yang mulai di dalam [t0, t1)."""
        lo = bisect_left(self.starts, t0)
        hi = bisect_left(self.starts, t1)
        return [word for word in self.words[lo:hi] if word.strip()]


def load_asr(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# pemeriksaan sinkronisasi
# --------------------------------------------------------------------------- #
@dataclass
class CheckSettings:
    block_s: float = 20.0      # panjang satu blok pembanding, detik
    max_blocks: int = 12       # jumlah blok yang diuji, tersebar sepanjang film
    onset_window_s: float = 0.75  # jendela pencarian kata terdekat dari awal cue
    pad_s: float = 0.5         # toleransi di tepi blok
    wer_ok: float = 0.30       # di bawah ini: subtitle selaras
    wer_poor: float = 0.55     # di atas ini: subtitle bukan untuk potongan ini


def build_blocks(cues: list[Cue], block_s: float, max_blocks: int) -> list[list[Cue]]:
    """Kelompokkan cue berurutan menjadi blok, lalu ambil blok yang tersebar.

    Membandingkan per cue satu per satu menyesatkan: satu kata yang melewati
    batas cue dihitung sebagai hapus di cue pertama *dan* sisip di cue
    berikutnya, sehingga WER naik tanpa ada kesalahan transkripsi. Dengan blok
    20 detik, efek batas hanya terjadi di tepi blok, bukan di setiap cue.
    """
    ordered = sorted((cue for cue in cues if cue.text.strip()), key=lambda cue: cue.start)
    chunks: list[list[Cue]] = []
    current: list[Cue] = []
    for cue in ordered:
        current.append(cue)
        if cue.end - current[0].start >= block_s:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    if len(chunks) <= max_blocks:
        return chunks
    step = len(chunks) / max_blocks
    return [chunks[int(i * step)] for i in range(max_blocks)]


def block_error(
    blocks: list[list[Cue]], index: WordIndex, *, shift_s: float = 0.0, pad_s: float = 0.5
) -> tuple[float, int, int]:
    """WER gabungan: total kata salah dibagi total kata acuan.

    Digabung, bukan dirata-ratakan per blok, supaya blok pendek tidak
    berbobot sama dengan blok panjang.
    """
    edits = 0
    total = 0
    for block in blocks:
        reference = normalize_words(" ".join(cue.text for cue in block))
        if not reference:
            continue
        hypothesis = normalize_words(
            " ".join(
                index.between(
                    block[0].start + shift_s - pad_s, block[-1].end + shift_s + pad_s
                )
            )
        )
        edits += edit_distance(reference, hypothesis)
        total += len(reference)
    return (round(edits / total, 4) if total else 1.0), edits, total


def onset_offsets(cues: list[Cue], index: WordIndex, window_s: float = 0.75) -> list[float]:
    """Selisih antara awal cue dan awal kata terdekat, satu nilai per cue.

    Positif berarti cue mulai **setelah** kata, yaitu subtitle terlambat.

    Yang diukur adalah jarak ke onset kata terdekat, bukan tumpang-tindih
    topeng aktivitas. Topeng selalu terlihat "telat" karena subtitle bertahan
    di layar setelah kalimat selesai — pada About Time cara itu memberi +0,50 s
    padahal onset kata dan energi audio menunjukkan 0,00 s.
    """
    offsets: list[float] = []
    for cue in cues:
        lo = bisect_left(index.starts, cue.start - window_s)
        hi = bisect_left(index.starts, cue.start + window_s)
        if hi <= lo:
            continue
        nearest = min(index.starts[lo:hi], key=lambda start: abs(start - cue.start))
        offsets.append(cue.start - nearest)
    return offsets


def median_onset_offset(
    cues: list[Cue], index: WordIndex, window_s: float = 0.75
) -> tuple[float | None, int]:
    """Median geseran onset, atau None bila terlalu sedikit cue yang terukur."""
    offsets = onset_offsets(cues, index, window_s)
    if len(offsets) < max(5, len(cues) // 5):
        return None, len(offsets)
    return round(statistics.median(offsets), 3), len(offsets)


def coverage(cues: list[Cue], index: WordIndex, pad_s: float = 1.0) -> float:
    """Bagian kata transkripsi yang jatuh di dalam salah satu cue.

    Nilai rendah berarti subtitle dan film bukan potongan yang sama, atau
    ada bagian besar film yang tidak disubtitle.
    """
    if not index.starts:
        return 0.0
    spans = sorted((cue.start - pad_s, cue.end + pad_s) for cue in cues)
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    begins = [span[0] for span in merged]

    inside = 0
    for start in index.starts:
        position = bisect_left(begins, start) - 1
        if position >= 0 and merged[position][1] >= start:
            inside += 1
    return round(inside / len(index.starts), 4)


def check_sync(film_id: str, cues: list[Cue], asr: dict, settings: CheckSettings) -> dict:
    """Bandingkan subtitle dengan transkripsi: WER, geseran, dan cakupan.

    WER diukur setelah geseran subtitle dikoreksi, karena subtitle yang
    terlambat 0,3 detik tetap benar isinya.
    """
    index = WordIndex.from_segments(asr.get("segments") or [])
    blocks = build_blocks(cues, settings.block_s, settings.max_blocks)

    row: dict[str, Any] = {
        "film_id": film_id,
        "n_cues": len(cues),
        "n_blocks": len(blocks),
        "n_cues_checked": sum(len(block) for block in blocks),
        "srt_words": sum(len(cue.text.split()) for cue in cues),
        "asr_words": len(index.words),
        "coverage": coverage(cues, index, pad_s=settings.pad_s),
        "wer": None,
        "wer_at_zero": None,
        "offset": None,
        "offset_cues": None,
        "offset_first": None,
        "offset_mid": None,
        "offset_last": None,
        "drift_s": None,
        "verdict": "tidak_ada_data",
        "notes": "",
    }
    if not blocks or not index.words:
        row["notes"] = "cue atau kata transkripsi kosong"
        return row

    offset, measured = median_onset_offset(cues, index, settings.onset_window_s)
    row["offset"] = offset
    row["offset_cues"] = measured
    row["wer_at_zero"] = block_error(blocks, index, pad_s=settings.pad_s)[0]
    shift = -(offset or 0.0)
    row["wer"] = block_error(blocks, index, shift_s=shift, pad_s=settings.pad_s)[0]

    lo = min(cue.start for cue in cues)
    hi = max(cue.end for cue in cues)
    width = (hi - lo) / 3
    for position, key in enumerate(("offset_first", "offset_mid", "offset_last")):
        begin = lo + position * width
        end = begin + width
        part = [cue for cue in cues if begin <= cue.start < end]
        if part:
            row[key], _measured = median_onset_offset(part, index, settings.onset_window_s)

    offsets = [row[key] for key in ("offset_first", "offset_mid", "offset_last")]
    offsets = [value for value in offsets if value is not None]
    if len(offsets) >= 2:
        row["drift_s"] = round(max(offsets) - min(offsets), 3)

    if row["wer"] <= settings.wer_ok:
        row["verdict"] = "selaras"
    elif row["wer"] <= settings.wer_poor:
        row["verdict"] = "kasar"
    else:
        row["verdict"] = "tidak_cocok"

    notes: list[str] = []
    if row["drift_s"] is not None and row["drift_s"] > 1.0:
        notes.append(f"geseran berubah {row['drift_s']:.2f} s dari awal ke akhir")
    if row["offset"] is None:
        notes.append("geseran tidak bisa diukur, kata terlalu jauh dari cue")
    elif abs(row["offset"]) > 0.5:
        notes.append(
            f"subtitle {'terlambat' if row['offset'] > 0 else 'terlalu awal'} "
            f"{abs(row['offset']):.2f} s"
        )
    if row["coverage"] < 0.6:
        notes.append("banyak ucapan di luar cue subtitle")
    row["notes"] = "; ".join(notes)
    return row


# --------------------------------------------------------------------------- #
# transkripsi
# --------------------------------------------------------------------------- #
_PRELOADED = False


def preload_cuda_libs() -> list[str]:
    """Muat libcublas/cuDNN dari paket pip ``nvidia-*`` ke dalam proses.

    ctranslate2 memanggil ``dlopen("libcublas.so.12")`` saat runtime. Kalau
    library itu hanya ada di ``site-packages/nvidia/...``, pencarian normal
    tidak menemukannya dan hasilnya error "Library libcublas.so.12 is not
    found" walau paketnya sudah terpasang.

    Dimuat di sini dengan RTLD_GLOBAL sebelum ctranslate2 diimpor, supaya
    pengguna tidak perlu mengatur LD_LIBRARY_PATH di shell-nya masing-masing
    (dan lupa saat pindah laptop).
    """
    global _PRELOADED
    if _PRELOADED:
        return []

    import ctypes
    import os
    import sys
    import sysconfig

    _PRELOADED = True
    root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    if not root.exists():
        return []

    libraries = sorted(root.glob("cublas/lib/lib*.so*")) + sorted(root.glob("cudnn/lib/lib*.so*"))
    loaded: list[str] = []
    for library in libraries:
        try:
            if sys.platform == "win32":
                os.add_dll_directory(str(library.parent))
            else:
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            loaded.append(library.name)
        except OSError:
            continue
    return loaded


def cuda_device_count() -> int:
    preload_cuda_libs()
    try:
        import ctranslate2
    except ImportError:
        return 0
    try:
        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001 - driver rusak tidak boleh menghentikan skrip
        return 0


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if cuda_device_count() > 0 else "cpu"


def load_model(name: str, device: str, compute_type: str, cache_dir: Path):
    from faster_whisper import WhisperModel

    if device == "cuda":
        preload_cuda_libs()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return WhisperModel(
        name,
        device=device,
        compute_type=compute_type,
        download_root=str(cache_dir),
    )


def transcribe_kwargs(settings: dict) -> dict:
    """Argumen ``model.transcribe`` yang berasal dari config, di satu tempat.

    Semua nilai yang memengaruhi isi transkrip diambil dari sini, bukan dari
    default faster-whisper, supaya empat laptop menghasilkan teks sebanding.
    """
    return {
        "language": settings["language"],
        "beam_size": int(settings["beam_size"]),
        "word_timestamps": True,
        "vad_filter": bool(settings["vad_filter"]),
        "vad_parameters": {
            "threshold": float(settings.get("vad_threshold", 0.5)),
            "min_silence_duration_ms": int(settings["vad_min_silence_ms"]),
            "speech_pad_ms": int(settings["vad_speech_pad_ms"]),
        },
        "condition_on_previous_text": bool(settings["condition_on_previous_text"]),
        "compression_ratio_threshold": float(settings["compression_ratio_threshold"]),
        "no_speech_threshold": float(settings["no_speech_threshold"]),
        "temperature": [float(t) for t in settings["temperatures"]],
    }


def _segment_dicts(segments, info) -> Iterator[dict]:
    for segment in segments:
        words = [
            {
                "word": word.word,
                "start": round(float(word.start), 3),
                "end": round(float(word.end), 3),
                "prob": round(float(word.probability), 4),
            }
            for word in (segment.words or [])
            if word.start is not None and word.end is not None
        ]
        # Segmen tanpa kata sering berasal dari musik atau kebisingan; tetap
        # disimpan agar bisa diperiksa, tetapi ditandai.
        suspicious = bool(
            segment.no_speech_prob > 0.6 and (segment.avg_logprob or 0.0) < -0.8
        )
        yield {
            "start": round(float(segment.start), 3),
            "end": round(float(segment.end), 3),
            "text": segment.text.strip(),
            "avg_logprob": round(float(segment.avg_logprob or 0.0), 4),
            "no_speech_prob": round(float(segment.no_speech_prob or 0.0), 4),
            "temperature": round(float(getattr(segment, "temperature", 0.0) or 0.0), 3),
            "suspicious": suspicious,
            "n_words": len(words),
            "words": words,
        }


@dataclass
class TranscriptionResult:
    segments: list[dict]
    language: str
    language_probability: float
    audio_duration_s: float
    runtime_s: float

    @property
    def n_words(self) -> int:
        return sum(segment["n_words"] for segment in self.segments)

    @property
    def speedup(self) -> float:
        return round(self.audio_duration_s / self.runtime_s, 2) if self.runtime_s > 0 else 0.0


def transcribe(
    model,
    audio_path: Path,
    settings: dict,
    *,
    partial_path: Path | None = None,
    duration_s: float = 0.0,
    progress=None,
) -> TranscriptionResult:
    """Jalankan transkripsi, simpan tiap segmen ke JSONL begitu dihasilkan.

    JSONL parsial ditulis supaya hasil 90 menit pertama tidak hilang kalau
    proses mati di akhir. Berkas itu dihapus setelah JSON final selesai.
    """
    segments, info = model.transcribe(str(audio_path), **transcribe_kwargs(settings))
    total = duration_s or float(getattr(info, "duration", 0.0) or 0.0)
    started = time.time()
    collected: list[dict] = []

    handle = None
    if partial_path is not None:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        handle = partial_path.open("w", encoding="utf-8")
    try:
        for record in _segment_dicts(segments, info):
            collected.append(record)
            if handle is not None:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            if progress is not None and total > 0:
                progress(min(1.0, record["end"] / total))
    finally:
        if handle is not None:
            handle.close()

    return TranscriptionResult(
        segments=collected,
        language=str(getattr(info, "language", settings["language"])),
        language_probability=round(float(getattr(info, "language_probability", 0.0) or 0.0), 4),
        audio_duration_s=round(total, 3),
        runtime_s=round(time.time() - started, 2),
    )
