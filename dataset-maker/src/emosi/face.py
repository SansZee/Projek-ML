"""Screening wajah per ujaran: deteksi (RetinaFace) dan embedding (ArcFace).

Tahap 04 menjawab dua pertanyaan untuk setiap ujaran:

1. **Apakah wajahnya terlihat?** Kalau wajah hanya muncul di seperempat frame,
   modalitas visual tidak bisa dipakai dan ujaran itu dibuang di QC.
2. **Siapa orangnya?** Embedding ArcFace memungkinkan pengelompokan aktor,
   yang dibutuhkan split C (satu aktor tidak boleh ada di train dan test).

Modul ini sengaja memisahkan bagian yang murni (agregasi, pengelompokan) dari
bagian yang memanggil model, supaya bisa diuji tanpa GPU dan tanpa unduhan
bobot.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# pengaturan
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FaceSettings:
    model_pack: str = "buffalo_l"
    det_size: int = 640
    det_thresh: float = 0.5
    sample_fps: float = 2.0
    frame_width: int = 960
    min_face_px: int = 48
    dominance_min_ratio: float = 1.3
    min_face_frames: int = 1
    face_ratio_ok: float = 0.60
    cluster_threshold: float = 0.42
    cluster_min_utterances: int = 3
    device: str = "auto"


# --------------------------------------------------------------------------- #
# pembacaan frame
# --------------------------------------------------------------------------- #


def scaled_size(width: int, height: int, target_width: int) -> tuple[int, int]:
    """Ukuran frame setelah diperkecil, tanpa memperbesar dan tanpa ganjil.

    Lebar dan tinggi dibuat genap karena ``rawvideo`` dan encoder menghitung
    ukuran per frame dari keduanya; satu piksel ganjil membuat seluruh pipe
    bergeser dan frame berikutnya rusak.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"ukuran video tidak valid: {width}x{height}")
    width_out = min(int(target_width), int(width))
    width_out -= width_out % 2
    height_out = max(2, int(round(height * width_out / width / 2)) * 2)
    return width_out, height_out


def iter_rgb_frames(
    path: Path,
    start_s: float,
    dur_s: float,
    fps: float,
    width: int,
    height: int,
) -> Iterator[np.ndarray]:
    """Frame RGB dari ffmpeg, dibaca bertahap lewat pipe.

    ``subprocess.run`` tidak dipakai karena satu adegan 240 detik pada 2 fps
    menghasilkan ~550 MB data mentah; seluruhnya di memori membuat empat
    proses paralel kehabisan RAM. stderr ditampung di berkas sementara supaya
    pipe tidak penuh dan pesan ffmpeg tetap bisa dibaca saat gagal.
    """
    cmd = [
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
        f"fps={fps:.3f},scale={width}:{height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    frame_size = width * height * 3
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        assert proc.stdout is not None
        finished = False
        try:
            while True:
                chunk = proc.stdout.read(frame_size)
                if len(chunk) < frame_size:
                    break
                yield np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3)
            finished = True
        finally:
            proc.stdout.close()
            if proc.poll() is None:
                # Konsumen berhenti lebih awal (mis. frame sudah cukup). Matikan
                # ffmpeg dengan sengaja; SIGPIPE akibat itu bukan kegagalan.
                proc.terminate()
            code = proc.wait()
            if finished and code != 0:
                err.seek(0)
                tail = err.read().decode("utf-8", "replace")[-800:]
                raise RuntimeError(
                    f"ffmpeg gagal membaca frame ({code}) dari {path}\n{tail}"
                )


# --------------------------------------------------------------------------- #
# deteksi
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Detection:
    """Satu wajah pada satu frame. Koordinat dalam piksel frame yang didekode."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    embedding: np.ndarray | None = None

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class FrameFaces:
    """Semua wajah yang lolos filter pada satu frame."""

    width: int
    height: int
    faces: list[Detection]

    @property
    def frame_area(self) -> float:
        return float(max(1, self.width * self.height))


def detections_from_result(result: list[Any], *, min_face_px: int = 0) -> list[Detection]:
    """Ubah keluaran ``insightface`` menjadi ``Detection`` yang sudah difilter.

    Wajah kecil dilewati: pada frame 960 px, wajah 40 px hanya menyisakan
    ~20 px untuk ArcFace setelah crop, dan embedding sebesar itu tidak stabil.
    """
    detections: list[Detection] = []
    for face in result:
        x1, y1, x2, y2 = (float(value) for value in face.bbox)
        width = x2 - x1
        height = y2 - y1
        if min(width, height) < min_face_px:
            continue
        embedding = getattr(face, "normed_embedding", None)
        detections.append(
            Detection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                score=float(face.det_score),
                embedding=(
                    np.asarray(embedding, dtype=np.float32)
                    if embedding is not None
                    else None
                ),
            )
        )
    return detections


def dominant_face(
    faces: list[Detection], *, dominance_min_ratio: float = 1.3
) -> tuple[Detection | None, bool, int]:
    """Wajah terbesar dan apakah ia jelas mendominasi frame.

    Pada percakapan dua orang ukuran wajah hampir sama. Embedding rata-rata
    dari dua orang berbeda bukan wajah siapa pun, jadi frame seperti itu tetap
    dihitung untuk face ratio tetapi tidak dipakai untuk embedding aktor.
    """
    if not faces:
        return None, False, 0
    ranked = sorted(faces, key=lambda face: face.area, reverse=True)
    best = ranked[0]
    if len(ranked) == 1:
        return best, True, 1
    second = max(ranked[1].area, 1e-9)
    return best, best.area >= dominance_min_ratio * second, len(ranked)


# --------------------------------------------------------------------------- #
# agregasi per ujaran
# --------------------------------------------------------------------------- #


@dataclass
class FaceSummary:
    n_frames: int = 0
    n_face_frames: int = 0
    face_ratio: float = 0.0
    n_faces_max: int = 0
    multi_face_share: float = 0.0
    dominant_share: float = 0.0
    face_area_ratio: float = 0.0
    face_width_frac: float = 0.0
    det_score: float = 0.0
    emb_frames: int = 0
    embedding: np.ndarray | None = None

    def as_dict(self) -> dict:
        return {
            "n_frames": self.n_frames,
            "n_frames_face": self.n_face_frames,
            "face_ratio": self.face_ratio,
            "n_faces_max": self.n_faces_max,
            "multi_face_share": self.multi_face_share,
            "dominant_share": self.dominant_share,
            "face_area_ratio": self.face_area_ratio,
            "face_width_frac": self.face_width_frac,
            "det_score": self.det_score,
            "emb_frames": self.emb_frames,
        }


def summarize_frames(frames: list[FrameFaces], settings: FaceSettings) -> FaceSummary:
    """Ringkas deteksi beberapa frame menjadi satu baris per ujaran."""
    summary = FaceSummary(n_frames=len(frames))
    if not frames:
        return summary

    face_frames = 0
    multi_frames = 0
    dominant_frames = 0
    areas: list[float] = []
    width_fracs: list[float] = []
    scores: list[float] = []
    vectors: list[np.ndarray] = []
    weights: list[float] = []

    for frame in frames:
        usable = [
            face
            for face in frame.faces
            if min(face.width, face.height) >= settings.min_face_px
        ]
        if not usable:
            continue
        face_frames += 1
        summary.n_faces_max = max(summary.n_faces_max, len(usable))
        if len(usable) > 1:
            multi_frames += 1

        best, unambiguous, _ = dominant_face(
            usable, dominance_min_ratio=settings.dominance_min_ratio
        )
        if best is None:
            continue
        area_ratio = best.area / frame.frame_area
        areas.append(area_ratio)
        width_fracs.append(best.width / max(1, frame.width))
        scores.append(best.score)
        if unambiguous and best.embedding is not None:
            dominant_frames += 1
            vectors.append(np.asarray(best.embedding, dtype=np.float32))
            weights.append(max(1e-6, area_ratio * best.score))

    summary.n_face_frames = face_frames
    summary.face_ratio = round(face_frames / len(frames), 4)
    summary.multi_face_share = round(multi_frames / len(frames), 4)
    summary.dominant_share = round(dominant_frames / face_frames, 4) if face_frames else 0.0
    summary.face_area_ratio = round(float(np.median(areas)), 5) if areas else 0.0
    summary.face_width_frac = round(float(np.median(width_fracs)), 4) if width_fracs else 0.0
    summary.det_score = round(float(np.median(scores)), 4) if scores else 0.0

    if vectors:
        matrix = np.stack(vectors)
        matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
        weight = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        pooled = (matrix * weight).sum(axis=0) / max(float(weight.sum()), 1e-9)
        summary.embedding = _unit(pooled)
        summary.emb_frames = len(vectors)

    return summary


def frames_by_utterance(
    frame_times: list[float], spans: list[tuple[float, float]]
) -> list[list[int]]:
    """Petakan indeks frame ke ujaran yang memuat waktunya.

    Ujaran yang tumpang-tindih memang berbagi frame; itu benar, karena keduanya
    berlangsung pada gambar yang sama. Ujaran pendek yang tidak kebagian frame
    (mis. 0,5 detik pada 2 fps) diberi frame terdekat supaya tidak menghasilkan
    ``n_frames=0`` yang tidak bisa dibedakan dari kegagalan dekode.
    """
    buckets: list[list[int]] = [[] for _ in spans]
    for frame_index, time_s in enumerate(frame_times):
        for utter_index, (start_s, end_s) in enumerate(spans):
            if start_s <= time_s < end_s:
                buckets[utter_index].append(frame_index)
    if frame_times:
        for utter_index, bucket in enumerate(buckets):
            if bucket:
                continue
            middle = (spans[utter_index][0] + spans[utter_index][1]) / 2
            buckets[utter_index].append(
                min(range(len(frame_times)), key=lambda i: abs(frame_times[i] - middle))
            )
    return buckets


# --------------------------------------------------------------------------- #
# pengelompokan aktor
# --------------------------------------------------------------------------- #


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def cluster_embeddings(
    embeddings: np.ndarray,
    *,
    threshold: float = 0.42,
    min_size: int = 3,
    order: list[int] | None = None,
) -> np.ndarray:
    """Kelompokkan embedding dengan jarak kosinus, deterministik.

    Dipakai *per film*, bukan lintas film: wajah aktor yang sama di dua film
    berbeda bisa berbeda umur dan tata rias. Penggabungan lintas film terjadi
    di tahap terpusat setelah nama aktor diisi manual.

    ``order`` menentukan urutan pemasukan; kirim urutan face ratio menurun
    supaya centroid dibentuk dari frame terbaik lebih dulu.
    """
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("embeddings harus matriks 2 dimensi")
    count = matrix.shape[0]
    labels = np.full(count, -1, dtype=int)
    if count == 0:
        return labels

    matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
    sequence = list(range(count)) if order is None else list(order)

    centroids: list[np.ndarray] = []
    sizes: list[int] = []
    for index in sequence:
        vector = matrix[index]
        if centroids:
            similarities = np.array([float(vector @ centroid) for centroid in centroids])
            best = int(np.argmax(similarities))
            if similarities[best] >= 1.0 - threshold:
                labels[index] = best
                centroids[best] = _unit(centroids[best] * sizes[best] + vector)
                sizes[best] += 1
                continue
        labels[index] = len(centroids)
        centroids.append(vector.copy())
        sizes.append(1)

    # Klaster berisi satu-dua ujaran hampir selalu wajah salah orang atau
    # profil buram. Mereka ditandai -1 (tidak diketahui), bukan dipaksa masuk.
    remap = {
        label: new_label
        for new_label, label in enumerate(
            label for label, size in enumerate(sizes) if size >= min_size
        )
    }
    return np.array([remap.get(int(label), -1) for label in labels], dtype=int)


# --------------------------------------------------------------------------- #
# pemanggilan model
# --------------------------------------------------------------------------- #


def build_analyzer(settings: FaceSettings) -> tuple[Any, str]:
    """Siapkan RetinaFace + ArcFace. Mengembalikan (analyzer, nama provider)."""
    try:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
    except ImportError as exc:  # pragma: no cover - hanya terjadi tanpa extra face
        raise RuntimeError(
            "insightface belum terpasang. Jalankan: uv sync --locked --extra face"
        ) from exc

    # onnxruntime-gpu tidak menemukan libcublas/libcudnn dari paket nvidia-*-cu12
    # di site-packages tanpa bantuan ini; kegagalannya senyap dan semuanya jatuh
    # ke CPU (sekitar 10x lebih lambat).
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:  # pragma: no cover - CPU tetap bisa dipakai
            pass

    available = ort.get_available_providers()
    use_gpu = settings.device != "cpu" and "CUDAExecutionProvider" in available
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_gpu
        else ["CPUExecutionProvider"]
    )
    analyzer = FaceAnalysis(
        name=settings.model_pack,
        providers=providers,
        allowed_modules=["detection", "recognition"],
    )
    analyzer.prepare(
        ctx_id=0 if use_gpu else -1,
        det_size=(settings.det_size, settings.det_size),
        det_thresh=settings.det_thresh,
    )
    return analyzer, providers[0]


def analyze_frame(analyzer: Any, frame_rgb: np.ndarray, *, min_face_px: int) -> list[Detection]:
    """Deteksi wajah pada satu frame RGB.

    InsightFace memakai konvensi OpenCV (BGR). Frame dari ffmpeg berwarna RGB;
    tanpa konversi ini warna merah dan biru tertukar dan deteksi melemah.
    """
    bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
    return detections_from_result(analyzer.get(bgr), min_face_px=min_face_px)
