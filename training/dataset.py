"""Dataset loader teks, audio WAV, video/wajah MP4, dan cache fitur ter-ekstrak."""

from __future__ import annotations

from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF
from torch.utils.data import Dataset

from config_loader import ProjectConfig


def load_master_dataframe(cfg: ProjectConfig, *, require_splits: bool = True) -> pd.DataFrame:
    """Gabungkan ``clips.csv``, teks dari ``pool.csv``, dan ``splits.csv`` bila tersedia."""
    clips_path = cfg.data_dir / "clips.csv"
    pool_path = cfg.data_dir / "pool.csv"
    splits_path = cfg.data_dir / "splits.csv"

    if not clips_path.exists():
        raise FileNotFoundError(f"Tidak ada {clips_path}. Jalankan 10b_klip.py --gabung dulu.")
    if require_splits and not splits_path.exists():
        raise FileNotFoundError(
            f"Tidak ada {splits_path}. Minta splits.csv dari koordinator atau jalankan scripts/11_split.py."
        )

    clips = pd.read_csv(clips_path)
    clips = clips[clips["status"].isin(("ok", "ada"))].copy()
    if clips.empty:
        raise RuntimeError("Tidak ada klip berstatus ok/ada di data/clips.csv.")

    clips["clip_id"] = clips["clip_id"].astype(str)
    clips["utterance_id"] = clips["utterance_id"].astype(str)
    clips["label"] = clips["label"].astype(str)

    if pool_path.exists():
        pool = pd.read_csv(pool_path, usecols=["utterance_id", "text"]).drop_duplicates("utterance_id")
        pool["utterance_id"] = pool["utterance_id"].astype(str)
        pool["text"] = pool["text"].fillna("").astype(str)
        clips = clips.merge(pool, on="utterance_id", how="left")
    else:
        clips["text"] = ""
    clips["text"] = clips["text"].fillna("").astype(str)

    if splits_path.exists():
        splits = pd.read_csv(splits_path)
        splits["clip_id"] = splits["clip_id"].astype(str)
        split_cols = [c for c in ("clip_id", "split_A", "split_B", "split_C") if c in splits.columns]
        clips = clips.merge(splits[split_cols].drop_duplicates("clip_id"), on="clip_id", how="inner")

    label_to_idx = {name: i for i, name in enumerate(cfg.classes)}
    clips = clips[clips["label"].isin(label_to_idx)].copy()
    clips["label_idx"] = clips["label"].map(label_to_idx).astype(int)
    return clips.sort_values("clip_id", kind="stable").reset_index(drop=True)


def load_waveform(path: Path, *, target_sr: int = 16000, max_seconds: float = 10.0) -> torch.Tensor:
    """Baca audio ``.wav`` mono ``target_sr`` Hz, pad/potong hingga ``max_seconds``."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = torch.from_numpy(data.mean(axis=1))
    if sr != target_sr:
        mono = AF.resample(mono, orig_freq=sr, new_freq=target_sr)
    max_len = int(target_sr * max_seconds)
    if mono.numel() > max_len:
        mono = mono[:max_len]
    elif mono.numel() < max_len:
        pad = torch.zeros(max_len - mono.numel(), dtype=mono.dtype)
        mono = torch.cat([mono, pad], dim=0)
    return mono


def load_video_frames(
    path: Path,
    *,
    num_frames: int = 16,
    size: int = 224,
    fallback_path: Path | None = None,
) -> torch.Tensor:
    """Sampling merata ``num_frames`` frame RGB ``[T, C, H, W]`` bernilai ``uint8``."""
    frames: list[np.ndarray] = []
    source = path if path.exists() else fallback_path
    if source is not None and source.exists():
        try:
            with av.open(str(source)) as container:
                for frame in container.decode(video=0):
                    frames.append(frame.to_ndarray(format="rgb24"))
        except Exception:
            cap = cv2.VideoCapture(str(source))
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            cap.release()

    if not frames:
        return torch.zeros((num_frames, 3, size, size), dtype=torch.uint8)

    indices = np.linspace(0, len(frames) - 1, num=num_frames, dtype=int)
    sampled = []
    for idx in indices:
        img = frames[int(idx)]
        if img.shape[0] != size or img.shape[1] != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        sampled.append(torch.from_numpy(img).permute(2, 0, 1))
    return torch.stack(sampled, dim=0).contiguous()


class RawClipDataset(Dataset):
    """Dataset multimodal mentah dari disk untuk ekstraksi fitur atau unimodal fine-tuning."""

    def __init__(
        self,
        frame: pd.DataFrame,
        media_dir: Path,
        *,
        modality: str,
        target_sr: int = 16000,
        max_audio_s: float = 10.0,
        num_frames: int = 16,
        image_size: int = 224,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.media_dir = Path(media_dir)
        self.modality = modality
        self.target_sr = target_sr
        self.max_audio_s = max_audio_s
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.frame.iloc[idx]
        item: dict[str, object] = {
            "clip_id": str(row["clip_id"]),
            "utterance_id": str(row["utterance_id"]),
            "label": int(row["label_idx"]),
        }
        if self.modality in ("text", "all"):
            item["text"] = str(row["text"])
        if self.modality in ("audio", "all"):
            audio_rel = str(row.get("audio_path", "") or f"audio/{row['clip_id']}.wav")
            item["audio"] = load_waveform(
                self.media_dir / audio_rel,
                target_sr=self.target_sr,
                max_seconds=self.max_audio_s,
            )
        if self.modality in ("visual", "all"):
            face_rel = str(row.get("face_path", "") or f"faces/{row['clip_id']}.mp4")
            clip_rel = str(row.get("clip_path", "") or f"clips/{row['clip_id']}.mp4")
            item["video"] = load_video_frames(
                self.media_dir / face_rel,
                num_frames=self.num_frames,
                size=self.image_size,
                fallback_path=self.media_dir / clip_rel,
            )
        return item


class CachedFeatureDataset(Dataset):
    """Dataset fitur ter-cache untuk pelatihan cepat head unimodal, fusi, dan ablasi."""

    def __init__(
        self,
        frame: pd.DataFrame,
        caches: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        self.clip_ids = frame["clip_id"].astype(str).tolist()
        self.utterance_ids = frame["utterance_id"].astype(str).tolist()
        self.labels = torch.tensor(frame["label_idx"].to_numpy(), dtype=torch.long)
        self.caches = caches

    def __len__(self) -> int:
        return len(self.clip_ids)

    def __getitem__(self, idx: int) -> dict[str, object]:
        cid = self.clip_ids[idx]
        item: dict[str, object] = {
            "clip_id": cid,
            "utterance_id": self.utterance_ids[idx],
            "label": self.labels[idx],
        }
        for mod, mapping in self.caches.items():
            item[mod] = mapping[cid]
        return item


def collate_cached_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    """Gabungkan fitur sekuensial/pooled ter-cache menjadi batch dengan padding bila ``3D``."""
    out: dict[str, object] = {
        "clip_id": [b["clip_id"] for b in batch],
        "utterance_id": [b["utterance_id"] for b in batch],
        "label": torch.stack([b["label"] for b in batch]),  # type: ignore[arg-type]
    }
    for mod in ("text", "audio", "visual"):
        if mod not in batch[0]:
            continue
        tensors = [b[mod] for b in batch]  # type: ignore[misc]
        first = tensors[0]
        if first.ndim == 1:
            out[mod] = torch.stack(tensors).float()
        else:
            max_t = max(t.shape[0] for t in tensors)
            dim = first.shape[-1]
            padded = torch.zeros((len(tensors), max_t, dim), dtype=torch.float32)
            mask = torch.ones((len(tensors), max_t), dtype=torch.bool)  # True = padded
            for i, t in enumerate(tensors):
                n = t.shape[0]
                padded[i, :n] = t.float()
                mask[i, :n] = False
            out[mod] = padded
            out[f"{mod}_pad_mask"] = mask
    return out
