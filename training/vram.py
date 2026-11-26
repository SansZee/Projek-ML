"""Penjaga batas VRAM, pemilih micro-batch aman, dan jeda antar-run."""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

MAX_COOLDOWN_SECONDS = 300
DEFAULT_RESERVED_LIMIT_GB = 7.25


@dataclass(frozen=True)
class BatchPlan:
    """Konfigurasi micro-batch fisik dan gradient accumulation yang aman VRAM."""

    micro_batch_size: int
    grad_accum_steps: int
    effective_batch_size: int
    peak_allocated_gb: float
    peak_reserved_gb: float


def setup_vram_guard(memory_fraction: float = 0.92) -> torch.device:
    """Aktifkan pagar VRAM (~7,2–7,4 GB di RTX 5060 8 GB) untuk mencegah spill ke RAM."""
    if not torch.cuda.is_available():
        return torch.device("cpu")

    fraction = min(max(float(memory_fraction), 0.50), 0.95)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    return torch.device("cuda:0")


def preferred_amp_dtype(device: torch.device) -> torch.dtype:
    """Pilih ``bfloat16`` bila didukung native (RTX 5060), else ``float16``."""
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def vram_stats_gb() -> tuple[float, float]:
    """Kembalikan ``(max_allocated_gb, max_reserved_gb)`` saat ini."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    gib = 1024**3
    return (
        torch.cuda.max_memory_allocated() / gib,
        torch.cuda.max_memory_reserved() / gib,
    )


def cleanup_cuda(*objs: object) -> None:
    """Bebaskan referensi tensor/model secara agresif agar tidak bocor antar-split."""
    for obj in objs:
        if isinstance(obj, torch.nn.Module):
            obj.to("cpu")
    del objs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()


def select_safe_micro_batch(
    probe_fn: Callable[[int], None],
    *,
    effective_batch_size: int = 16,
    candidates: tuple[int, ...] = (16, 8, 4, 2, 1),
    reserved_limit_gb: float = DEFAULT_RESERVED_LIMIT_GB,
) -> BatchPlan:
    """Cari ``micro_batch_size`` terbesar yang menjaga ``max_memory_reserved()`` di batas aman."""
    if not torch.cuda.is_available():
        bs = min(effective_batch_size, candidates[0])
        return BatchPlan(bs, max(1, effective_batch_size // bs), effective_batch_size, 0.0, 0.0)

    filtered = tuple(c for c in candidates if c <= effective_batch_size) or (candidates[-1],)
    last_error: Exception | None = None

    for micro_bs in filtered:
        cleanup_cuda()
        try:
            probe_fn(micro_bs)
            alloc_gb, res_gb = vram_stats_gb()
            if res_gb <= reserved_limit_gb or micro_bs == filtered[-1]:
                accum = max(1, (effective_batch_size + micro_bs - 1) // micro_bs)
                return BatchPlan(
                    micro_batch_size=micro_bs,
                    grad_accum_steps=accum,
                    effective_batch_size=micro_bs * accum,
                    peak_allocated_gb=round(alloc_gb, 3),
                    peak_reserved_gb=round(res_gb, 3),
                )
        except torch.cuda.OutOfMemoryError as exc:
            last_error = exc
        finally:
            cleanup_cuda()

    raise RuntimeError(f"Semua kandidat micro-batch {filtered} melewati batas VRAM: {last_error}")


def cooldown(seconds: int, *, label: str = "") -> None:
    """Jeda singkat antar-training untuk mendinginkan laptop, dibatasi keras <= 5 menit."""
    wait_s = max(0, min(int(seconds), MAX_COOLDOWN_SECONDS))
    if wait_s <= 0:
        return
    tag = f" ({label})" if label else ""
    print(f"[cooldown] Jeda {wait_s} detik (maks {MAX_COOLDOWN_SECONDS}s){tag}...", flush=True)
    time.sleep(wait_s)
