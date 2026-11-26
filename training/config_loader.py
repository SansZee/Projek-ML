"""Loader konfigurasi bersama dari ``dataset-maker/config.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TRAINING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_ROOT.parent
DATASET_MAKER_DIR = REPO_ROOT / "dataset-maker"
DEFAULT_CONFIG_PATH = DATASET_MAKER_DIR / "config.yaml"

EVAL_SEEDS: tuple[int, ...] = (42, 123, 2026)
SPLIT_VARIANTS: tuple[str, ...] = ("split_A", "split_B", "split_C")
ABLATION_COMBOS: tuple[tuple[str, ...], ...] = (
    ("text",),
    ("audio",),
    ("visual",),
    ("text", "audio"),
    ("text", "visual"),
    ("audio", "visual"),
    ("text", "audio", "visual"),
)


@dataclass(frozen=True)
class ProjectConfig:
    """Konfigurasi terpusat agar training selaras dengan ``dataset-maker``."""

    raw: dict[str, Any]
    dataset_dir: Path
    data_dir: Path
    media_dir: Path
    runs_dir: Path
    cache_dir: Path
    classes: tuple[str, ...]
    seed: int
    optuna_trials: int
    batch_size: int
    epochs: int
    early_stop_patience: int
    lr: float
    primary_metric: str


def load_project_config(
    config_path: Path | str | None = None,
    *,
    runs_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> ProjectConfig:
    """Muat ``config.yaml`` milik ``dataset-maker`` dan petakan direktori training."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    base = path.parent
    paths = raw.get("paths", {})
    train = raw.get("train", {})
    labels = raw.get("labels", {})
    proj = raw.get("project", {})

    resolved_runs = Path(runs_dir) if runs_dir else (TRAINING_ROOT / "runs")
    resolved_cache = Path(cache_dir) if cache_dir else (TRAINING_ROOT / "cache")

    return ProjectConfig(
        raw=raw,
        dataset_dir=base,
        data_dir=(base / paths.get("data_dir", "data")).resolve(),
        media_dir=(base / paths.get("media_dir", "media")).resolve(),
        runs_dir=resolved_runs.resolve(),
        cache_dir=resolved_cache.resolve(),
        classes=tuple(str(c) for c in labels.get("classes", [])),
        seed=int(proj.get("seed", 42)),
        optuna_trials=int(train.get("optuna_trials", 15)),
        batch_size=int(train.get("batch_size", 16)),
        epochs=int(train.get("epochs", 30)),
        early_stop_patience=int(train.get("early_stop_patience", 5)),
        lr=float(train.get("lr", 1e-4)),
        primary_metric=str(train.get("primary_metric", "macro_f1")),
    )
