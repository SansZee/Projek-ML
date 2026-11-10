from __future__ import annotations

from pathlib import Path

import pytest

from emosi.config import ConfigError, load_config
from emosi.media import parse_fps

ROOT = Path(__file__).resolve().parents[1]


def test_loads_project_config():
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.classes() == ["senang", "sedih", "marah", "takut", "netral"]
    assert cfg.get("audition.face_ratio_min") == 0.50
    assert len(cfg.sha256) == 64


def test_dotted_lookup_and_default():
    cfg = load_config(ROOT / "config.yaml")
    assert isinstance(cfg.get("asr.model"), str)
    assert cfg.get("tidak.ada", "cadangan") == "cadangan"


def test_missing_key_raises():
    cfg = load_config(ROOT / "config.yaml")
    with pytest.raises(ConfigError):
        cfg.get("tidak.ada")


def test_resolve_makes_paths_absolute():
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.resolve("paths.data_dir").is_absolute()


def test_parse_fps_handles_fractions():
    assert parse_fps("24000/1001") == pytest.approx(23.976, abs=1e-3)
    assert parse_fps("25/1") == 25.0
    assert parse_fps(None) == 0.0
