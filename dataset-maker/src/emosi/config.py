"""Pembacaan dan validasi config.yaml.

Config adalah satu-satunya sumber ambang. Skrip tidak boleh punya nilai
default yang berbeda dari config, karena itu membuat dataset dari empat
laptop tidak bisa digabung.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP = (
    "project",
    "paths",
    "audition",
    "utterance",
    "asr",
    "qc",
    "labels",
    "selection",
    "split",
    "train",
)


class ConfigError(ValueError):
    """Config tidak valid."""


@dataclass(frozen=True)
class Config:
    path: Path
    raw: dict[str, Any]
    sha256: str

    @property
    def root(self) -> Path:
        """Folder tempat config.yaml berada (root projek)."""
        return self.path.parent

    def get(self, dotted: str, default: Any = None) -> Any:
        """Ambil nilai bertingkat, mis. ``cfg.get('audition.face_ratio_min')``."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is not None:
                    return default
                raise ConfigError(f"kunci config tidak ditemukan: {dotted}")
            node = node[part]
        return node

    def resolve(self, dotted: str) -> Path:
        """Ambil path dari config, relatif terhadap root projek."""
        value = Path(str(self.get(dotted))).expanduser()
        return value if value.is_absolute() else (self.root / value)

    def classes(self) -> list[str]:
        return list(self.get("labels.classes"))


def load_config(path: str | Path = "config.yaml") -> Config:
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise ConfigError(f"config tidak ditemukan: {cfg_path}")
    text = cfg_path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config harus berupa mapping YAML: {cfg_path}")
    missing = [key for key in REQUIRED_TOP if key not in raw]
    if missing:
        raise ConfigError("bagian wajib tidak ada di config: " + ", ".join(missing))
    return Config(path=cfg_path, raw=raw, sha256=hashlib.sha256(text.encode()).hexdigest())
