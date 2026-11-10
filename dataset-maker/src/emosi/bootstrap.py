"""Pemeriksaan environment: cetak dan simpan environment.lock.json.

Dijalankan setiap orang sebelum menyerahkan data:

    uv run python -m emosi.bootstrap --verify

Berkas yang dihasilkan dilampirkan bersama data, sehingga saat 20 film dari
empat laptop digabung ada bukti tertulis bahwa ambang dan versi memang sama.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .config import load_config


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ffmpeg_version() -> str | None:
    if shutil.which("ffmpeg") is None:
        return None
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        return out.stdout.splitlines()[0]
    except (subprocess.CalledProcessError, IndexError):
        return None


def torch_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"installed": False}

    info: dict = {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["cuda_runtime"] = torch.version.cuda
        info["device"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    return info


def collect(config_path: str | Path, root: Path) -> dict:
    cfg = load_config(config_path)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "config_path": str(cfg.path),
        "config_sha256": cfg.sha256,
        "uv_lock_sha256": sha256_file(root / "uv.lock"),
        "pyproject_sha256": sha256_file(root / "pyproject.toml"),
        "ffmpeg": ffmpeg_version(),
        "torch": torch_info(),
        "classes": cfg.classes(),
        "asr_model": cfg.get("asr.model"),
        "seed": cfg.get("project.seed"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--verify", action="store_true", help="cetak laporan dan simpan lock")
    parser.add_argument("--out", default="environment.lock.json")
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    root = config_path.parent
    info = collect(config_path, root)

    print("=" * 68)
    print("  environment check")
    print("=" * 68)
    rows = [
        ("python", info["python"]),
        ("platform", info["platform"]),
        ("config", f"{info['config_sha256'][:16]}…"),
        ("uv.lock", (info["uv_lock_sha256"] or "TIDAK ADA — jalankan uv lock")[:16]),
        ("ffmpeg", info["ffmpeg"] or "TIDAK ADA"),
    ]
    torch = info["torch"]
    if torch.get("installed"):
        rows.append(("torch", f"{torch['version']}  cuda={torch['cuda_available']}"))
        if torch.get("cuda_available"):
            rows.append(("gpu", f"{torch['device']} ({torch['vram_gb']} GB)"))
    else:
        rows.append(("torch", "belum dipasang (opsional, hanya untuk tahap model)"))

    for key, value in rows:
        print(f"  {key:<10} : {value}")
    print("=" * 68)

    problems = []
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        problems.append("ffmpeg/ffprobe tidak ada di PATH")
    if info["uv_lock_sha256"] is None:
        problems.append("uv.lock belum ada — jalankan `uv lock` lalu commit")

    if problems:
        print("PERINGATAN:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("Siap dipakai.")

    if args.verify:
        out_path = root / args.out
        out_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Ditulis: {out_path}")

    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
