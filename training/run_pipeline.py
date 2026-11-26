"""Master runner otomatis (Gelombang 1 Hemat -> Gelombang 2 Maksimal + Audit)."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from backbones import (
    FULL_SWEEP_EXTRA,
    MODEL_SPECS,
    PRIORITY_MODELS,
    extract_backbone_features,
    finetune_unimodal_backbone,
)
from config_loader import SPLIT_VARIANTS, load_project_config
from dataset import load_master_dataframe
from summarize_results import build_bab4_report
from train_experiments import (
    run_7_ablations,
    run_stacking_ensemble,
    train_audio_crnn,
    train_cached_experiment,
)
from vram import MAX_COOLDOWN_SECONDS, cleanup_cuda, cooldown, setup_vram_guard


def _fmt_eta(minutes: float) -> str:
    mins = max(0, int(round(minutes)))
    if mins >= 60:
        return f"{mins // 60}j {mins % 60:02d}m"
    return f"{mins} mnt"


@dataclass
class WaveTracker:
    """Pelacak progres & ETA dinamis per gelombang setelah setiap job selesai."""

    runs_dir: Path
    wave_weights: dict[str, list[float]] = field(default_factory=lambda: {"W1": [], "W2": []})
    wave_done_idx: dict[str, int] = field(default_factory=lambda: {"W1": 0, "W2": 0})
    wave_actual_min: dict[str, float] = field(default_factory=lambda: {"W1": 0.0, "W2": 0.0})
    wave_est_done: dict[str, float] = field(default_factory=lambda: {"W1": 0.0, "W2": 0.0})

    def register(self, wave: str, est_minutes: float) -> None:
        self.wave_weights.setdefault(wave, []).append(max(0.1, est_minutes))

    def _wave_eta_min(self, wave: str) -> float:
        weights = self.wave_weights.get(wave, [])
        idx = self.wave_done_idx.get(wave, 0)
        rem_est = sum(weights[idx:])
        est_done = self.wave_est_done.get(wave, 0.0)
        scale = (self.wave_actual_min.get(wave, 0.0) / est_done) if est_done > 1.0 else 1.0
        scale = min(max(scale, 0.35), 2.5)
        return rem_est * scale

    def record_done(self, wave: str, label: str, elapsed_min: float, ok: bool = True) -> None:
        weights = self.wave_weights.get(wave, [])
        idx = self.wave_done_idx.get(wave, 0)
        est_w = weights[idx] if idx < len(weights) else max(0.5, elapsed_min)
        self.wave_done_idx[wave] = idx + 1
        self.wave_actual_min[wave] = self.wave_actual_min.get(wave, 0.0) + elapsed_min
        self.wave_est_done[wave] = self.wave_est_done.get(wave, 0.0) + est_w

        total = max(1, len(weights))
        done = min(total, self.wave_done_idx[wave])
        pct = done * 100 // total
        eta_w = self._wave_eta_min(wave)
        eta_all = self._wave_eta_min("W1") + self._wave_eta_min("W2")
        status = f"SELESAI ({elapsed_min:.1f} mnt)" if ok else "GAGAL (lihat errors.log)"
        msg = (
            f"[{wave} {done}/{total} ({pct}%)] {status}: {label} "
            f"| ETA {wave}: ~{_fmt_eta(eta_w)} | ETA Total: ~{_fmt_eta(eta_all)}"
        )
        _log_progress(self.runs_dir, msg)
        (self.runs_dir / "eta_status.json").write_text(
            json.dumps(
                {
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "last_job": label,
                    "wave": wave,
                    "wave_progress": f"{done}/{total} ({pct}%)",
                    "eta_wave_min": round(eta_w, 1),
                    "eta_total_min": round(eta_all, 1),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _log_progress(runs_dir: Path, message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    with (runs_dir / "progress.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _safe_run(
    runs_dir: Path,
    label: str,
    fn,
    *args,
    tracker: WaveTracker | None = None,
    wave: str | None = None,
    **kwargs,
):
    """Isolasi error per job agar kegagalan 1 model tidak menghentikan antrean."""
    _log_progress(runs_dir, f"MULAI: {label}")
    t0 = time.time()
    try:
        out = fn(*args, **kwargs)
        dt_min = (time.time() - t0) / 60.0
        if tracker is not None and wave is not None:
            tracker.record_done(wave, label, dt_min, ok=True)
        else:
            _log_progress(runs_dir, f"SELESAI ({dt_min:.1f} mnt): {label}")
        return out
    except Exception:
        dt_min = (time.time() - t0) / 60.0
        err = traceback.format_exc()
        if tracker is not None and wave is not None:
            tracker.record_done(wave, label, dt_min, ok=False)
        else:
            _log_progress(runs_dir, f"GAGAL: {label} (lihat errors.log)")
        with (runs_dir / "errors.log").open("a", encoding="utf-8") as handle:
            handle.write(f"\n=== ERROR {datetime.now().isoformat()} :: {label} ===\n{err}\n")
        return None
    finally:
        cleanup_cuda()


def _run_smoke_test(cfg, df) -> int:
    """Uji cepat end-to-end (<3 menit) memakai subset kecil & model hemat + fusi."""
    from dataclasses import replace

    print("[smoke-test] Memulai uji alur lengkap dengan 24 klip sampel...")
    setup_vram_guard(0.92)
    sample = df.head(24).copy()
    sample["split_smoke"] = ["train"] * 12 + ["val"] * 6 + ["test"] * 6
    smoke_cfg = replace(cfg, runs_dir=cfg.runs_dir / "smoke")
    tmp_cache = cfg.cache_dir / "smoke"

    t_cache = extract_backbone_features(smoke_cfg, "text_roberta_base", sample, tmp_cache / "text.pt")
    a_cache = extract_backbone_features(smoke_cfg, "audio_wav2vec2_base", sample, tmp_cache / "audio.pt")
    v_cache = extract_backbone_features(smoke_cfg, "visual_emotieff_b0", sample, tmp_cache / "visual.pt")

    res = train_cached_experiment(
        smoke_cfg,
        name="smoke_gated_fusion",
        modality_caches={"text": t_cache, "audio": a_cache, "visual": v_cache},
        df=sample,
        split_col="split_smoke",
        arch="gated_cross_attn",
        n_trials=1,
        seeds=(42,),
        save_logits=False,
    )
    print(f"[smoke-test] SUKSES! Val/Test Macro-F1 smoke: {res['macro_f1_mean']:.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path ke dataset-maker/config.yaml")
    parser.add_argument("--wave", choices=("1", "2", "all"), default="all")
    parser.add_argument("--scale", choices=("priority", "full"), default="priority")
    parser.add_argument("--splits", nargs="+", default=list(SPLIT_VARIANTS))
    parser.add_argument(
        "--cooldown-s",
        type=int,
        default=60,
        help=f"Jeda antar-model berat dalam detik (maks {MAX_COOLDOWN_SECONDS}s = 5 menit)",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Uji cepat 24 klip tanpa splits.csv")
    parser.add_argument(
        "--extract-frozen-only",
        action="store_true",
        help="Ekstrak seluruh cache fitur beku tanpa menunggu splits.csv (0% kebocoran)",
    )
    args = parser.parse_args(argv)

    cooldown_s = max(0, min(int(args.cooldown_s), MAX_COOLDOWN_SECONDS))
    cfg = load_project_config(args.config)
    need_splits = not (args.smoke_test or args.extract_frozen_only)
    df = load_master_dataframe(cfg, require_splits=need_splits)

    if args.smoke_test:
        return _run_smoke_test(cfg, df)

    frozen_dir = cfg.cache_dir / "frozen"

    if args.extract_frozen_only:
        keys: list[str] = []
        if args.wave in ("1", "all"):
            keys += list(PRIORITY_MODELS["hemat_extract"])
            if args.scale == "full":
                keys += [k for k in FULL_SWEEP_EXTRA if MODEL_SPECS[k].tier == "hemat"]
        if args.wave in ("2", "all"):
            keys += list(PRIORITY_MODELS["maksimal_extract_only"])
        for key in keys:
            _safe_run(
                cfg.runs_dir,
                f"Frozen Extract {key}",
                extract_backbone_features,
                cfg,
                key,
                df,
                frozen_dir / f"{key}.pt",
            )
            cooldown(min(cooldown_s, 30), label=key)
        _log_progress(cfg.runs_dir, f"SELESAI ekstraksi beku ({len(keys)} model). Siap lanjut training begitu splits.csv ada.")
        return 0

    splits = [s for s in args.splits if s in df.columns]
    if not splits:
        print(f"Kolom split {args.splits} tidak ditemukan di splits.csv.")
        return 1

    tracker = WaveTracker(cfg.runs_dir)
    hemat_keys = list(PRIORITY_MODELS["hemat_extract"])
    if args.scale == "full":
        hemat_keys += [k for k in FULL_SWEEP_EXTRA if MODEL_SPECS[k].tier == "hemat"]
    ft_keys = list(PRIORITY_MODELS["maksimal_finetune"])
    if args.scale == "full":
        ft_keys += [k for k in FULL_SWEEP_EXTRA if MODEL_SPECS[k].tier == "maksimal"]

    if args.wave in ("1", "all"):
        for key in hemat_keys:
            tracker.register("W1", 0.05 if (frozen_dir / f"{key}.pt").exists() else 3.0)
        for _ in splits:
            tracker.register("W1", 2.5)  # audio_crnn
            for _ in hemat_keys:
                tracker.register("W1", 1.2)  # unimodal head + Optuna + 3 seeds
            tracker.register("W1", 1.2)  # hemat_early
            tracker.register("W1", 1.2)  # hemat_late
            tracker.register("W1", 5.0)  # 7 ablations

    if args.wave in ("2", "all"):
        for key in PRIORITY_MODELS["maksimal_extract_only"]:
            tracker.register("W2", 0.05 if (frozen_dir / f"{key}.pt").exists() else 3.5)
        for key in ft_keys:
            mod = MODEL_SPECS[key].modality
            est_ft = {"text": 6.0, "audio": 13.0, "visual": 18.0}.get(mod, 10.0)
            for _ in splits:
                tracker.register("W2", est_ft + cooldown_s / 60.0)
        for _ in splits:
            for _ in PRIORITY_MODELS["maksimal_extract_only"]:
                tracker.register("W2", 1.2)
            for _ in ("early", "late", "gated_cross_attn", "mult"):
                tracker.register("W2", 1.6)
            tracker.register("W2", 6.0)  # 7 ablations
            tracker.register("W2", 0.2)  # stacking ensemble

    _log_progress(
        cfg.runs_dir,
        f"PLAN START ({args.wave}/{args.scale}): ETA awal W1=~{_fmt_eta(tracker._wave_eta_min('W1'))} "
        f"| ETA awal W2=~{_fmt_eta(tracker._wave_eta_min('W2'))} "
        f"| Total=~{_fmt_eta(tracker._wave_eta_min('W1') + tracker._wave_eta_min('W2'))}",
    )

    # ── GELOMBANG 1: MODEL HEMAT (BASELINE CEPAT) ────────────────────────────
    if args.wave in ("1", "all"):
        for key in hemat_keys:
            _safe_run(
                cfg.runs_dir,
                f"W1 Extract {key}",
                extract_backbone_features,
                cfg,
                key,
                df,
                frozen_dir / f"{key}.pt",
                tracker=tracker,
                wave="W1",
            )
            if not (frozen_dir / f"{key}.pt").exists():
                cooldown(min(cooldown_s, 30), label=key)

        for split_col in splits:
            _safe_run(
                cfg.runs_dir,
                f"W1 Train audio_crnn ({split_col})",
                train_audio_crnn,
                cfg,
                df,
                split_col,
                tracker=tracker,
                wave="W1",
            )
            for key in hemat_keys:
                cpath = frozen_dir / f"{key}.pt"
                if cpath.exists():
                    mod = MODEL_SPECS[key].modality
                    _safe_run(
                        cfg.runs_dir,
                        f"W1 Unimodal {key} ({split_col})",
                        train_cached_experiment,
                        cfg,
                        name=f"unimodal_{key}",
                        modality_caches={mod: cpath},
                        df=df,
                        split_col=split_col,
                        arch="unimodal",
                        tracker=tracker,
                        wave="W1",
                    )

            hemat_pair = {
                "text": frozen_dir / "text_deberta_base.pt",
                "audio": frozen_dir / "audio_wavlm_base.pt",
                "visual": frozen_dir / "visual_emotieff_b0.pt",
            }
            if all(p.exists() for p in hemat_pair.values()):
                for arch in ("early", "late"):
                    _safe_run(
                        cfg.runs_dir,
                        f"W1 Fusion {arch} ({split_col})",
                        train_cached_experiment,
                        cfg,
                        name=f"hemat_{arch}",
                        modality_caches=hemat_pair,
                        df=df,
                        split_col=split_col,
                        arch=arch,
                        tracker=tracker,
                        wave="W1",
                    )
                _safe_run(
                    cfg.runs_dir,
                    f"W1 Ablation 7-kombinasi late ({split_col})",
                    run_7_ablations,
                    cfg,
                    arch="late",
                    modality_caches=hemat_pair,
                    df=df,
                    split_col=split_col,
                    prefix="hemat_ablation",
                    tracker=tracker,
                    wave="W1",
                )
        _safe_run(cfg.runs_dir, "Ringkas Sementara Pasca Gelombang 1", build_bab4_report, cfg)
        cooldown(cooldown_s, label="selesai Gelombang 1")

    # ── GELOMBANG 2: MODEL MAKSIMAL (FINE-TUNING 8 GB & FUSI PUNCAK) ─────────
    if args.wave in ("2", "all"):
        for key in PRIORITY_MODELS["maksimal_extract_only"]:
            _safe_run(
                cfg.runs_dir,
                f"W2 Extract {key}",
                extract_backbone_features,
                cfg,
                key,
                df,
                frozen_dir / f"{key}.pt",
                tracker=tracker,
                wave="W2",
            )

        for key in ft_keys:
            for split_col in splits:
                _safe_run(
                    cfg.runs_dir,
                    f"W2 Fine-tune {key} ({split_col})",
                    finetune_unimodal_backbone,
                    cfg,
                    key,
                    df,
                    split_col,
                    seed=cfg.seed,
                    tracker=tracker,
                    wave="W2",
                )
                cooldown(cooldown_s, label=f"{key} {split_col}")

        for split_col in splits:
            ft_dir = cfg.cache_dir / "finetuned" / split_col
            for key in PRIORITY_MODELS["maksimal_extract_only"]:
                cpath = frozen_dir / f"{key}.pt"
                if cpath.exists():
                    mod = MODEL_SPECS[key].modality
                    _safe_run(
                        cfg.runs_dir,
                        f"W2 Probe {key} ({split_col})",
                        train_cached_experiment,
                        cfg,
                        name=f"unimodal_{key}",
                        modality_caches={mod: cpath},
                        df=df,
                        split_col=split_col,
                        arch="unimodal",
                        tracker=tracker,
                        wave="W2",
                    )

            def _first_existing(*candidates: Path) -> Path | None:
                return next((p for p in candidates if p.exists()), None)

            best_t = _first_existing(
                ft_dir / "text_deberta_large.pt",
                ft_dir / "text_modernbert_large.pt",
                frozen_dir / "text_deberta_base.pt",
            )
            best_a = _first_existing(
                ft_dir / "audio_wavlm_large.pt",
                frozen_dir / "audio_emotion2vec_large.pt",
                ft_dir / "audio_wav2vec2_msp.pt",
                frozen_dir / "audio_wavlm_base.pt",
            )
            best_v = _first_existing(
                ft_dir / "visual_videomaev2_base.pt",
                ft_dir / "visual_convnext_base.pt",
                frozen_dir / "visual_emotieff_b2.pt",
                frozen_dir / "visual_emotieff_b0.pt",
            )
            max_pair = {m: p for m, p in (("text", best_t), ("audio", best_a), ("visual", best_v)) if p is not None}
            if len(max_pair) == 3:
                for arch in ("early", "late", "gated_cross_attn", "mult"):
                    _safe_run(
                        cfg.runs_dir,
                        f"W2 Fusion {arch} ({split_col})",
                        train_cached_experiment,
                        cfg,
                        name=f"maksimal_{arch}",
                        modality_caches=max_pair,
                        df=df,
                        split_col=split_col,
                        arch=arch,
                        tracker=tracker,
                        wave="W2",
                    )
                _safe_run(
                    cfg.runs_dir,
                    f"W2 Ablation 7-kombinasi gated_cross_attn ({split_col})",
                    run_7_ablations,
                    cfg,
                    arch="gated_cross_attn",
                    modality_caches=max_pair,
                    df=df,
                    split_col=split_col,
                    prefix="maksimal_ablation",
                    tracker=tracker,
                    wave="W2",
                )
                _safe_run(
                    cfg.runs_dir,
                    f"W2 Stacking Ensemble ({split_col})",
                    run_stacking_ensemble,
                    cfg,
                    model_names=[
                        "hemat_late",
                        "maksimal_early",
                        "maksimal_late",
                        "maksimal_gated_cross_attn",
                        "maksimal_mult",
                    ],
                    split_col=split_col,
                    tracker=tracker,
                    wave="W2",
                )

    _safe_run(cfg.runs_dir, "Ringkas Hasil & Grafik Bab IV", build_bab4_report, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
