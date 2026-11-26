"""Pelatihan unimodal, 2D CRNN, fusi multimodal, Optuna tuning, 7 ablasi, & stacking."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from config_loader import ABLATION_COMBOS, EVAL_SEEDS, ProjectConfig
from dataset import CachedFeatureDataset, RawClipDataset, collate_cached_batch
from losses_metrics import build_loss_fn, evaluate_logits, export_audit_predictions
from models import CRNN2DAudio, UnimodalHead, build_fusion_model
from vram import cleanup_cuda, preferred_amp_dtype, setup_vram_guard

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_cache_map(cache_path: Path) -> tuple[dict[str, torch.Tensor], int]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    feats: dict[str, torch.Tensor] = payload["feats"]
    sample = next(iter(feats.values()))
    return feats, int(sample.shape[-1])


def _run_cached_training(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader | None,
    train_labels: np.ndarray,
    classes: tuple[str, ...],
    lr: float,
    weight_decay: float,
    loss_name: str,
    epochs: int,
    patience: int,
    device: torch.device,
    active_modalities: tuple[str, ...],
    is_unimodal: bool = False,
    trial: optuna.Trial | None = None,
) -> dict[str, object]:
    """Loop latih seragam untuk UnimodalHead maupun Fusi di atas tensor ter-cache."""
    criterion = build_loss_fn(loss_name, train_labels, len(classes), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / warmup_steps
        if s < warmup_steps
        else 0.5 * (1.0 + math.cos(math.pi * (s - warmup_steps) / max(1, total_steps - warmup_steps))),
    )
    amp_dtype = preferred_amp_dtype(device)

    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_val_eval: dict[str, object] | None = None
    no_improve = 0

    def _forward_batch(batch: dict[str, object]) -> torch.Tensor:
        if is_unimodal:
            mod = active_modalities[0]
            x = batch[mod].to(device)  # type: ignore[union-attr]
            pm = batch.get(f"{mod}_pad_mask")
            pm = pm.to(device) if pm is not None else None  # type: ignore[union-attr]
            return model(x, pm)
        inputs = {m: batch[m].to(device) for m in active_modalities if m in batch}  # type: ignore[union-attr]
        masks = {
            m: (batch[f"{m}_pad_mask"].to(device) if f"{m}_pad_mask" in batch else None)  # type: ignore[union-attr]
            for m in active_modalities
        }
        return model(inputs, masks)

    def _predict_loader(loader: DataLoader) -> tuple[np.ndarray, np.ndarray, list[str]]:
        model.eval()
        logits_all, targets_all, uids = [], [], []
        with torch.no_grad():
            for batch in loader:
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                    logits = _forward_batch(batch)
                logits_all.append(logits.float().cpu().numpy())
                targets_all.append(batch["label"].numpy())  # type: ignore[union-attr]
                uids.extend(list(batch["utterance_id"]))  # type: ignore[arg-type]
        return np.concatenate(logits_all), np.concatenate(targets_all), uids

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits = _forward_batch(batch)
                loss = criterion(logits, batch["label"].to(device))  # type: ignore[union-attr]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        v_logits, v_targets, _ = _predict_loader(val_loader)
        v_eval = evaluate_logits(v_logits, v_targets, classes)
        val_f1 = float(v_eval["macro_f1"])  # type: ignore[arg-type]

        if trial is not None:
            trial.report(val_f1, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_val_eval = v_eval
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    out: dict[str, object] = {
        "val_macro_f1": best_f1,
        "val_eval": best_val_eval,
        "best_state": best_state,
    }
    if test_loader is not None:
        t_logits, t_targets, t_uids = _predict_loader(test_loader)
        t_eval = evaluate_logits(t_logits, t_targets, classes)
        out["test_eval"] = t_eval
        out["test_logits"] = t_logits
        out["test_targets"] = t_targets
        out["test_uids"] = t_uids
        v_logits, v_targets, v_uids = _predict_loader(val_loader)
        out["val_logits"] = v_logits
        out["val_targets"] = v_targets
        out["val_uids"] = v_uids
    return out


def _aggregate_seed_results(
    *,
    tag: str,
    split_col: str,
    seed_runs: list[dict[str, object]],
    best_params: dict[str, object],
    classes: tuple[str, ...],
) -> dict[str, object]:
    f1s = [float(r["test_macro_f1"]) for r in seed_runs]
    accs = [float(r["test_accuracy"]) for r in seed_runs]
    best_run = max(seed_runs, key=lambda r: float(r["test_macro_f1"]))
    return {
        "name": tag,
        "split": split_col,
        "best_params": best_params,
        "macro_f1_mean": round(float(np.mean(f1s)), 4),
        "macro_f1_std": round(float(np.std(f1s, ddof=0)), 4),
        "accuracy_mean": round(float(np.mean(accs)), 4),
        "accuracy_std": round(float(np.std(accs, ddof=0)), 4),
        "seeds": seed_runs,
        "best_seed_per_class_f1": best_run["per_class_f1"],
        "best_seed_confusion_matrix": best_run["confusion_matrix"],
        "classes": list(classes),
    }


def train_cached_experiment(
    cfg: ProjectConfig,
    *,
    name: str,
    modality_caches: dict[str, Path],
    df: pd.DataFrame,
    split_col: str,
    arch: str = "unimodal",  # "unimodal" | "early" | "late" | "gated_cross_attn" | "mult"
    n_trials: int | None = None,
    seeds: tuple[int, ...] = EVAL_SEEDS,
    active_modalities: tuple[str, ...] | None = None,
    save_logits: bool = True,
) -> dict[str, object]:
    """Jalankan Optuna tuning (validasi) + re-evaluasi 3 seed di test set pada fitur ter-cache."""
    result_path = cfg.runs_dir / "experiments" / f"{name}__{split_col}.json"
    if result_path.exists():
        print(f"[resume] Lewati {name} ({split_col}): {result_path} sudah ada.")
        return json.loads(result_path.read_text(encoding="utf-8"))

    device = setup_vram_guard(0.92)
    caches: dict[str, dict[str, torch.Tensor]] = {}
    in_dims: dict[str, int] = {}
    for mod, path in modality_caches.items():
        feat_map, dim = _load_cache_map(path)
        caches[mod] = feat_map
        in_dims[mod] = dim

    used_mods = active_modalities or tuple(caches.keys())
    valid_df = df[df["clip_id"].astype(str).isin(set(next(iter(caches.values())).keys()))].copy()
    train_df = valid_df[valid_df[split_col] == "train"].copy()
    val_df = valid_df[valid_df[split_col] == "val"].copy()
    test_df = valid_df[valid_df[split_col] == "test"].copy()
    train_labels = train_df["label_idx"].to_numpy()

    def _make_loaders(bs: int, subset_mods: tuple[str, ...]) -> tuple[DataLoader, DataLoader, DataLoader]:
        sub_cache = {m: caches[m] for m in subset_mods}
        return (
            DataLoader(CachedFeatureDataset(train_df, sub_cache), batch_size=bs, shuffle=True, collate_fn=collate_cached_batch),
            DataLoader(CachedFeatureDataset(val_df, sub_cache), batch_size=bs, shuffle=False, collate_fn=collate_cached_batch),
            DataLoader(CachedFeatureDataset(test_df, sub_cache), batch_size=bs, shuffle=False, collate_fn=collate_cached_batch),
        )

    trials = n_trials if n_trials is not None else cfg.optuna_trials
    study_csv = cfg.runs_dir / "optuna" / f"{name}__{split_col}.csv"
    study_csv.parent.mkdir(parents=True, exist_ok=True)

    def _objective(trial: optuna.Trial) -> float:
        _set_seed(cfg.seed)
        lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        proj_dim = trial.suggest_categorical("proj_dim", [128, 256, 512])
        num_layers = trial.suggest_categorical("num_layers", [1, 2])
        num_heads = trial.suggest_categorical("num_heads", [4, 8])
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
        bs = trial.suggest_categorical("batch_size", [16, 32])
        loss_name = trial.suggest_categorical("loss_fn", ["weighted_ce", "label_smooth_0.1", "focal_gamma_2"])
        mod_drop = trial.suggest_float("modality_dropout", 0.0, 0.25) if arch != "unimodal" else 0.0

        tr_loader, va_loader, _ = _make_loaders(bs, used_mods)
        if arch == "unimodal":
            mod = used_mods[0]
            model = UnimodalHead(
                in_dims[mod], len(cfg.classes),
                proj_dim=proj_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout,
            ).to(device)
        else:
            model = build_fusion_model(
                arch, in_dims, len(cfg.classes),
                proj_dim=proj_dim, num_layers=num_layers, num_heads=num_heads,
                dropout=dropout, modality_dropout=mod_drop,
            ).to(device)

        res = _run_cached_training(
            model=model,
            train_loader=tr_loader,
            val_loader=va_loader,
            test_loader=None,
            train_labels=train_labels,
            classes=cfg.classes,
            lr=lr,
            weight_decay=weight_decay,
            loss_name=loss_name,
            epochs=cfg.epochs,
            patience=cfg.early_stop_patience,
            device=device,
            active_modalities=used_mods,
            is_unimodal=(arch == "unimodal"),
            trial=trial,
        )
        cleanup_cuda(model)
        return float(res["val_macro_f1"])  # type: ignore[arg-type]

    t_opt_start = time.time()

    def _optuna_cb(study_obj: optuna.Study, trial_obj: optuna.trial.FrozenTrial) -> None:
        done = len(study_obj.trials)
        elapsed_m = (time.time() - t_opt_start) / 60.0
        eta_m = (elapsed_m / done) * max(0, trials - done)
        best_val = study_obj.best_value if study_obj.best_trial is not None else 0.0
        print(
            f"[{name} | {split_col}] Optuna {done}/{trials} ({done*100//max(1,trials)}%) "
            f"best_val_f1={best_val:.4f} | ETA tuning=~{eta_m:.1f}m",
            flush=True,
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    study.optimize(_objective, n_trials=max(1, trials), callbacks=[_optuna_cb])
    study.trials_dataframe().to_csv(study_csv, index=False)
    best_params = dict(study.best_params)

    # Re-evaluasi 3 random seeds memakai hyperparameter terbaik
    seed_runs: list[dict[str, object]] = []
    tr_loader, va_loader, te_loader = _make_loaders(int(best_params["batch_size"]), used_mods)

    best_overall_f1 = -1.0
    best_ckpt_state = None
    t_seeds_start = time.time()

    for s_idx, seed in enumerate(seeds, start=1):
        _set_seed(seed)
        if arch == "unimodal":
            mod = used_mods[0]
            model = UnimodalHead(
                in_dims[mod], len(cfg.classes),
                proj_dim=int(best_params["proj_dim"]),
                num_layers=int(best_params["num_layers"]),
                num_heads=int(best_params["num_heads"]),
                dropout=float(best_params["dropout"]),
            ).to(device)
        else:
            model = build_fusion_model(
                arch, in_dims, len(cfg.classes),
                proj_dim=int(best_params["proj_dim"]),
                num_layers=int(best_params["num_layers"]),
                num_heads=int(best_params["num_heads"]),
                dropout=float(best_params["dropout"]),
                modality_dropout=float(best_params.get("modality_dropout", 0.0)),
            ).to(device)

        run_out = _run_cached_training(
            model=model,
            train_loader=tr_loader,
            val_loader=va_loader,
            test_loader=te_loader,
            train_labels=train_labels,
            classes=cfg.classes,
            lr=float(best_params["lr"]),
            weight_decay=float(best_params["weight_decay"]),
            loss_name=str(best_params["loss_fn"]),
            epochs=cfg.epochs,
            patience=cfg.early_stop_patience,
            device=device,
            active_modalities=used_mods,
            is_unimodal=(arch == "unimodal"),
        )
        t_eval = run_out["test_eval"]  # type: ignore[assignment]
        t_f1 = float(t_eval["macro_f1"])  # type: ignore[index]
        pred_csv = cfg.runs_dir / "predictions" / f"{name}__{split_col}__seed{seed}.csv"
        export_audit_predictions(run_out["test_uids"], t_eval["probs"], cfg.classes, pred_csv)  # type: ignore[arg-type,index]

        if save_logits:
            logit_dir = cfg.runs_dir / "logits" / f"{name}__{split_col}__seed{seed}.npz"
            logit_dir.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                logit_dir,
                val_logits=run_out["val_logits"],
                val_targets=run_out["val_targets"],
                test_logits=run_out["test_logits"],
                test_targets=run_out["test_targets"],
            )

        if t_f1 > best_overall_f1:
            best_overall_f1 = t_f1
            best_ckpt_state = run_out["best_state"]

        seed_runs.append(
            {
                "seed": seed,
                "val_macro_f1": round(float(run_out["val_macro_f1"]), 4),  # type: ignore[arg-type]
                "test_macro_f1": round(t_f1, 4),
                "test_accuracy": round(float(t_eval["accuracy"]), 4),  # type: ignore[index]
                "per_class_f1": t_eval["per_class_f1"],  # type: ignore[index]
                "confusion_matrix": t_eval["confusion_matrix"],  # type: ignore[index]
                "predictions_csv": str(pred_csv),
            }
        )
        s_elapsed = (time.time() - t_seeds_start) / 60.0
        s_eta = (s_elapsed / s_idx) * (len(seeds) - s_idx)
        print(
            f"[{name} | {split_col}] seed {s_idx}/{len(seeds)} ({s_idx*100//len(seeds)}%) "
            f"test_f1={t_f1:.4f} | ETA seeds=~{s_eta:.1f}m",
            flush=True,
        )
        cleanup_cuda(model)

    if best_ckpt_state is not None:
        ckpt_path = cfg.runs_dir / "checkpoints" / f"{name}__{split_col}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"arch": arch, "in_dims": in_dims, "params": best_params, "state_dict": best_ckpt_state, "classes": cfg.classes},
            ckpt_path,
        )

    summary = _aggregate_seed_results(
        tag=name, split_col=split_col, seed_runs=seed_runs, best_params=best_params, classes=cfg.classes
    )
    summary["optuna_csv"] = str(study_csv)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[selesai] {name} ({split_col}) | Macro-F1={summary['macro_f1_mean']:.4f}±{summary['macro_f1_std']:.4f} "
        f"| Acc={summary['accuracy_mean']:.4f}±{summary['accuracy_std']:.4f}"
    )
    return summary


def train_audio_crnn(
    cfg: ProjectConfig,
    df: pd.DataFrame,
    split_col: str,
    *,
    seeds: tuple[int, ...] = EVAL_SEEDS,
) -> dict[str, object]:
    """Latih 2D CRNN dari scratch di Log-Mel Spectrogram dan simpan cache fitur sekuensial."""
    result_path = cfg.runs_dir / "experiments" / f"audio_crnn__{split_col}.json"
    cache_out = cfg.cache_dir / "finetuned" / split_col / "audio_crnn.pt"
    if result_path.exists() and cache_out.exists():
        print(f"[resume] Lewati audio_crnn ({split_col}): sudah ada.")
        return json.loads(result_path.read_text(encoding="utf-8"))

    device = setup_vram_guard(0.92)
    amp_dtype = preferred_amp_dtype(device)
    train_df = df[df[split_col] == "train"].copy()
    val_df = df[df[split_col] == "val"].copy()
    test_df = df[df[split_col] == "test"].copy()
    train_labels = train_df["label_idx"].to_numpy()

    tr_loader = DataLoader(RawClipDataset(train_df, cfg.media_dir, modality="audio"), batch_size=16, shuffle=True, num_workers=2, persistent_workers=True)
    va_loader = DataLoader(RawClipDataset(val_df, cfg.media_dir, modality="audio"), batch_size=16, shuffle=False, num_workers=2, persistent_workers=True)
    te_loader = DataLoader(RawClipDataset(test_df, cfg.media_dir, modality="audio"), batch_size=16, shuffle=False, num_workers=2)

    seed_runs: list[dict[str, object]] = []
    best_model_state = None
    best_f1_overall = -1.0

    for s_idx, seed in enumerate(seeds, start=1):
        _set_seed(seed)
        t_crnn = time.time()
        model = CRNN2DAudio(len(cfg.classes)).to(device)
        criterion = build_loss_fn("weighted_ce", train_labels, len(cfg.classes), device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        best_val_f1, no_improve, best_state = -1.0, 0, None

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            for batch in tr_loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                    logits = model(batch["audio"].to(device))  # type: ignore[union-attr]
                    loss = criterion(logits, batch["label"].to(device))  # type: ignore[union-attr]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            v_log, v_tgt = [], []
            with torch.no_grad():
                for batch in va_loader:
                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                        logits = model(batch["audio"].to(device))  # type: ignore[union-attr]
                    v_log.append(logits.float().cpu().numpy())
                    v_tgt.append(batch["label"].numpy())  # type: ignore[union-attr]
            v_eval = evaluate_logits(np.concatenate(v_log), np.concatenate(v_tgt), cfg.classes)
            f1 = float(v_eval["macro_f1"])  # type: ignore[arg-type]
            el_m = (time.time() - t_crnn) / 60.0
            eta_m = (el_m / epoch) * (cfg.epochs - epoch)
            print(
                f"[audio_crnn | {split_col} s{s_idx}/{len(seeds)}] ep {epoch}/{cfg.epochs} "
                f"val_f1={f1:.4f} | ETA=~{eta_m:.1f}m",
                flush=True,
            )
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.early_stop_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        t_log, t_tgt, t_uids = [], [], []
        with torch.no_grad():
            for batch in te_loader:
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                    logits = model(batch["audio"].to(device))  # type: ignore[union-attr]
                t_log.append(logits.float().cpu().numpy())
                t_tgt.append(batch["label"].numpy())  # type: ignore[union-attr]
                t_uids.extend(list(batch["utterance_id"]))  # type: ignore[arg-type]
        t_eval = evaluate_logits(np.concatenate(t_log), np.concatenate(t_tgt), cfg.classes)
        pred_csv = cfg.runs_dir / "predictions" / f"audio_crnn__{split_col}__seed{seed}.csv"
        export_audit_predictions(t_uids, t_eval["probs"], cfg.classes, pred_csv)  # type: ignore[arg-type]
        t_f1 = float(t_eval["macro_f1"])  # type: ignore[arg-type]
        if t_f1 > best_f1_overall:
            best_f1_overall = t_f1
            best_model_state = best_state
        seed_runs.append(
            {
                "seed": seed,
                "val_macro_f1": round(best_val_f1, 4),
                "test_macro_f1": round(t_f1, 4),
                "test_accuracy": round(float(t_eval["accuracy"]), 4),  # type: ignore[arg-type]
                "per_class_f1": t_eval["per_class_f1"],
                "confusion_matrix": t_eval["confusion_matrix"],
                "predictions_csv": str(pred_csv),
            }
        )
        cleanup_cuda(model, optimizer, criterion)

    # Ekstrak sequence features untuk cache fusi bila diperlukan
    if best_model_state is not None and not cache_out.exists():
        model = CRNN2DAudio(len(cfg.classes)).to(device)
        model.load_state_dict(best_model_state)
        model.eval()
        all_loader = DataLoader(RawClipDataset(df, cfg.media_dir, modality="audio"), batch_size=32, shuffle=False, num_workers=2)
        feat_map: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for batch in all_loader:
                seq = model.encode_sequence(batch["audio"].to(device)).to(torch.float16).cpu()  # type: ignore[union-attr]
                for i, cid in enumerate(batch["clip_id"]):  # type: ignore[arg-type]
                    feat_map[str(cid)] = seq[i].contiguous()
        cache_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_key": "audio_crnn", "feats": feat_map}, cache_out)
        cleanup_cuda(model)

    summary = _aggregate_seed_results(
        tag="audio_crnn", split_col=split_col, seed_runs=seed_runs, best_params={"lr": 3e-4}, classes=cfg.classes
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_7_ablations(
    cfg: ProjectConfig,
    *,
    arch: str,
    modality_caches: dict[str, Path],
    df: pd.DataFrame,
    split_col: str,
    prefix: str = "ablation",
) -> list[dict[str, object]]:
    """Uji seluruh 7 kombinasi ablasi modalitas pada arsitektur fusi terbaik."""
    results = []
    for combo in ABLATION_COMBOS:
        tag = f"{prefix}_{arch}__{'+'.join(combo)}"
        sub_caches = {m: modality_caches[m] for m in combo}
        res = train_cached_experiment(
            cfg,
            name=tag,
            modality_caches=sub_caches,
            df=df,
            split_col=split_col,
            arch=arch,
            n_trials=min(8, cfg.optuna_trials),
            active_modalities=combo,
            save_logits=False,
        )
        results.append(res)
    return results


def run_stacking_ensemble(
    cfg: ProjectConfig,
    *,
    model_names: list[str],
    split_col: str,
    seeds: tuple[int, ...] = EVAL_SEEDS,
) -> dict[str, object] | None:
    """Out-of-Fold / Val-tuned Stacking Ensemble menggabungkan logit Hemat + Maksimal."""
    out_path = cfg.runs_dir / "experiments" / f"stacking_ensemble__{split_col}.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))

    seed_runs: list[dict[str, object]] = []
    for seed in seeds:
        x_val_list, x_test_list = [], []
        y_val = y_test = None
        for mname in model_names:
            npz_path = cfg.runs_dir / "logits" / f"{mname}__{split_col}__seed{seed}.npz"
            if not npz_path.exists():
                continue
            data = np.load(npz_path)
            x_val_list.append(torch.softmax(torch.from_numpy(data["val_logits"]), dim=-1).numpy())
            x_test_list.append(torch.softmax(torch.from_numpy(data["test_logits"]), dim=-1).numpy())
            y_val = data["val_targets"]
            y_test = data["test_targets"]
        if len(x_val_list) < 2 or y_val is None or y_test is None:
            continue

        x_val = np.concatenate(x_val_list, axis=1)
        x_test = np.concatenate(x_test_list, axis=1)
        meta = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=seed)
        meta.fit(x_val, y_val)
        test_probs = meta.predict_proba(x_test)
        test_logits = np.log(np.clip(test_probs, 1e-7, 1.0))
        t_eval = evaluate_logits(test_logits, y_test, cfg.classes)
        seed_runs.append(
            {
                "seed": seed,
                "val_macro_f1": round(float(evaluate_logits(np.log(np.clip(meta.predict_proba(x_val), 1e-7, 1.0)), y_val, cfg.classes)["macro_f1"]), 4),  # type: ignore[arg-type]
                "test_macro_f1": round(float(t_eval["macro_f1"]), 4),  # type: ignore[arg-type]
                "test_accuracy": round(float(t_eval["accuracy"]), 4),  # type: ignore[arg-type]
                "per_class_f1": t_eval["per_class_f1"],
                "confusion_matrix": t_eval["confusion_matrix"],
            }
        )

    if not seed_runs:
        return None
    summary = _aggregate_seed_results(
        tag="stacking_ensemble",
        split_col=split_col,
        seed_runs=seed_runs,
        best_params={"base_models": model_names},
        classes=cfg.classes,
    )
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
