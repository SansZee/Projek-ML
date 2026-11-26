"""Kompilator tabel Bab IV (Hemat vs Maksimal, Δ A→B→C, 7 Ablasi) & grafik PNG."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config_loader import SPLIT_VARIANTS, ProjectConfig, load_project_config


def _collect_records(cfg: ProjectConfig) -> list[dict]:
    records: list[dict] = []
    exp_dir = cfg.runs_dir / "experiments"
    if exp_dir.exists():
        for path in sorted(exp_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                {
                    "name": data["name"],
                    "split": data["split"],
                    "f1": float(data["macro_f1_mean"]),
                    "f1_std": float(data["macro_f1_std"]),
                    "acc": float(data["accuracy_mean"]),
                    "acc_std": float(data["accuracy_std"]),
                    "raw": data,
                }
            )
    ft_dir = cfg.runs_dir / "finetuned_unimodal"
    if ft_dir.exists():
        for path in sorted(ft_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                {
                    "name": f"finetune_{data['model_key']}",
                    "split": data["split"],
                    "f1": float(data["test_macro_f1"]),
                    "f1_std": 0.0,
                    "acc": float(data["test_accuracy"]),
                    "acc_std": 0.0,
                    "raw": data,
                }
            )
    return records


def build_bab4_report(cfg: ProjectConfig | None = None) -> Path:
    """Buat ``runs/bab4_summary.md`` + Confusion Matrix & grafik ablasi 300 DPI."""
    cfg = cfg or load_project_config()
    records = _collect_records(cfg)
    plots_dir = cfg.runs_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    by_model: dict[str, dict[str, dict]] = {}
    for rec in records:
        by_model.setdefault(rec["name"], {})[rec["split"]] = rec

    lines = [
        "# Ringkasan Hasil Eksperimen untuk BAB IV",
        "",
        "## 1. Perbandingan Model Utama (`Split A`, `Split B`, `Split C` & Kebocoran `Δ`)",
        "",
        "| Model | Split A (Macro-F1) | Split B (Macro-F1) | Split C (Macro-F1) | Δ A→B | Δ B→C |",
        "|---|---|---|---|---|---|",
    ]

    split_c_candidates = []
    for name in sorted(by_model):
        if "ablation" in name:
            continue
        row_map = by_model[name]
        cells = []
        f1_vals = {}
        for sp in SPLIT_VARIANTS:
            if sp in row_map:
                r = row_map[sp]
                f1_vals[sp] = r["f1"]
                cells.append(f"{r['f1']:.4f} ± {r['f1_std']:.4f}")
            else:
                cells.append("-")
        d_ab = f"{f1_vals['split_A'] - f1_vals['split_B']:+.4f}" if {"split_A", "split_B"} <= set(f1_vals) else "-"
        d_bc = f"{f1_vals['split_B'] - f1_vals['split_C']:+.4f}" if {"split_B", "split_C"} <= set(f1_vals) else "-"
        lines.append(f"| `{name}` | {cells[0]} | {cells[1]} | {cells[2]} | {d_ab} | {d_bc} |")
        if "split_C" in row_map:
            split_c_candidates.append(row_map["split_C"])

    lines.extend(["", "## 2. Ablasi 7 Kombinasi Modalitas", "", "| Kombinasi | Split A | Split B | Split C |", "|---|---|---|---|"])
    for name in sorted(by_model):
        if "ablation" not in name:
            continue
        row_map = by_model[name]
        cells = [
            f"{row_map[sp]['f1']:.4f} ± {row_map[sp]['f1_std']:.4f}" if sp in row_map else "-"
            for sp in SPLIT_VARIANTS
        ]
        lines.append(f"| `{name}` | {cells[0]} | {cells[1]} | {cells[2]} |")

    if split_c_candidates:
        best = max(split_c_candidates, key=lambda r: r["f1"])
        raw = best["raw"]
        cm = np.asarray(raw.get("best_seed_confusion_matrix") or raw.get("confusion_matrix"), dtype=int)
        if cm.size > 0:
            fig, ax = plt.subplots(figsize=(7.5, 6.2))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=cfg.classes, yticklabels=cfg.classes, ax=ax,
            )
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Label Aktual")
            ax.set_title(f"Confusion Matrix Split C — {best['name']} (F1={best['f1']:.4f})")
            fig.tight_layout()
            fig.savefig(plots_dir / "confusion_matrix_best_splitC.png", dpi=300)
            plt.close(fig)

    # Grafik perbandingan Macro-F1 Split A vs B vs C model utama
    main_names = [n for n in sorted(by_model) if "ablation" not in n and "split_C" in by_model[n]]
    if main_names:
        main_names.sort(key=lambda n: by_model[n]["split_C"]["f1"])
        y = np.arange(len(main_names))
        fig, ax = plt.subplots(figsize=(9.5, max(4.2, 0.38 * len(main_names))))
        h = 0.26
        for idx, (sp, label, color) in enumerate(
            (("split_A", "Split A (Acak)", "#4c78a8"), ("split_B", "Split B (Per Film)", "#f58518"), ("split_C", "Split C (Per Aktor)", "#54a24b"))
        ):
            vals = [by_model[n][sp]["f1"] if sp in by_model[n] else 0.0 for n in main_names]
            ax.barh(y + (1 - idx) * h, vals, height=h, label=label, color=color)
        ax.set_yticks(y)
        ax.set_yticklabels(main_names, fontsize=8)
        ax.set_xlabel("Macro-F1")
        ax.set_title("Perbandingan Macro-F1 Model Utama (Split A vs B vs C)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(plots_dir / "model_comparison_splits.png", dpi=300)
        plt.close(fig)

    # Grafik ablasi 7 kombinasi bila sudah tersedia
    abl_names = [n for n in sorted(by_model) if "ablation" in n and "split_C" in by_model[n]]
    if abl_names:
        abl_names.sort(key=lambda n: by_model[n]["split_C"]["f1"])
        y = np.arange(len(abl_names))
        fig, ax = plt.subplots(figsize=(8.8, max(3.6, 0.42 * len(abl_names))))
        ax.barh(y, [by_model[n]["split_C"]["f1"] for n in abl_names], color="#4c78a8")
        ax.set_yticks(y)
        ax.set_yticklabels([n.split("__")[-1] for n in abl_names], fontsize=9)
        ax.set_xlabel("Macro-F1 (Split C - Per Aktor)")
        ax.set_title("Uji Ablasi 7 Kombinasi Modalitas pada Split C")
        fig.tight_layout()
        fig.savefig(plots_dir / "ablation_splitC.png", dpi=300)
        plt.close(fig)

    out_md = cfg.runs_dir / "bab4_summary.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[laporan] Ringkasan Bab IV ditulis ke {out_md}")
    return out_md


if __name__ == "__main__":
    build_bab4_report()
