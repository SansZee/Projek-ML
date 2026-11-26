"""Aplikasi Streamlit demo klasifikasi emosi multimodal + probabilitas per modalitas."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from backbones import MODEL_SPECS, extract_backbone_features
from config_loader import load_project_config
from models import build_fusion_model


def main() -> None:
    st.set_page_config(page_title="Klasifikasi Emosi Multimodal Film", layout="wide")
    st.title("Klasifikasi Emosi Multimodal Dialog Film")
    st.caption("Menampilkan prediksi emosi, probabilitas per modalitas, dan deteksi modalitas hilang.")

    cfg = load_project_config()
    ckpt_dir = cfg.runs_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
    if not ckpts:
        st.warning("Belum ada checkpoint model fusi di runs/checkpoints/. Jalankan training terlebih dahulu.")
        return

    chosen_ckpt = st.sidebar.selectbox("Pilih Model Checkpoint", ckpts, format_func=lambda p: p.stem)
    payload = torch.load(chosen_ckpt, map_location="cpu", weights_only=False)
    classes = tuple(payload["classes"])
    params = payload["params"]

    model = build_fusion_model(
        payload["arch"],
        payload["in_dims"],
        len(classes),
        proj_dim=int(params["proj_dim"]),
        num_layers=int(params["num_layers"]),
        num_heads=int(params["num_heads"]),
        dropout=0.0,
        modality_dropout=0.0,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()

    clips_csv = cfg.data_dir / "clips.csv"
    pool_csv = cfg.data_dir / "pool.csv"
    if not clips_csv.exists():
        st.error("data/clips.csv tidak ditemukan.")
        return

    clips = pd.read_csv(clips_csv)
    clips = clips[clips["status"].isin(("ok", "ada"))].copy()
    if pool_csv.exists():
        pool = pd.read_csv(pool_csv, usecols=["utterance_id", "text"]).drop_duplicates("utterance_id")
        clips = clips.merge(pool, on="utterance_id", how="left")
    else:
        clips["text"] = ""

    clip_id = st.selectbox("Pilih Klip Uji", clips["clip_id"].astype(str).tolist())
    row = clips[clips["clip_id"].astype(str) == clip_id].iloc[0]

    col_t, col_a, col_v = st.columns(3)
    use_text = col_t.checkbox("Gunakan Teks", value=True)
    use_audio = col_a.checkbox("Gunakan Audio", value=True)
    use_visual = col_v.checkbox("Gunakan Wajah/Video", value=True)

    missing_mods = [m for m, on in (("Teks", use_text), ("Audio", use_audio), ("Visual", use_visual)) if not on]
    if missing_mods:
        st.warning(f"Peringatan Modalitas Hilang: {', '.join(missing_mods)} tidak tersedia — model beradaptasi pada modalitas tersisa.")

    st.write(f"**Dialog:** {row.get('text', '')}")
    st.write(f"**Label Ground Truth:** `{row['label']}`")

    # Baca dari cache fitur bila ada
    split_guess = chosen_ckpt.stem.split("__")[-1]
    cache_candidates = {
        "text": [cfg.cache_dir / "finetuned" / split_guess / "text_deberta_large.pt", cfg.cache_dir / "frozen" / "text_deberta_base.pt"],
        "audio": [cfg.cache_dir / "finetuned" / split_guess / "audio_wavlm_large.pt", cfg.cache_dir / "frozen" / "audio_emotion2vec_large.pt", cfg.cache_dir / "frozen" / "audio_wavlm_base.pt"],
        "visual": [cfg.cache_dir / "finetuned" / split_guess / "visual_videomaev2_base.pt", cfg.cache_dir / "finetuned" / split_guess / "visual_convnext_base.pt", cfg.cache_dir / "frozen" / "visual_emotieff_b0.pt"],
    }

    inputs: dict[str, torch.Tensor] = {}
    for mod, enabled in (("text", use_text), ("audio", use_audio), ("visual", use_visual)):
        if not enabled:
            continue
        for cpath in cache_candidates[mod]:
            if cpath.exists():
                data = torch.load(cpath, map_location="cpu", weights_only=False)["feats"]
                if clip_id in data:
                    t = data[clip_id].float()
                    inputs[mod] = t.unsqueeze(0)
                    break

    if not inputs:
        st.info("Minimal 1 modalitas harus aktif dan memiliki cache fitur.")
        return

    with torch.no_grad():
        logits, details = model(inputs, {}, return_details=True)
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

    pred_label = classes[int(probs.argmax())]
    st.success(f"**Prediksi Emosi:** `{pred_label}` ({probs.max():.1%})")
    st.bar_chart(pd.DataFrame({"Probabilitas": probs}, index=classes))

    if "weights" in details and details["weights"]:
        st.subheader("Kontribusi Bobot Fusi per Modalitas")
        w_df = pd.DataFrame(
            {"Bobot": [float(details["weights"][m].item()) for m in details["weights"]]},
            index=list(details["weights"].keys()),
        )
        st.bar_chart(w_df)

    if "per_modality_logits" in details and details["per_modality_logits"]:
        st.subheader("Probabilitas Emosi per Modalitas")
        mod_probs = {
            m: torch.softmax(lg, dim=-1).squeeze(0).numpy()
            for m, lg in details["per_modality_logits"].items()
        }
        st.dataframe(pd.DataFrame(mod_probs, index=classes).style.format("{:.1%}"))


if __name__ == "__main__":
    main()
