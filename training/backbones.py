"""Registry model Hemat & Maksimal, ekstraksi fitur cache, dan fine-tuning selektif 8 GB VRAM."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, VideoMAEModel

from config_loader import ProjectConfig
from dataset import RawClipDataset
from losses_metrics import build_loss_fn, evaluate_logits
from models import AttentiveStatsPool
from vram import cleanup_cuda, preferred_amp_dtype, select_safe_micro_batch, setup_vram_guard


@dataclass(frozen=True)
class BackboneSpec:
    """Spesifikasi terstandar untuk setiap encoder Teks, Audio, dan Visual."""

    key: str
    modality: str       # "text" | "audio" | "visual"
    tier: str           # "hemat" | "maksimal"
    backend: str        # "hf_text" | "hf_audio" | "funasr" | "timm" | "emotieff" | "videomae" | "crnn"
    model_id: str
    use_lora: bool = False
    lora_targets: tuple[str, ...] = ()
    default_effective_bs: int = 16
    candidates: tuple[int, ...] = (16, 8, 4, 2, 1)


MODEL_SPECS: dict[str, BackboneSpec] = {
    # ── Teks ─────────────────────────────────────────────────────────────────
    "text_roberta_base": BackboneSpec(
        "text_roberta_base", "text", "hemat", "hf_text", "FacebookAI/roberta-base",
        use_lora=True, lora_targets=("query", "value"), candidates=(16, 8, 4),
    ),
    "text_deberta_base": BackboneSpec(
        "text_deberta_base", "text", "hemat", "hf_text", "microsoft/deberta-v3-base",
        use_lora=True, lora_targets=("query_proj", "value_proj"), candidates=(16, 8, 4),
    ),
    "text_distilroberta_emo": BackboneSpec(
        "text_distilroberta_emo", "text", "hemat", "hf_text", "j-hartmann/emotion-english-distilroberta-base",
        use_lora=True, lora_targets=("query", "value"), candidates=(16, 8, 4),
    ),
    "text_deberta_large": BackboneSpec(
        "text_deberta_large", "text", "maksimal", "hf_text", "microsoft/deberta-v3-large",
        use_lora=True, lora_targets=("query_proj", "value_proj"), candidates=(8, 4, 2),
    ),
    "text_modernbert_large": BackboneSpec(
        "text_modernbert_large", "text", "maksimal", "hf_text", "answerdotai/ModernBERT-large",
        use_lora=True, lora_targets=("Wqkv", "Wo"), candidates=(8, 4, 2),
    ),
    "text_roberta_large_emo": BackboneSpec(
        "text_roberta_large_emo", "text", "maksimal", "hf_text", "cardiffnlp/twitter-roberta-large-emotion-latest",
        use_lora=True, lora_targets=("query", "value"), candidates=(8, 4, 2),
    ),
    # ── Audio / SER ──────────────────────────────────────────────────────────
    "audio_crnn": BackboneSpec(
        "audio_crnn", "audio", "hemat", "crnn", "crnn_2d_mel", candidates=(16, 8, 4),
    ),
    "audio_wav2vec2_base": BackboneSpec(
        "audio_wav2vec2_base", "audio", "hemat", "hf_audio", "facebook/wav2vec2-base-960h",
        use_lora=True, lora_targets=("q_proj", "v_proj"), candidates=(8, 4, 2),
    ),
    "audio_hubert_base": BackboneSpec(
        "audio_hubert_base", "audio", "hemat", "hf_audio", "facebook/hubert-base-ls960",
        use_lora=True, lora_targets=("q_proj", "v_proj"), candidates=(8, 4, 2),
    ),
    "audio_wavlm_base": BackboneSpec(
        "audio_wavlm_base", "audio", "hemat", "hf_audio", "microsoft/wavlm-base-plus",
        use_lora=True, lora_targets=("q_proj", "v_proj"), candidates=(8, 4, 2),
    ),
    "audio_emotion2vec_base": BackboneSpec(
        "audio_emotion2vec_base", "audio", "hemat", "funasr", "emotion2vec/emotion2vec_plus_base",
        candidates=(8, 4, 2),
    ),
    "audio_emotion2vec_large": BackboneSpec(
        "audio_emotion2vec_large", "audio", "maksimal", "funasr", "emotion2vec/emotion2vec_plus_large",
        candidates=(8, 4, 2),
    ),
    "audio_wavlm_large": BackboneSpec(
        "audio_wavlm_large", "audio", "maksimal", "hf_audio", "microsoft/wavlm-large",
        use_lora=True, lora_targets=("q_proj", "v_proj"), candidates=(4, 2, 1),
    ),
    "audio_wav2vec2_msp": BackboneSpec(
        "audio_wav2vec2_msp", "audio", "maksimal", "hf_audio", "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        use_lora=True, lora_targets=("q_proj", "v_proj"), candidates=(8, 4, 2),
    ),
    # ── Visual / FER ─────────────────────────────────────────────────────────
    "visual_resnet50": BackboneSpec(
        "visual_resnet50", "visual", "hemat", "timm", "resnet50", candidates=(8, 4, 2),
    ),
    "visual_effnet_b0": BackboneSpec(
        "visual_effnet_b0", "visual", "hemat", "timm", "efficientnet_b0", candidates=(16, 8, 4),
    ),
    "visual_emotieff_b0": BackboneSpec(
        "visual_emotieff_b0", "visual", "hemat", "emotieff", "enet_b0_8_best_afew", candidates=(16, 8, 4),
    ),
    "visual_vit_base": BackboneSpec(
        "visual_vit_base", "visual", "hemat", "timm", "vit_base_patch16_224", candidates=(8, 4, 2),
    ),
    "visual_convnext_tiny": BackboneSpec(
        "visual_convnext_tiny", "visual", "hemat", "timm", "convnext_tiny", candidates=(8, 4, 2),
    ),
    "visual_videomaev2_base": BackboneSpec(
        "visual_videomaev2_base", "visual", "maksimal", "videomae", "MCG-NJU/videomae-base",
        candidates=(4, 2, 1),
    ),
    "visual_convnext_base": BackboneSpec(
        "visual_convnext_base", "visual", "maksimal", "timm", "convnext_base", candidates=(4, 2, 1),
    ),
    "visual_emotieff_b2": BackboneSpec(
        "visual_emotieff_b2", "visual", "maksimal", "emotieff", "enet_b2_8", candidates=(8, 4, 2),
    ),
}

PRIORITY_MODELS: dict[str, tuple[str, ...]] = {
    "hemat_extract": (
        "text_roberta_base",
        "text_deberta_base",
        "text_distilroberta_emo",
        "audio_wav2vec2_base",
        "audio_wavlm_base",
        "visual_resnet50",
        "visual_emotieff_b0",
        "visual_vit_base",
    ),
    "maksimal_finetune": (
        "text_deberta_large",
        "text_modernbert_large",
        "audio_wavlm_large",
        "audio_wav2vec2_msp",
        "visual_convnext_base",
        "visual_videomaev2_base",
    ),
    "maksimal_extract_only": (
        "audio_emotion2vec_large",
        "visual_emotieff_b2",
    ),
}

FULL_SWEEP_EXTRA: tuple[str, ...] = (
    "text_roberta_large_emo",
    "audio_hubert_base",
    "audio_emotion2vec_base",
    "visual_effnet_b0",
    "visual_convnext_tiny",
)


def _normalize_video_batch(video: torch.Tensor, *, size: int = 224) -> torch.Tensor:
    """Konversi ``uint8 [B, T, C, H, W]`` ke tensor float terstandarisasi ImageNet."""
    x = video.float() / 255.0
    b, t, c, h, w = x.shape
    if h != size or w != size:
        x = F.interpolate(x.view(b * t, c, h, w), size=(size, size), mode="bilinear", align_corners=False)
        x = x.view(b, t, c, size, size)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 1, 3, 1, 1)
    return (x - mean) / std


def _enable_grad_ckpt(module: nn.Module) -> None:
    if hasattr(module, "gradient_checkpointing_enable"):
        try:
            module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            module.gradient_checkpointing_enable()


def _unfreeze_tail_blocks(module: nn.Module, *, n_blocks: int = 2) -> None:
    """Bekukan seluruh parameter kecuali ``n_blocks`` blok terakhir + norm akhir."""
    for p in module.parameters():
        p.requires_grad = False
    for attr in ("blocks", "stages", "layers", "layer", "encoder"):
        if hasattr(module, attr):
            container = getattr(module, attr)
            if isinstance(container, (nn.ModuleList, nn.Sequential)) and len(container) > 0:
                for block in list(container)[-n_blocks:]:
                    for p in block.parameters():
                        p.requires_grad = True
                break
    for attr in ("norm", "fc_norm", "head_norm", "layer4"):
        if hasattr(module, attr):
            for p in getattr(module, attr).parameters():
                p.requires_grad = True
    if not any(p.requires_grad for p in module.parameters()):
        # Fallback: unfreeze 25% parameter terakhir
        params = list(module.parameters())
        for p in params[-max(1, len(params) // 4) :]:
            p.requires_grad = True


class UnimodalBackboneWrapper(nn.Module):
    """Pembungkus seragam encoder pralatih + kepala klasifikasi untuk fine-tuning."""

    def __init__(
        self,
        spec: BackboneSpec,
        num_classes: int,
        *,
        for_finetune: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.tokenizer = None
        self.image_size = 224

        if spec.backend == "hf_text":
            self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
            encoder = AutoModel.from_pretrained(spec.model_id)
            hidden_dim = int(encoder.config.hidden_size)
            if for_finetune:
                _enable_grad_ckpt(encoder)
                if spec.use_lora and spec.lora_targets:
                    try:
                        lora_cfg = LoraConfig(
                            task_type=TaskType.FEATURE_EXTRACTION,
                            r=16,
                            lora_alpha=32,
                            lora_dropout=0.1,
                            target_modules=list(spec.lora_targets),
                        )
                        encoder = get_peft_model(encoder, lora_cfg)
                    except Exception:
                        _unfreeze_tail_blocks(encoder, n_blocks=4)
                else:
                    _unfreeze_tail_blocks(encoder, n_blocks=4)
            self.encoder = encoder
        elif spec.backend == "hf_audio":
            encoder = AutoModel.from_pretrained(spec.model_id)
            hidden_dim = int(encoder.config.hidden_size)
            if hasattr(encoder, "feature_extractor"):
                for p in encoder.feature_extractor.parameters():
                    p.requires_grad = False
            if for_finetune:
                _enable_grad_ckpt(encoder)
                _unfreeze_tail_blocks(encoder.encoder if hasattr(encoder, "encoder") else encoder, n_blocks=4)
            self.encoder = encoder
        elif spec.backend == "timm":
            encoder = timm.create_model(spec.model_id, pretrained=True, num_classes=0)
            hidden_dim = int(encoder.num_features)
            if for_finetune:
                if hasattr(encoder, "set_grad_checkpointing"):
                    encoder.set_grad_checkpointing(True)
                _unfreeze_tail_blocks(encoder, n_blocks=2)
            self.encoder = encoder
        elif spec.backend == "emotieff":
            from emotiefflib.facial_analysis import EmotiEffLibRecognizerTorch

            rec = EmotiEffLibRecognizerTorch(model_name=spec.model_id, device="cpu")
            timm_arch = "tf_efficientnet_b2.ns_jft_in1k" if "b2" in spec.model_id else "tf_efficientnet_b0.ns_jft_in1k"
            encoder = timm.create_model(timm_arch, pretrained=False, num_classes=0)
            encoder.load_state_dict(rec.model.state_dict(), strict=False)
            self.image_size = 260 if "b2" in spec.model_id else 224
            hidden_dim = int(encoder.num_features)
            if for_finetune:
                if hasattr(encoder, "set_grad_checkpointing"):
                    encoder.set_grad_checkpointing(True)
                _unfreeze_tail_blocks(encoder, n_blocks=2)
            self.encoder = encoder
        elif spec.backend == "videomae":
            encoder = VideoMAEModel.from_pretrained(spec.model_id)
            hidden_dim = int(encoder.config.hidden_size)
            if for_finetune:
                _enable_grad_ckpt(encoder)
                _unfreeze_tail_blocks(encoder.encoder if hasattr(encoder, "encoder") else encoder, n_blocks=3)
            self.encoder = encoder
        else:
            raise ValueError(f"Backend {spec.backend} ditangani terpisah, bukan UnimodalBackboneWrapper.")

        self.hidden_dim = hidden_dim
        self.pool = AttentiveStatsPool(hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def encode_sequence(
        self,
        batch: dict[str, object],
        device: torch.device,
        *,
        max_len: int = 128,
        audio_stride: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Kembalikan representasi sekuensial ``[B, T, D]`` dan ``pad_mask`` opsional."""
        if self.spec.backend == "hf_text":
            assert self.tokenizer is not None
            toks = self.tokenizer(
                list(batch["text"]),  # type: ignore[arg-type]
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)
            out = self.encoder(**toks)
            pad_mask = toks["attention_mask"] == 0
            return out.last_hidden_state, pad_mask

        if self.spec.backend == "hf_audio":
            wav = batch["audio"].to(device)  # type: ignore[union-attr]
            out = self.encoder(wav)
            seq = out.last_hidden_state
            if audio_stride > 1 and seq.shape[1] > audio_stride:
                seq = seq[:, ::audio_stride, :]
            return seq, None

        if self.spec.backend in ("timm", "emotieff"):
            video = batch["video"].to(device)  # type: ignore[union-attr]
            x = _normalize_video_batch(video, size=self.image_size)
            b, t, c, h, w = x.shape
            feats = self.encoder(x.view(b * t, c, h, w))
            return feats.view(b, t, -1), None

        if self.spec.backend == "videomae":
            video = batch["video"].to(device)  # type: ignore[union-attr]
            x = _normalize_video_batch(video, size=224)
            out = self.encoder(pixel_values=x)
            # Token ruang-waktu [B, 1568, D] diagregasi per 8 frame waktu -> [B, 8, D]
            seq = out.last_hidden_state
            b, n_tok, d = seq.shape
            t_tube = max(1, n_tok // 196)
            if t_tube * 196 == n_tok:
                seq = seq.view(b, t_tube, 196, d).mean(dim=2)
            return seq, None

        raise RuntimeError(f"Unsupported backend: {self.spec.backend}")

    def forward(self, batch: dict[str, object], device: torch.device) -> torch.Tensor:
        seq, pad_mask = self.encode_sequence(batch, device)
        pooled = self.pool(seq.float(), pad_mask)
        return self.head(pooled)


def _extract_funasr_features(
    spec: BackboneSpec,
    df: pd.DataFrame,
    media_dir: Path,
    out_path: Path,
) -> Path:
    """Ekstrak embedding 50 Hz / pooled dari ``emotion2vec_plus_*`` via FunASR."""
    from funasr import AutoModel as FunASRModel

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = None
    for hub in ("hf", "ms"):
        try:
            model = FunASRModel(model=spec.model_id, hub=hub, device=device_str, disable_update=True)
            break
        except Exception:
            continue
    if model is None:
        raise RuntimeError(f"Gagal memuat FunASR model {spec.model_id}")

    cache: dict[str, torch.Tensor] = {}
    for row in tqdm(df.itertuples(), total=len(df), desc=f"extract:{spec.key}", unit="klip"):
        audio_rel = str(getattr(row, "audio_path", "") or f"audio/{row.clip_id}.wav")
        wav_path = str(media_dir / audio_rel)
        res = model.generate(wav_path, granularity="frame", extract_embedding=True)
        feats = res[0]["feats"]
        tensor = torch.as_tensor(feats, dtype=torch.float16)
        if tensor.ndim == 2 and tensor.shape[0] > 4:
            tensor = tensor[::2]  # ~25 Hz agar ukuran cache tetap ringkas
        cache[str(row.clip_id)] = tensor.cpu().contiguous()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_key": spec.key, "feats": cache}, out_path)
    cleanup_cuda(model)
    return out_path


def extract_backbone_features(
    cfg: ProjectConfig,
    model_key: str,
    df: pd.DataFrame,
    out_path: Path,
    *,
    wrapper: UnimodalBackboneWrapper | None = None,
    batch_size: int = 16,
    num_workers: int = 2,
) -> Path:
    """Ekstrak & simpan fitur sekuensial fp16 ke ``out_path`` (auto-skip bila sudah lengkap)."""
    if out_path.exists():
        existing = torch.load(out_path, map_location="cpu", weights_only=False)
        if len(existing.get("feats", {})) >= len(df):
            print(f"[cache] Lewati ekstraksi {model_key}: sudah ada di {out_path}")
            return out_path

    spec = MODEL_SPECS[model_key]
    if spec.backend == "funasr":
        return _extract_funasr_features(spec, df, cfg.media_dir, out_path)

    device = setup_vram_guard(0.92)
    amp_dtype = preferred_amp_dtype(device)
    owns_wrapper = wrapper is None
    if wrapper is None:
        wrapper = UnimodalBackboneWrapper(spec, len(cfg.classes), for_finetune=False)
    wrapper.to(device).eval()

    ds = RawClipDataset(df, cfg.media_dir, modality=spec.modality)
    loader_bs = min(batch_size, spec.candidates[0] * 2)
    loader = DataLoader(
        ds,
        batch_size=loader_bs,
        shuffle=False,
        num_workers=0 if spec.modality == "text" else num_workers,
        pin_memory=False,
    )

    cache: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"extract:{model_key}", unit="batch"):
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                seq, pad_mask = wrapper.encode_sequence(batch, device)
            seq_cpu = seq.to(torch.float16).cpu()
            pad_cpu = pad_mask.cpu() if pad_mask is not None else None
            for i, cid in enumerate(batch["clip_id"]):  # type: ignore[arg-type]
                t = seq_cpu[i]
                if pad_cpu is not None:
                    valid_len = int((~pad_cpu[i]).sum().item())
                    t = t[: max(1, valid_len)]
                cache[str(cid)] = t.contiguous()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_key": model_key, "feats": cache}, out_path)
    if owns_wrapper:
        cleanup_cuda(wrapper)
    return out_path


def finetune_unimodal_backbone(
    cfg: ProjectConfig,
    model_key: str,
    df: pd.DataFrame,
    split_col: str,
    *,
    seed: int = 42,
    epochs: int = 8,
    patience: int = 3,
    lr_backbone: float = 2e-5,
    head_lr_mult: float = 10.0,
    weight_decay: float = 0.01,
    loss_name: str = "weighted_ce",
) -> dict[str, object]:
    """Fine-tune encoder per split (hanya di set ``train``) + ekstrak fitur bebas kebocoran."""
    spec = MODEL_SPECS[model_key]
    cache_out = cfg.cache_dir / "finetuned" / split_col / f"{model_key}.pt"
    metric_out = cfg.runs_dir / "finetuned_unimodal" / f"{model_key}__{split_col}.json"
    if cache_out.exists() and metric_out.exists():
        import json

        print(f"[resume] Lewati fine-tune {model_key} ({split_col}): sudah selesai.")
        return json.loads(metric_out.read_text(encoding="utf-8"))

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = setup_vram_guard(0.92)
    amp_dtype = preferred_amp_dtype(device)

    train_df = df[df[split_col] == "train"].copy()
    val_df = df[df[split_col] == "val"].copy()
    test_df = df[df[split_col] == "test"].copy()

    # Skenario terberat untuk uji aman VRAM: urutkan dari durasi/teks terpanjang
    probe_df = train_df.sort_values("duration_s", ascending=False).head(max(16, spec.candidates[0]))
    probe_ds = RawClipDataset(probe_df, cfg.media_dir, modality=spec.modality)

    def _clean_probe(micro_bs: int) -> None:
        w = UnimodalBackboneWrapper(spec, len(cfg.classes), for_finetune=True).to(device)
        crit = build_loss_fn(loss_name, train_df["label_idx"].to_numpy(), len(cfg.classes), device)
        probe_loader = DataLoader(probe_ds, batch_size=micro_bs, shuffle=False)
        b = next(iter(probe_loader))
        w.train()
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            lg = w(b, device)
            ls = crit(lg, b["label"].to(device))  # type: ignore[union-attr]
        ls.backward()
        cleanup_cuda(w, crit)

    plan = select_safe_micro_batch(
        _clean_probe,
        effective_batch_size=spec.default_effective_bs,
        candidates=spec.candidates,
    )
    print(
        f"[vram-plan] {model_key} ({split_col}): micro_bs={plan.micro_batch_size} "
        f"x accum={plan.grad_accum_steps} | peak_alloc={plan.peak_allocated_gb:.2f} GB "
        f"peak_res={plan.peak_reserved_gb:.2f} GB"
    )

    wrapper = UnimodalBackboneWrapper(spec, len(cfg.classes), for_finetune=True).to(device)
    criterion = build_loss_fn(loss_name, train_df["label_idx"].to_numpy(), len(cfg.classes), device)

    enc_params = [p for p in wrapper.encoder.parameters() if p.requires_grad]
    head_params = list(wrapper.pool.parameters()) + list(wrapper.head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_backbone * head_lr_mult},
        ],
        weight_decay=weight_decay,
    )

    train_loader = DataLoader(
        RawClipDataset(train_df, cfg.media_dir, modality=spec.modality),
        batch_size=plan.micro_batch_size,
        shuffle=True,
        num_workers=0 if spec.modality == "text" else 2,
        prefetch_factor=2 if spec.modality != "text" else None,
        persistent_workers=(spec.modality != "text"),
        pin_memory=False,
    )
    val_loader = DataLoader(
        RawClipDataset(val_df, cfg.media_dir, modality=spec.modality),
        batch_size=max(plan.micro_batch_size, 4),
        shuffle=False,
        num_workers=0 if spec.modality == "text" else 2,
        persistent_workers=(spec.modality != "text"),
        pin_memory=False,
    )

    total_steps = max(1, epochs * math.ceil(len(train_loader) / plan.grad_accum_steps))
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup_steps
        if step < warmup_steps
        else 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps))),
    )

    best_val_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    t_ft_start = time.time()

    for epoch in range(1, epochs + 1):
        wrapper.train()
        optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(tqdm(train_loader, desc=f"{model_key} {split_col} ep{epoch}", leave=False)):
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits = wrapper(batch, device)
                loss = criterion(logits, batch["label"].to(device)) / plan.grad_accum_steps  # type: ignore[union-attr]
            loss.backward()
            if (step_idx + 1) % plan.grad_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        # Evaluasi validation
        wrapper.eval()
        val_logits, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                    logits = wrapper(batch, device)
                val_logits.append(logits.float().cpu().numpy())
                val_targets.append(batch["label"].numpy())  # type: ignore[union-attr]
        val_eval = evaluate_logits(np.concatenate(val_logits), np.concatenate(val_targets), cfg.classes)
        val_f1 = float(val_eval["macro_f1"])  # type: ignore[arg-type]
        elapsed_m = (time.time() - t_ft_start) / 60.0
        eta_m = (elapsed_m / epoch) * (epochs - epoch)
        print(
            f"[{model_key} | {split_col}] ep {epoch}/{epochs} ({epoch*100//epochs}%) "
            f"val_f1={val_f1:.4f} acc={val_eval['accuracy']:.4f} | elapsed={elapsed_m:.1f}m | ETA model=~{eta_m:.1f}m",
            flush=True,
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in wrapper.state_dict().items() if v.requires_grad or k.startswith(("pool.", "head."))}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        wrapper.load_state_dict(best_state, strict=False)

    # Evaluasi test set
    test_loader = DataLoader(
        RawClipDataset(test_df, cfg.media_dir, modality=spec.modality),
        batch_size=max(plan.micro_batch_size, 4),
        shuffle=False,
        num_workers=0 if spec.modality == "text" else 2,
        pin_memory=False,
    )
    wrapper.eval()
    test_logits, test_targets, test_uids = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits = wrapper(batch, device)
            test_logits.append(logits.float().cpu().numpy())
            test_targets.append(batch["label"].numpy())  # type: ignore[union-attr]
            test_uids.extend(list(batch["utterance_id"]))  # type: ignore[arg-type]
    test_eval = evaluate_logits(np.concatenate(test_logits), np.concatenate(test_targets), cfg.classes)

    # Ekstrak seluruh fitur split-specific menggunakan bobot terbaik
    extract_backbone_features(cfg, model_key, df, cache_out, wrapper=wrapper, batch_size=max(8, plan.micro_batch_size * 2))

    import json

    from losses_metrics import export_audit_predictions

    pred_csv = cfg.runs_dir / "predictions" / f"{model_key}__{split_col}__seed{seed}.csv"
    export_audit_predictions(test_uids, test_eval["probs"], cfg.classes, pred_csv)  # type: ignore[arg-type]

    result = {
        "model_key": model_key,
        "tier": spec.tier,
        "modality": spec.modality,
        "split": split_col,
        "seed": seed,
        "val_macro_f1": round(best_val_f1, 4),
        "test_macro_f1": round(float(test_eval["macro_f1"]), 4),  # type: ignore[arg-type]
        "test_accuracy": round(float(test_eval["accuracy"]), 4),  # type: ignore[arg-type]
        "per_class_f1": test_eval["per_class_f1"],
        "confusion_matrix": test_eval["confusion_matrix"],
        "vram_plan": {
            "micro_bs": plan.micro_batch_size,
            "grad_accum": plan.grad_accum_steps,
            "peak_allocated_gb": plan.peak_allocated_gb,
            "peak_reserved_gb": plan.peak_reserved_gb,
        },
        "cache_path": str(cache_out),
        "predictions_csv": str(pred_csv),
    }
    metric_out.parent.mkdir(parents=True, exist_ok=True)
    metric_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    cleanup_cuda(wrapper, optimizer, criterion)
    return result
