"""Arsitektur 2D CRNN audio, kepala temporal unimodal, dan fusi anti-dominasi teks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


class AttentiveStatsPool(nn.Module):
    """Pooling atensi menghasilkan ``[weighted_mean; weighted_std]`` (dimensi ``2 * D``)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = max(32, dim // 2)
        self.attn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 2:
            zeros = torch.zeros_like(x)
            return torch.cat([x, zeros], dim=-1)
        scores = self.attn(x).squeeze(-1)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, -1e4)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        mean = (x * weights).sum(dim=1)
        var = ((x - mean.unsqueeze(1)).pow(2) * weights).sum(dim=1).clamp(min=1e-6)
        return torch.cat([mean, var.sqrt()], dim=-1)


class CRNN2DAudio(nn.Module):
    """2D CRNN (Conv2d + BiGRU + Attentive Stats Pooling) di atas Log-Mel Spectrogram."""

    def __init__(self, num_classes: int, *, hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.melspec = T.MelSpectrogram(
            sample_rate=16000, n_fft=512, hop_length=160, n_mels=80, power=2.0
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),
        )
        self.gru = nn.GRU(
            input_size=128 * 10,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        seq_dim = hidden_dim * 2
        self.pool = AttentiveStatsPool(seq_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(seq_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(seq_dim * 2, seq_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_dim, num_classes),
        )

    def encode_sequence(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            mel = torch.log(self.melspec(wav.float()).clamp(min=1e-6)).unsqueeze(1)
        x = self.conv(mel)  # [B, 128, 10, T']
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        seq, _ = self.gru(x)
        return seq

    def forward(self, wav: torch.Tensor, return_feats: bool = False):
        seq = self.encode_sequence(wav)
        pooled = self.pool(seq)
        logits = self.classifier(pooled)
        if return_feats:
            return logits, seq
        return logits


class TemporalEncoderBlock(nn.Module):
    """Proyeksi fitur ter-cache + Transformer Temporal opsional."""

    def __init__(
        self,
        in_dim: int,
        proj_dim: int,
        *,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.encoder: nn.TransformerEncoder | None = None
        if num_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=proj_dim,
                nhead=num_heads,
                dim_feedforward=proj_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = AttentiveStatsPool(proj_dim)

    def encode_seq(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
            pad_mask = None
        z = self.proj(x)
        if self.encoder is not None and z.shape[1] > 1:
            z = self.encoder(z, src_key_padding_mask=pad_mask)
        return z

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.encode_seq(x, pad_mask)
        pooled = self.pool(seq, pad_mask if seq.shape[1] > 1 else None)
        return seq, pooled


class UnimodalHead(nn.Module):
    """Kepala klasifikasi unimodal di atas fitur sekuensial/pooled ter-cache."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        *,
        proj_dim: int = 256,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = TemporalEncoderBlock(
            in_dim, proj_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout
        )
        self.head = nn.Sequential(
            nn.LayerNorm(proj_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        _, pooled = self.backbone(x, pad_mask)
        return self.head(pooled)


def _apply_modality_dropout(
    vecs: dict[str, torch.Tensor],
    active: dict[str, torch.Tensor],
    p_mod: float,
    training: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Matikan fitur teks secara acak saat training (p_mod) bila modalitas lain aktif."""
    if not training or p_mod <= 0.0 or "text" not in vecs or len(vecs) < 2:
        return vecs, active
    first = next(iter(vecs.values()))
    keep_prob = torch.rand(first.shape[0], device=first.device)
    drop_text = keep_prob < p_mod
    new_vecs = dict(vecs)
    new_active = dict(active)
    if drop_text.any():
        mask = (~drop_text).float().view(-1, *([1] * (vecs["text"].ndim - 1)))
        new_vecs["text"] = vecs["text"] * mask
        new_active["text"] = active["text"] & (~drop_text)
    return new_vecs, new_active


class EarlyFusionNet(nn.Module):
    """Early Fusion: proyeksi dimensi seragam -> concat -> Residual MLP."""

    def __init__(
        self,
        in_dims: dict[str, int],
        num_classes: int,
        *,
        proj_dim: int = 256,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.3,
        modality_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.modalities = ("text", "audio", "visual")
        self.modality_dropout = modality_dropout
        self.encoders = nn.ModuleDict(
            {
                m: TemporalEncoderBlock(
                    in_dims[m], proj_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout
                )
                for m in self.modalities
                if m in in_dims
            }
        )
        total_dim = proj_dim * 2 * len(self.encoders)
        self.mlp = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Dropout(dropout),
            nn.Linear(total_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        pad_masks: dict[str, torch.Tensor | None] | None = None,
        return_details: bool = False,
    ):
        pad_masks = pad_masks or {}
        vecs: dict[str, torch.Tensor] = {}
        active: dict[str, torch.Tensor] = {}
        ref = next(iter(inputs.values()))
        bsz = ref.shape[0]
        for m, enc in self.encoders.items():
            if m in inputs and inputs[m] is not None:
                _, pooled = enc(inputs[m], pad_masks.get(m))
                vecs[m] = pooled
                active[m] = torch.ones(bsz, dtype=torch.bool, device=pooled.device)
            else:
                vecs[m] = torch.zeros((bsz, enc.pool.attn[0].in_features * 2), device=ref.device)
                active[m] = torch.zeros(bsz, dtype=torch.bool, device=ref.device)

        vecs, _ = _apply_modality_dropout(vecs, active, self.modality_dropout, self.training)
        fused = torch.cat([vecs[m] for m in self.encoders], dim=-1)
        logits = self.mlp(fused)
        if return_details:
            return logits, {}
        return logits


class LateFusionNet(nn.Module):
    """Late Fusion berbobot adaptif: tahan terhadap missing modality saat deployment."""

    def __init__(
        self,
        in_dims: dict[str, int],
        num_classes: int,
        *,
        proj_dim: int = 256,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.3,
        modality_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.modalities = tuple(m for m in ("text", "audio", "visual") if m in in_dims)
        self.modality_dropout = modality_dropout
        self.heads = nn.ModuleDict(
            {
                m: UnimodalHead(
                    in_dims[m],
                    num_classes,
                    proj_dim=proj_dim,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for m in self.modalities
            }
        )
        self.gate = nn.Sequential(
            nn.Linear(len(self.modalities) * num_classes, 64),
            nn.GELU(),
            nn.Linear(64, len(self.modalities)),
        )
        self.num_classes = num_classes

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        pad_masks: dict[str, torch.Tensor | None] | None = None,
        return_details: bool = False,
    ):
        pad_masks = pad_masks or {}
        ref = next(iter(inputs.values()))
        bsz = ref.shape[0]
        logits_map: dict[str, torch.Tensor] = {}
        active_map: dict[str, torch.Tensor] = {}
        for m in self.modalities:
            if m in inputs and inputs[m] is not None:
                logits_map[m] = self.heads[m](inputs[m], pad_masks.get(m))
                active_map[m] = torch.ones(bsz, dtype=torch.bool, device=ref.device)
            else:
                logits_map[m] = torch.zeros((bsz, self.num_classes), device=ref.device)
                active_map[m] = torch.zeros(bsz, dtype=torch.bool, device=ref.device)

        logits_map, active_map = _apply_modality_dropout(
            logits_map, active_map, self.modality_dropout, self.training
        )
        concat_logits = torch.cat([logits_map[m] for m in self.modalities], dim=-1)
        raw_weights = self.gate(concat_logits)
        active_mask = torch.stack([active_map[m] for m in self.modalities], dim=-1)
        raw_weights = raw_weights.masked_fill(~active_mask, -1e4)
        weights = torch.softmax(raw_weights, dim=-1)

        stacked = torch.stack([logits_map[m] for m in self.modalities], dim=1)  # [B, M, C]
        fused_logits = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        if return_details:
            return fused_logits, {
                "per_modality_logits": logits_map,
                "weights": {m: weights[:, i] for i, m in enumerate(self.modalities)},
            }
        return fused_logits


class GatedCrossModalNet(nn.Module):
    """Cross-Modal Attention (Text <- Audio/Visual) + Gated Multimodal Unit (GMU)."""

    def __init__(
        self,
        in_dims: dict[str, int],
        num_classes: int,
        *,
        proj_dim: int = 256,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.3,
        modality_dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.modalities = tuple(m for m in ("text", "audio", "visual") if m in in_dims)
        self.proj_dim = proj_dim
        self.modality_dropout = modality_dropout
        self.encoders = nn.ModuleDict(
            {
                m: TemporalEncoderBlock(
                    in_dims[m], proj_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout
                )
                for m in self.modalities
            }
        )
        self.cross_ta = nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_tv = nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ta = nn.LayerNorm(proj_dim)
        self.norm_tv = nn.LayerNorm(proj_dim)
        self.pool = AttentiveStatsPool(proj_dim)

        n_streams = len(self.modalities)
        pooled_dim = proj_dim * 2
        self.gmu_proj = nn.ModuleDict(
            {m: nn.Linear(pooled_dim, pooled_dim) for m in self.modalities}
        )
        self.gmu_gate = nn.Linear(pooled_dim * n_streams, n_streams)
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        pad_masks: dict[str, torch.Tensor | None] | None = None,
        return_details: bool = False,
    ):
        pad_masks = pad_masks or {}
        ref = next(iter(inputs.values()))
        bsz = ref.shape[0]
        seqs: dict[str, torch.Tensor] = {}
        active: dict[str, torch.Tensor] = {}
        for m in self.modalities:
            if m in inputs and inputs[m] is not None:
                seqs[m] = self.encoders[m].encode_seq(inputs[m], pad_masks.get(m))
                active[m] = torch.ones(bsz, dtype=torch.bool, device=ref.device)
            else:
                seqs[m] = torch.zeros((bsz, 1, self.proj_dim), device=ref.device)
                active[m] = torch.zeros(bsz, dtype=torch.bool, device=ref.device)

        seqs, active = _apply_modality_dropout(seqs, active, self.modality_dropout, self.training)
        pooled: dict[str, torch.Tensor] = {}
        if "text" in seqs:
            t_seq = seqs["text"]
            enriched = t_seq
            if "audio" in seqs and active["audio"].any():
                ca, _ = self.cross_ta(t_seq, seqs["audio"], seqs["audio"], need_weights=False)
                enriched = enriched + self.norm_ta(ca)
            if "visual" in seqs and active["visual"].any():
                cv, _ = self.cross_tv(t_seq, seqs["visual"], seqs["visual"], need_weights=False)
                enriched = enriched + self.norm_tv(cv)
            pooled["text"] = self.pool(enriched)
        for m in ("audio", "visual"):
            if m in seqs:
                pooled[m] = self.encoders[m].pool(seqs[m], pad_masks.get(m) if seqs[m].shape[1] > 1 else None)

        transformed = torch.stack(
            [torch.tanh(self.gmu_proj[m](pooled[m])) for m in self.modalities], dim=1
        )
        gate_logits = self.gmu_gate(torch.cat([pooled[m] for m in self.modalities], dim=-1))
        active_mask = torch.stack([active[m] for m in self.modalities], dim=-1)
        gate_logits = gate_logits.masked_fill(~active_mask, -1e4)
        gates = torch.softmax(gate_logits, dim=-1)
        fused = (transformed * gates.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(fused)
        if return_details:
            return logits, {"weights": {m: gates[:, i] for i, m in enumerate(self.modalities)}}
        return logits


class MulTFusionNet(nn.Module):
    """Bidirectional Cross-Modal Transformer (MulT) antar seluruh pasangan modalitas."""

    def __init__(
        self,
        in_dims: dict[str, int],
        num_classes: int,
        *,
        proj_dim: int = 256,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.3,
        modality_dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.modalities = tuple(m for m in ("text", "audio", "visual") if m in in_dims)
        self.proj_dim = proj_dim
        self.modality_dropout = modality_dropout
        self.encoders = nn.ModuleDict(
            {
                m: TemporalEncoderBlock(
                    in_dims[m], proj_dim, num_layers=1, num_heads=num_heads, dropout=dropout
                )
                for m in self.modalities
            }
        )
        self.pair_attns = nn.ModuleDict()
        self.pair_norms = nn.ModuleDict()
        for q_mod in self.modalities:
            for kv_mod in self.modalities:
                if q_mod == kv_mod:
                    continue
                key = f"{q_mod}_from_{kv_mod}"
                self.pair_attns[key] = nn.MultiheadAttention(
                    proj_dim, num_heads, dropout=dropout, batch_first=True
                )
                self.pair_norms[key] = nn.LayerNorm(proj_dim)

        self.pools = nn.ModuleDict({m: AttentiveStatsPool(proj_dim) for m in self.modalities})
        total_dim = proj_dim * 2 * len(self.modalities)
        self.classifier = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Dropout(dropout),
            nn.Linear(total_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        pad_masks: dict[str, torch.Tensor | None] | None = None,
        return_details: bool = False,
    ):
        pad_masks = pad_masks or {}
        ref = next(iter(inputs.values()))
        bsz = ref.shape[0]
        seqs: dict[str, torch.Tensor] = {}
        active: dict[str, torch.Tensor] = {}
        for m in self.modalities:
            if m in inputs and inputs[m] is not None:
                seqs[m] = self.encoders[m].encode_seq(inputs[m], pad_masks.get(m))
                active[m] = torch.ones(bsz, dtype=torch.bool, device=ref.device)
            else:
                seqs[m] = torch.zeros((bsz, 1, self.proj_dim), device=ref.device)
                active[m] = torch.zeros(bsz, dtype=torch.bool, device=ref.device)

        seqs, _ = _apply_modality_dropout(seqs, active, self.modality_dropout, self.training)
        pooled: list[torch.Tensor] = []
        for q_mod in self.modalities:
            stream = seqs[q_mod]
            for kv_mod in self.modalities:
                if q_mod == kv_mod:
                    continue
                key = f"{q_mod}_from_{kv_mod}"
                attn_out, _ = self.pair_attns[key](
                    seqs[q_mod], seqs[kv_mod], seqs[kv_mod], need_weights=False
                )
                stream = stream + self.pair_norms[key](attn_out)
            pooled.append(self.pools[q_mod](stream))

        logits = self.classifier(torch.cat(pooled, dim=-1))
        if return_details:
            return logits, {}
        return logits


FUSION_REGISTRY: dict[str, type[nn.Module]] = {
    "early": EarlyFusionNet,
    "late": LateFusionNet,
    "gated_cross_attn": GatedCrossModalNet,
    "mult": MulTFusionNet,
}


def build_fusion_model(
    arch: str,
    in_dims: dict[str, int],
    num_classes: int,
    **hparams,
) -> nn.Module:
    """Bangun model fusi dari ``FUSION_REGISTRY`` secara seragam."""
    if arch not in FUSION_REGISTRY:
        raise ValueError(f"Arsitektur fusi tidak dikenal: {arch}")
    return FUSION_REGISTRY[arch](in_dims=in_dims, num_classes=num_classes, **hparams)
