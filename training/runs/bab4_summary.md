# Ringkasan Hasil Eksperimen untuk BAB IV

## 1. Perbandingan Model Utama (`Split A`, `Split B`, `Split C` & Kebocoran `Δ`)

| Model | Split A (Macro-F1) | Split B (Macro-F1) | Split C (Macro-F1) | Δ A→B | Δ B→C |
|---|---|---|---|---|---|
| `audio_crnn` | 0.2820 ± 0.0243 | 0.1807 ± 0.0067 | 0.1836 ± 0.0429 | +0.1013 | -0.0029 |
| `finetune_audio_wav2vec2_msp` | 0.1647 ± 0.0000 | 0.1452 ± 0.0000 | 0.1971 ± 0.0000 | +0.0195 | -0.0519 |
| `finetune_audio_wavlm_large` | 0.2294 ± 0.0000 | 0.2056 ± 0.0000 | 0.2052 ± 0.0000 | +0.0238 | +0.0004 |
| `finetune_text_deberta_large` | 0.2602 ± 0.0000 | 0.2399 ± 0.0000 | 0.2233 ± 0.0000 | +0.0203 | +0.0166 |
| `finetune_text_modernbert_large` | 0.2717 ± 0.0000 | 0.2596 ± 0.0000 | 0.2672 ± 0.0000 | +0.0121 | -0.0076 |
| `finetune_text_roberta_large_emo` | 0.3466 ± 0.0000 | 0.2595 ± 0.0000 | 0.2555 ± 0.0000 | +0.0871 | +0.0040 |
| `finetune_visual_convnext_base` | 0.4059 ± 0.0000 | 0.2272 ± 0.0000 | 0.3044 ± 0.0000 | +0.1787 | -0.0772 |
| `finetune_visual_videomaev2_base` | 0.2821 ± 0.0000 | 0.1639 ± 0.0000 | 0.1666 ± 0.0000 | +0.1182 | -0.0027 |
| `hemat_early` | 0.4477 ± 0.0046 | 0.2836 ± 0.0054 | 0.3165 ± 0.0111 | +0.1641 | -0.0329 |
| `hemat_late` | 0.4427 ± 0.0266 | 0.2804 ± 0.0188 | 0.3226 ± 0.0174 | +0.1623 | -0.0422 |
| `hybrid_best_early` | 0.4449 ± 0.0026 | 0.2258 ± 0.0099 | 0.3035 ± 0.0026 | +0.2191 | -0.0777 |
| `hybrid_best_gated_cross_attn` | 0.4254 ± 0.0124 | 0.2122 ± 0.0052 | 0.2994 ± 0.0106 | +0.2132 | -0.0872 |
| `hybrid_best_late` | 0.4117 ± 0.0076 | 0.2088 ± 0.0082 | 0.2853 ± 0.0118 | +0.2029 | -0.0765 |
| `hybrid_best_mult` | 0.4242 ± 0.0061 | 0.2263 ± 0.0079 | 0.3167 ± 0.0137 | +0.1979 | -0.0904 |
| `hybrid_max_gated_cross_attn` | 0.4354 ± 0.0154 | 0.2065 ± 0.0068 | 0.3000 ± 0.0025 | +0.2289 | -0.0935 |
| `hybrid_max_late` | 0.4139 ± 0.0057 | 0.2223 ± 0.0099 | 0.2952 ± 0.0030 | +0.1916 | -0.0729 |
| `hybrid_stacking_ensemble` | 0.4700 ± 0.0116 | 0.2704 ± 0.0148 | 0.3262 ± 0.0139 | +0.1996 | -0.0558 |
| `maksimal_early` | 0.3551 ± 0.0051 | 0.2137 ± 0.0040 | 0.2423 ± 0.0059 | +0.1414 | -0.0286 |
| `maksimal_gated_cross_attn` | 0.3518 ± 0.0096 | 0.1985 ± 0.0195 | 0.2437 ± 0.0056 | +0.1533 | -0.0452 |
| `maksimal_late` | 0.3658 ± 0.0099 | 0.1624 ± 0.0099 | 0.2281 ± 0.0140 | +0.2034 | -0.0657 |
| `maksimal_mult` | 0.3782 ± 0.0115 | 0.2225 ± 0.0109 | 0.2416 ± 0.0281 | +0.1557 | -0.0191 |
| `stacking_ensemble` | 0.4276 ± 0.0018 | 0.2477 ± 0.0011 | 0.3058 ± 0.0155 | +0.1799 | -0.0581 |
| `unimodal_audio_emotion2vec_base` | 0.2430 ± 0.0080 | 0.2158 ± 0.0095 | 0.2260 ± 0.0014 | +0.0272 | -0.0102 |
| `unimodal_audio_emotion2vec_large` | 0.2452 ± 0.0175 | 0.2456 ± 0.0277 | 0.2545 ± 0.0073 | -0.0004 | -0.0089 |
| `unimodal_audio_hubert_base` | 0.2571 ± 0.0278 | 0.1753 ± 0.0077 | 0.2489 ± 0.0086 | +0.0818 | -0.0736 |
| `unimodal_audio_wav2vec2_base` | 0.2017 ± 0.0050 | 0.1313 ± 0.0020 | 0.2060 ± 0.0171 | +0.0704 | -0.0747 |
| `unimodal_audio_wavlm_base` | 0.2389 ± 0.0308 | 0.1735 ± 0.0207 | 0.2500 ± 0.0066 | +0.0654 | -0.0765 |
| `unimodal_text_deberta_base` | 0.2736 ± 0.0050 | 0.2581 ± 0.0102 | 0.2748 ± 0.0141 | +0.0155 | -0.0167 |
| `unimodal_text_distilroberta_emo` | 0.2756 ± 0.0053 | 0.2563 ± 0.0130 | 0.2694 ± 0.0137 | +0.0193 | -0.0131 |
| `unimodal_text_roberta_base` | 0.2956 ± 0.0050 | 0.2271 ± 0.0070 | 0.2403 ± 0.0096 | +0.0685 | -0.0132 |
| `unimodal_visual_convnext_tiny` | 0.4049 ± 0.0157 | 0.1592 ± 0.0296 | 0.2570 ± 0.0039 | +0.2457 | -0.0978 |
| `unimodal_visual_effnet_b0` | 0.3740 ± 0.0093 | 0.1640 ± 0.0076 | 0.1757 ± 0.0046 | +0.2100 | -0.0117 |
| `unimodal_visual_emotieff_b0` | 0.4229 ± 0.0220 | 0.2043 ± 0.0234 | 0.2751 ± 0.0103 | +0.2186 | -0.0708 |
| `unimodal_visual_emotieff_b2` | 0.2597 ± 0.0097 | 0.2654 ± 0.0059 | 0.2634 ± 0.0148 | -0.0057 | +0.0020 |
| `unimodal_visual_resnet50` | 0.3850 ± 0.0056 | 0.1749 ± 0.0063 | 0.1989 ± 0.0125 | +0.2101 | -0.0240 |
| `unimodal_visual_vit_base` | 0.4269 ± 0.0092 | 0.1700 ± 0.0045 | 0.2392 ± 0.0291 | +0.2569 | -0.0692 |

## 2. Ablasi 7 Kombinasi Modalitas

| Kombinasi | Split A | Split B | Split C |
|---|---|---|---|
| `hemat_ablation_late__audio` | 0.2697 ± 0.0123 | 0.1924 ± 0.0079 | 0.2352 ± 0.0058 |
| `hemat_ablation_late__audio+visual` | 0.4138 ± 0.0224 | 0.2633 ± 0.0103 | 0.2838 ± 0.0113 |
| `hemat_ablation_late__text` | 0.2832 ± 0.0096 | 0.2472 ± 0.0109 | 0.2726 ± 0.0105 |
| `hemat_ablation_late__text+audio` | 0.2889 ± 0.0166 | 0.2679 ± 0.0064 | 0.2817 ± 0.0046 |
| `hemat_ablation_late__text+audio+visual` | 0.4252 ± 0.0148 | 0.2817 ± 0.0181 | 0.2998 ± 0.0073 |
| `hemat_ablation_late__text+visual` | 0.4296 ± 0.0197 | 0.2536 ± 0.0243 | 0.3166 ± 0.0126 |
| `hemat_ablation_late__visual` | 0.4294 ± 0.0243 | 0.2411 ± 0.0173 | 0.2619 ± 0.0117 |
| `maksimal_ablation_gated_cross_attn__audio` | 0.2530 ± 0.0135 | 0.2245 ± 0.0094 | 0.2223 ± 0.0096 |
| `maksimal_ablation_gated_cross_attn__audio+visual` | 0.3392 ± 0.0082 | 0.1473 ± 0.0062 | 0.2162 ± 0.0121 |
| `maksimal_ablation_gated_cross_attn__text` | 0.2539 ± 0.0063 | 0.2411 ± 0.0084 | 0.2637 ± 0.0008 |
| `maksimal_ablation_gated_cross_attn__text+audio` | 0.2841 ± 0.0362 | 0.2068 ± 0.0159 | 0.2951 ± 0.0129 |
| `maksimal_ablation_gated_cross_attn__text+audio+visual` | 0.3602 ± 0.0143 | 0.1991 ± 0.0194 | 0.2442 ± 0.0051 |
| `maksimal_ablation_gated_cross_attn__text+visual` | 0.3404 ± 0.0150 | 0.2092 ± 0.0206 | 0.1954 ± 0.0218 |
| `maksimal_ablation_gated_cross_attn__visual` | 0.3008 ± 0.0164 | 0.1470 ± 0.0038 | 0.1586 ± 0.0085 |
