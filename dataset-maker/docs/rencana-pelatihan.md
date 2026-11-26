# Rancangan Pelatihan Model Klasifikasi Emosi Multimodal (RTX 5060 8 GB VRAM)

Dokumen ini disusun sebagai acuan eksperimen pelatihan model (**Tahap 12–13**) dan landasan penulisan **BAB III (Metodologi Penelitian)** serta **BAB IV (Hasil dan Pembahasan)**, menyesuaikan kondisi perangkat keras pusat training (**NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM, PyTorch `2.14.1+cu130` dengan dukungan native `bfloat16`**), spesifikasi `dataset-maker/config.yaml`, proposal revisi, dan rubrik penilaian mata kuliah (*Project Machine Learning.pptx*).

---

## 1. Kepatuhan terhadap Rubrik Proyek & Pipeline Dataset

| Aspek | Ketentuan Rubrik (`PPT`) & `config.yaml` | Implementasi di Rancangan Pelatihan |
|---|---|---|
| **Modalitas Data** | Wajib multimodal (`text`, `audio`, `video`), bukan tabular | Memanfaatkan output `10_cut.py`: teks dialog (`clips.csv`), `audio/<clip_id>.wav` (16 kHz mono), wajah `faces/<clip_id>.mp4` (224×224 @ 5 fps), dan video klip `clips/<clip_id>.mp4` |
| **Jumlah Model Pembanding** | Minimal **3** model (*Basic*); **>3 model** untuk nilai **Bonus** | Minimal **9 model utama** (6 unimodal: 2 per modalitas + 3 arsitektur fusi), dapat diperluas ke **11 model** bila memasukkan pembanding *emotion-pretrained* |
| **Evaluasi & Split** | Menguji klaim fusi & kebocoran data (`11_split.py`) | 3 varian split (`A_random_utterance`, `B_per_film`, `C_per_actor`), **3 random seed** per skenario (`mean ± std`), dan **7 kombinasi ablasi modalitas** |
| **Metrik Utama** | Menangani ketimpangan kelas | **`Macro-F1`** (`config.yaml`: `primary_metric: macro_f1`), didukung Accuracy, F1 per kelas, dan Confusion Matrix |
| **Hyperparameter Tuning** | Wajib dibahas di Bab III & tabel tuning | **Optuna (`TPESampler` + `MedianPruner`, `15 trials`)** pada validation set tiap split (tanpa menyentuh test set) |
| **Deployment & Artefak** | Aplikasi lokal (bonus cloud) + **1 Jupyter Notebook runnable** | Aplikasi Streamlit menampilkan probabilitas per modalitas + peringatan modalitas hilang; notebook pelatihan + skrip terstruktur + `12_audit.py` |

> [!IMPORTANT]
> **Sinkronisasi Jumlah Kelas (Proposal vs `config.yaml`):**
> Proposal lama (9 Nov) mencantumkan 5 kelas, sedangkan `config.yaml` dan fase anotasi tim telah memakai **7 kelas**: `senang`, `sedih`, `marah`, `takut`, `netral`, `terkejut`, dan `jijik`. Hasil akhir `09_select.py` dari koordinator menentukan apakah model dilatih pada **7 kelas** (dengan *class weights / focal loss* bila `jijik` minoritas) atau **5/6 kelas** jika kelas minoritas digabung/dibuang sebelum training.

---

## 2. Arsitektur Eksekusi untuk VRAM 8 GB (RTX 5060)

Meload encoder teks (~184M parameter), audio (~94M), dan video (~86M) secara *end-to-end* bersamaan pada `batch_size=16` akan membutuhkan **>12–16 GB VRAM** dan langsung *Out-of-Memory* (OOM). Untuk memaksimalkan akurasi di **8 GB VRAM**, pelatihan dibagi menjadi **dua jalur terdisiplin**:

### Jalur 1 — *Two-Stage Offline Feature Caching* + Pelatihan Kepala & Fusi (Cepat & Hemat VRAM)
1. **Ekstraksi Fitur Per Modalitas (1 model di GPU dalam satu waktu):**
   - Jalankan masing-masing *pretrained backbone* dengan `torch.amp.autocast("cuda", dtype=torch.bfloat16)` dan `torch.no_grad()` (`batch_size=16–32`, pemakaian VRAM hanya **1,5–3,5 GB**).
   - Simpan representasi sekuensial (`[T, D]`) maupun *pooled* (`[D]`) ke disk (`.pt` atau `.npy`) per `clip_id`.
2. **Pelatihan Fusi, Ablasi 7 Kombinasi, & Optuna Tuning:**
   - Karena fitur sudah berbentuk tensor di disk, melatih model fusi (Early, Late, Cross-Attention) dan menjalankan **15 trial Optuna × 3 split × 3 seed** hanya memakan **< 1,5 GB VRAM** dan selesai dalam hitungan menit per konfigurasi.

### Jalur 2 — *Selective Unimodal Fine-Tuning* (Mengangkat Performa Unimodal Sebelum Fusi)
Menggunakan fitur *backbone* mentah (*frozen* ImageNet/AudioSet) sering menghasilkan performa audio/wajah yang lemah. Untuk mengejar performa maksimal tanpa melebihi 8 GB VRAM:
- **Fine-tune satu modalitas pada satu waktu** memakai teknik efisiensi VRAM berikut:
  - **`bfloat16` Mixed Precision:** RTX 5060 mendukung `bf16` secara *native* (`torch.cuda.is_bf16_supported() == True`), memotong memori aktivasi hingga ~50% tanpa risiko *underflow* seperti `fp16`.
  - **Gradient Checkpointing** (`model.gradient_checkpointing_enable()`): Wajib aktif saat *fine-tuning* `DeBERTa-v3-base`, `WavLM`, atau `VideoMAE`.
  - **LoRA / Partial Unfreezing:** Gunakan `PEFT LoRA` (`r=8` atau `16`, `lora_alpha=16–32`) pada `DeBERTa-v3-base`/`WavLM`, atau cukup *unfreeze* **2–4 blok Transformer terakhir** + *classification head*. Ini memangkas memori *optimizer state* AdamW hingga >80%.
  - **Gradient Accumulation:** Gunakan *micro-batch* fisik kecil (`batch_size=4` atau `8`) dengan `grad_accum_steps=2–4` agar *effective batch size* tetap `16–32`.

> [!CAUTION]
> **Aturan Anti-Kebocoran untuk Split A / B / C:**
> Fitur dari bobot **pretrained beku** (belum pernah melihat dataset kita) aman diekstrak sekali untuk semua 3.000 klip. Namun, jika melakukan **fine-tuning** encoder pada klip dataset kita, **encoder tersebut WAJIB di-fine-tune terpisah hanya menggunakan set `train` milik masing-masing Split (`A`, `B`, `C`)** sebelum mengekstrak fitur untuk `val` dan `test`. Mengekstrak fitur dari model yang di-fine-tune pada data campuran akan merusak eksperimen kebocoran Split B dan Split C.

### Penjagaan Batas VRAM & Pencegahan Spillover ke RAM Sistem
Meskipun *display* laptop ditangani iGPU (**Radeon 880M**), Windows WDDM dan PyTorch CUDA *caching allocator* tetap membutuhkan buffer **~300–500 MB** di atas `memory_allocated()` untuk workspace kernel `cuBLAS/cuDNN` dan fragmentasi blok (`memory_reserved()`). Memaksa *peak allocated* mepet hingga **7,6–7,7 GB** berisiko melewati kapasitas fisik ~7,8–8,0 GB dan memicu **CUDA Sysmem Fallback** (tensor bocor ke RAM sistem lewat PCIe → kecepatan anjlok berlipat-lipat). Sebaliknya, mengecilkan `batch_size` terlalu rendah (`bs=1–2`) bila sebenarnya `bs=4–8` muat juga membuat utilisasi GPU kurang optimal.

Solusi paling seimbang (*cepat tapi anti-bocor*):
1. **Pagar Keras Proses (~7,2–7,4 GB):** Aktifkan `torch.cuda.set_per_process_memory_fraction(0.92)` di awal skrip (dan atur *NVIDIA Control Panel → CUDA - Sysmem Fallback Policy → **Prefer No Sysmem Fallback***) agar bila berlebih langsung *fail-fast* saat *smoke test*, bukan melambat diam-diam ke RAM sistem.
2. **Pilih `Micro-Batch` Tercepat yang Muat di Pagar:** Saat *smoke test* pada batch skenario terberat (**klip 10 detik + 128 token teks**), pilih `batch_size` terbesar (`16 → 8 → 4 → 2`) yang membuat **`torch.cuda.max_memory_reserved()` ≤ 7,1–7,3 GB**, lalu pertahankan *effective batch* lewat `grad_accum_steps`.
3. **Bersih Total Antar-Model & Antar-Split:** Catat metrik wajib memakai `.item()` / `.detach().cpu()`, batasi DataLoader (`num_workers=2, prefetch_factor=2`) agar RAM fisik tidak penuh buffer decode `.mp4`, dan jalankan `model.cpu(); del model, optimizer, scheduler; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()` setiap berpindah split atau model.

---

## 3. Desain Arsitektur Fusi & Mekanisme Anti-Dominasi Teks

Sesuai pertanyaan riset di **Subbab 1.1 & 2.3**, model fusi sering gagal melampaui model teks karena modalitas teks mendominasi gradien saat training. Rancangan arsitektur fusi berikut memisahkan baseline ringan hingga fusi atensi penuh agar perbandingannya adil dan diagnostik:

1. **Early Fusion (Projected Concatenation + LayerNorm Residual MLP):** Tiap modalitas diproyeksikan via Linear + LayerNorm + GELU ke dimensi seragam (`d_proj = 256`), lalu digabung `[z_text; z_audio; z_visual]` → 2-layer MLP (`Dropout`) → 7 logits.
2. **Late Fusion (Modality-Weighted / Gated Probability Fusion):** Tiga *classifier head* unimodal independen menghasilkan logits $l_t, l_a, l_v$, digabungkan lewat **Gated Weighting Network** (memprediksi bobot $w_t, w_a, w_v$ per sampel via `Softmax`). Paling tangguh menghadapi *missing modality* pada aplikasi **Streamlit Deployment**.
3. **Gated Cross-Modal Attention Fusion (Kandidat Utama):** Memproses sekuens fitur temporal (`T_seq`, `A_seq`, `V_seq`) dengan *Multi-Head Cross-Attention* (`T ← A` dan `T ← V`), diikuti **Gated Multimodal Unit (GMU)** dan **Modality Dropout (`p_mod = 0.15`)** (secara acak menolkan vektor teks pada 15% sampel batch saat training agar fusi terpaksa belajar membaca audio dan wajah).
4. **MulT (*Bidirectional Cross-Modal Transformer*) + *Stacking Ensemble* (Opsi Fusi Maksimal):** Interaksi dua arah penuh (`T↔A`, `T↔V`, `A↔V`) + opsi *Supervised Contrastive Loss* atau *Out-of-Fold Stacking Ensemble* yang menggabungkan prediksi terbaik dari jalur *Hemat* dan *Maksimal*.

---

## 4. Perbandingan Model Hemat vs Model Maksimal (Optimal untuk VRAM 8 GB)

Seluruh kandidat berikut telah dipilih agar **aman berjalan pada RTX 5060 (8 GB VRAM)** menggunakan **`torch.bfloat16`**. Bagian kiri (*Model Hemat*) sangat cepat untuk baseline & iterasi 3 split × 3 seed (`~1–4 GB VRAM`), sedangkan bagian kanan (*Model Maksimal*) memaksimalkan sisa *headroom* VRAM hingga **`~4,5–7,0 GB`** melalui ***fine-tuning* (`bf16` + *gradient checkpointing* + `LoRA` / *partial unfreeze*)**.

### A. Modalitas Teks (Dialog `.srt` / Whisper, `max_length=128`)

| Model Hemat (~1,5–4,5 GB VRAM) | Model Maksimal (~5,0–6,8 GB VRAM) | Perbandingan & Strategi Fine-Tune 8 GB |
|---|---|---|
| `FacebookAI/roberta-base` (~125M) — baseline `models.lock.yaml` | **`microsoft/deberta-v3-large`** (~435M) | *Disentangled attention* kapasitas penuh; paling tajam menangkap sarkasme & makna implisit dialog. (`bf16` + `grad_ckpt` + `LoRA r=16` atau *unfreeze* 6 blok akhir, `bs=8`, `~5,2–6,5 GB`) |
| `microsoft/deberta-v3-base` (~184M) — utama di `models.lock.yaml` | **`answerdotai/ModernBERT-large`** (~395M) | Arsitektur encoder modern (*RoPE* + efisiensi atensi) sebagai pembanding kuat generasi baru terhadap DeBERTa. (`bf16` + `grad_ckpt`, `bs=8`, `~5,0–6,2 GB`) |
| `j-hartmann/emotion-english-distilroberta-base` (~82M) — *emotion-pretrained* ringan + audit `12_audit.py` | **`cardiffnlp/twitter-roberta-large-emotion-latest`** (~355M) | Versi *Large* yang sudah beradaptasi langsung pada klasifikasi emosi bahasa Inggris. (`bf16` + `LoRA` / `grad_ckpt`, `bs=8`, `~4,8–5,8 GB`) |

### B. Modalitas Audio / Speech Emotion Recognition (`media/audio/<clip_id>.wav`, 16 kHz Mono, Maks 10 dtk)

Audio film penuh musik latar & bisikan; gunakan **Attentive Statistics Pooling** di atas fitur frame agar bagian hening tidak mengencerkan intonasi puncak.

| Model Hemat (~1,0–4,5 GB VRAM) | Model Maksimal (~4,5–6,8 GB VRAM) | Perbandingan & Strategi Fine-Tune 8 GB |
|---|---|---|
| **2D CRNN (CNN + BiGRU)** di atas Log-Mel Spectrogram (~2–5M) | **`emotion2vec/emotion2vec_plus_large`** (~300M, via `funasr`) | Dilatih khusus pada **~42.526 jam** data emosi suara (~9× lebih besar dari `emotion2vec_plus_base` yang 4.788 jam); ekstraksi fitur 50 Hz `bf16` + adapter/probe atau *top-layer fine-tune* (`~4,5–6,0 GB`) |
| `facebook/wav2vec2-base-960h` & `facebook/hubert-base-ls960` (~95M, `models.lock.yaml`) | **`microsoft/wavlm-large`** (~316M) | Dilatih dengan *masked speech denoising* kapasitas penuh — jauh lebih tahan musik latar & efek suara dibanding `wav2vec2-base` ASR. (`bf16` + `grad_ckpt` + `LoRA` / *top-4 layers*, `bs=4`, `grad_accum=4`, `~5,5–6,8 GB`) |
| `microsoft/wavlm-base-plus` (~94M) & `emotion2vec/emotion2vec_plus_base` (~90M) | **`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`** (~12 layer *pruned*) | Varian `wav2vec2-large-robust` yang dipangkas menjadi 12 layer & dilatih pada **MSP-Podcast** untuk dimensi *Arousal–Dominance–Valence*; sangat hemat & kaya sinyal emosi untuk fusi (`~4,0–5,2 GB`) |

### C. Modalitas Visual / Wajah (`media/faces/<clip_id>.mp4`, 224×224 @ 5 fps)

Klip berisi 10–50 frame wajah ter-crop. Ekstraktor per frame disampling merata **16 frame** lalu diagregasi **1–2 layer Temporal Transformer / BiGRU**.

| Model Hemat (~1,2–4,5 GB VRAM) | Model Maksimal (~5,0–6,8 GB VRAM) | Perbandingan & Strategi Fine-Tune 8 GB |
|---|---|---|
| `ResNet-50` / `EfficientNet-B0` (`models.lock.yaml`) + BiGRU (~5–25M) | **`OpenGVLab/VideoMAEv2-Base`** / `MCG-NJU/videomae-base` (~86M) | Transformer video spatiotemporal 16×224×224 (mencapai **~73,6–76,6% WAR** di DFEW); menangkap dinamika transisi ekspresi (*onset → apex → offset*). (`bf16` + `grad_ckpt`, `bs=4`, `grad_accum=4`, `~5,5–6,8 GB`) |
| **`EmotiEffLib` (`enet_b0_8_best_afew`, ~5M)** + *Temporal Self-Attn* *(sangat irit)* | **`POSTER++` (`POSTER_V2`, ~43,7M)** + *Temporal Transformer* | Memadukan fitur visual + *facial landmark cross-attention* (~92,2% RAF-DB / ~67,5% AffectNet-7); sangat kuat terhadap variasi pose & oklusi wajah film (`~4,5–5,8 GB`) |
| `ViT-B/16` (`timm`, ~86M), `ConvNeXt-Tiny` (~28M), `trpakov/vit-face-expression` / `dima806` | **`EmotiEffLib` `enet_b2_8`** atau **`ConvNeXt-Base`** (`timm`, ~88M) + *Temporal Attn* | Kapasitas spasial lebih dalam dengan *head* temporal terlatih di atas wajah film (`bf16` + `grad_ckpt`, `bs=8`, `~4,5–6,0 GB`) |

### D. Fusi Multimodal & Ensemble (`< 2 GB` di atas Fitur Ter-cache)

| Model Hemat (Cepat & Stabil) | Model Maksimal (Akurasi Puncak) | Perbandingan & Kegunaan |
|---|---|---|
| **Early Fusion** (Concat MLP) & **Late Fusion** (Gated Logits) | **Gated Cross-Modal Attention (`Modality Dropout`)**, **MulT**, & **Out-of-Fold Stacking Ensemble** | Jalur *Hemat* ideal untuk baseline ablasi & ketahanan *missing modality* saat deployment; jalur *Maksimal* menangkap interaksi lintas waktu antar-modalitas & menggabungkan model *Hemat* + *Maksimal* |

### Batas Aman VRAM 8 GB & Risiko pada ~3.000 Klip
1. **Model yang sengaja tidak dipaksa:** **`VideoMAEv2-Large`** (butuh ~12–24 GB VRAM untuk fine-tuning) serta **LLM 3B–7B** seperti `Qwen2.5-Omni-7B` (QLoRA 4-bit lambat di 3 split × 3 seed dan pada riset MELD cenderung kalah stabil pada **Macro-F1** dibanding encoder `DeBERTa-v3-large`).
2. **Adu *Hemat* vs *Maksimal* di Split A, B, dan C:** Dengan ~2.100 klip latih per split, model *Maksimal* (300M–435M parameter) lebih rentan *overfit* dan menghafal aktor di Split A lalu turun di Split C. Karena itu wajib memakai **`LoRA` / *top-layer partial unfreeze*** + *differential learning rate* (`backbone 1e-5` vs `head 1e-4`), sekaligus menjadikan perbandingan *Model Hemat vs Model Maksimal* sebagai temuan utama di Bab IV (*apakah kapasitas lebih besar menaikkan Macro-F1 Split C atau justru memperbesar inflasi Δ A → C?*).

---

## 5. Rancangan Hyperparameter Tuning (Optuna)

Sesuai `config.yaml`, pencarian menggunakan **Optuna** (`optuna_trials: 15`, metrik target **`val_macro_f1`**, arah `maximize`) dengan **MedianPruner** (`n_startup_trials=5`, `n_warmup_steps=5`) untuk memangkas trial yang jelas buruk.

### A. Parameter Tetap (Baseline Terstandar)
- **Optimizer:** `AdamW` ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$)
- **Precision:** `torch.bfloat16` (`autocast('cuda', dtype=torch.bfloat16)`)
- **Scheduler:** `Linear Warmup` (`warmup_ratio = 0.1`) + `CosineAnnealingLR`
- **Gradient Clipping:** `max_norm = 1.0`
- **Early Stopping:** `patience = 5` (untuk MLP/Fusi maks `30 epochs`) atau `patience = 3` (untuk *fine-tuning* transformer `8–10 epochs`), memantau **Validation Macro-F1**.
- **Evaluasi Akhir:** Konfigurasi terbaik per model dijalankan ulang dengan **3 seeds (`[42, 123, 2026]`)** → dilaporkan **`mean ± std`**.

### B. Ruang Pencarian Hyperparameter (Optuna Search Space)

#### 1. *Fine-Tuning* Model Unimodal (Teks / Audio / Visual)
| Hyperparameter | Ruang Pencarian | Alasan |
|---|---|---|
| `learning_rate` (backbone) | Log-uniform `1e-5` – `5e-5` | Mencegah *catastrophic forgetting* pada bobot pralatih |
| `head_lr_mult` | Categorical `[5, 10]` | Kepala klasifikasi belajar 5–10× lebih cepat daripada backbone |
| `weight_decay` | Categorical `[0.01, 0.05]` | Regularisasi terhadap ~2.100 klip latih |
| `dropout` | Uniform `0.1` – `0.4` | Mengurangi *overfitting* pada Split B & Split C |
| `effective_batch_size` | Categorical `[16, 32]` | Fisik `4–8` × `grad_accum 2–4` agar aman di 8 GB VRAM |
| `loss_fn` | Categorical `[weighted_ce, focal_gamma_2]` | Mengangkat recall kelas minoritas (`jijik`, `terkejut`, `takut`) demi Macro-F1 |

#### 2. Arsitektur Fusi & Kepala Temporal (di atas Fitur Ter-cache)
| Hyperparameter | Ruang Pencarian | Alasan |
|---|---|---|
| `learning_rate` | Log-uniform `5e-5` – `1e-3` | Jaringan fusi diinisialisasi acak sehingga butuh LR lebih tinggi daripada backbone |
| `proj_dim` (`d_model`) | Categorical `[128, 256, 512]` | Mengontrol kapasitas proyeksi agar tidak menghafal aktor di Split A |
| `num_heads` *(Cross-Attn)* | Categorical `[4, 8]` | Jumlah *attention head* selaras dengan `proj_dim` |
| `num_layers` *(Temporal/Attn)* | Categorical `[1, 2]` | Mencegah *over-parameterization* pada ukuran korpus 3.000 klip |
| `dropout` | Uniform `0.2` – `0.5` | Regularisasi utama di lapisan fusi |
| `modality_dropout` | Uniform `0.0` – `0.25` | Mencegah teks memonopoli gradien saat fusi |
| `weight_decay` | Log-uniform `1e-4` – `1e-2` | Stabilitas bobot MLP & proyeksi |
| `batch_size` | Categorical `[16, 32]` | Sesuai `config.yaml` (`16`) & opsi batch `32` pada fitur ter-cache |
| `loss_fn` | Categorical `[weighted_ce, label_smooth_0.1, focal_gamma_2]` | Menyeimbangkan kalibrasi probabilitas dan Macro-F1 |

---

## 6. Strategi Eksekusi 1 Device (2 Gelombang Training & Estimasi Waktu)

Seluruh pelatihan dijalankan terpusat di **1 mesin (`ASUS Zephyrus — RTX 5060 Laptop GPU 8 GB VRAM`)** agar tidak perlu memecah media `3–6 GB` atau menyamakan *environment* di banyak komputer. Secara operasional, training dibagi menjadi **2 Gelombang Eksekusi** (bukan sekadar 2 kali *run*, melainkan **2 fase terstruktur** yang masing-masing mengeksekusi unimodal + fusi + split + seed):

### Gelombang 1 — Jalur *Model Hemat* (*Baseline* Lengkap & Siap Presentasi)
* **Tujuan:** Mengunci seluruh angka wajib proyek (unimodal dasar, fusi Early/Late, 7 ablasi modalitas, 3 split × 3 seed, serta model demo Streamlit) secepat mungkin agar tim langsung punya bahan Bab IV.
* **Cakupan:** Ekstraksi fitur *frozen backbone* (`bf16`) + pelatihan `2D CRNN`, `ResNet/EmotiEffLib`, `Early Fusion`, dan `Late Fusion` + Optuna `15 trials` di atas fitur ter-cache.

### Gelombang 2 — Jalur *Model Maksimal* (*Fine-Tuning* & Fusi Puncak)
* **Tujuan:** Mendongkrak akurasi & Macro-F1 dengan memanfaatkan sisa *headroom* VRAM hingga ~5–7 GB (`bf16` + *gradient checkpointing* + `LoRA`/*partial unfreeze*), lalu membandingkannya langsung melawan Gelombang 1.
* **Aturan Hemat Waktu Anti-Bengkak:** Untuk *fine-tuning* encoder berat (`DeBERTa-v3-large`, `ModernBERT-large`, `WavLM-Large`, `VideoMAEv2-Base`, `POSTER++`) per **Split A / B / C**, cukup jalankan **1 seed (`42`) per split** saat mengekstrak fitur ter-fine-tune, kemudian jalankan **3 seed penuh (`42, 123, 2026`) + Optuna `15 trials`** di tahap Fusi (**Gated Cross-Attention**, **MulT**, **Out-of-Fold Stacking Ensemble**).
* **Dua Pilihan Skala Gelombang 2:**
  * **Paket Prioritas *(Disarankan)*:** Pilih 1–2 model Maksimal paling menjanjikan per modalitas (misal `DeBERTa-v3-large` + `ModernBERT-large`, `emotion2vec_plus_large` + `WavLM-Large`, `POSTER++` + `VideoMAEv2-Base`) + seluruh fusi Maksimal.
  * **Paket *Full Sweep*:** Menjalankan **seluruh** kandidat di tabel Bagian 4 tanpa terkecuali.

### Estimasi Total Waktu Komputasi (`~3.000 Klip`, RTX 5060 Laptop GPU 8 GB)

> [!NOTE]
> Angka berikut adalah **estimasi rentang waktu** untuk `~2.100 klip latih` per split menggunakan `torch.bfloat16` (belum termasuk waktu *download* awal bobot model dari HuggingFace/timm sekitar `15–45 menit` tergantung internet).

| Tahap Eksekusi | Aktivitas Utama | Estimasi Waktu |
|---|---|---|
| **Gelombang 1 — Fitur *Hemat*** | Ekstraksi fitur *frozen* Teks + Audio + Visual untuk 3.000 klip *(bottleneck di decode `.mp4`)* | **~0,7 – 1,5 jam** |
| **Gelombang 1 — Training *Hemat*** | `2D CRNN`, *head* unimodal, `Early/Late Fusion`, **7 Ablasi**, Optuna (`15 trials`), **3 Split × 3 Seed** | **~1,5 – 3,0 jam** |
| **Subtotal Gelombang 1** | **Semua target *baseline* selesai (aman 1 sesi sore/malam)** | **~2,5 – 4,5 jam** |
| **Gelombang 2 — Teks & Audio *Maksimal*** | *Fine-tuning* per Split A/B/C (`DeBERTa-v3-large`, `ModernBERT-large`, `emotion2vec_plus_large`, `WavLM-Large` / `wav2vec2-robust-12`) | **~2,0 – 4,0 jam** *(Prioritas)* <br> **~3,5 – 6,5 jam** *(Full Sweep)* |
| **Gelombang 2 — Visual *Maksimal*** | *Fine-tuning* spatiotemporal per Split A/B/C (`POSTER++`, `VideoMAEv2-Base`, opsional `ConvNeXt-Base`) + decode frame video | **~2,0 – 3,5 jam** *(Prioritas)* <br> **~4,5 – 7,0 jam** *(Full Sweep)* |
| **Gelombang 2 — Fusi *Maksimal* & Ensemble** | `Gated Cross-Attention`, `MulT`, Optuna (`15 trials` × 3 Split × 3 Seed), & `Out-of-Fold Stacking Ensemble` | **~1,0 – 1,5 jam** *(di atas cache)* |
| **Subtotal Gelombang 2** | **Penyelarasan model *Maksimal* (cocok 1–2 sesi semalaman)** | **~5,0 – 9,0 jam** *(Prioritas)* <br> **~9,0 – 15,0 jam** *(Full Sweep)* |
| **Evaluasi Akhir & Audit** | Jalankan `12_audit.py`, Confusion Matrix, analisis error Split C, & cek demo Streamlit | **~0,5 – 1,0 jam** |
| **TOTAL KESELURUHAN (Start → Finish)** | **Gelombang 1 + Gelombang 2 + Audit** | **~8 – 14,5 jam** *(Paket Prioritas)* <br> **~12 – 20,5 jam** *(Full Sweep)* |

**Tips Operasional di Laptop Zephyrus:**
1. Jalankan *pre-download* seluruh model *checkpoint*, cek *smoke test* batas `max_memory_reserved() ≤ 7,1–7,3 GB` pada batch klip 10 detik (lihat Bagian 2), dan jalankan *caching* fitur **Gelombang 1** di siang/sore hari sambil dicek apakah ada error *path*/OOM.
2. Jalankan antrean *fine-tuning* berat (**Gelombang 2**) saat malam/ditinggal tidur dengan mode *Sleep = Never*, kipas maksimal (*Turbo/Performance*), posisi belakang laptop sedikit terangkat untuk *airflow*, serta *checkpoint* otomatis per split dan error isolation (`try/except`) antar-model.

---

## 7. Matriks Eksperimen Lengkap untuk Bab IV

1. **Tabel Komparasi Utama (`Model Hemat` vs `Model Maksimal` × `3` Varian Split):**
   - Baris: seluruh pasangan *Model Hemat* vs *Model Maksimal* pada Teks, Audio, Visual, dan Fusi (`Early`, `Late`, `Gated Cross-Attention`, serta `MulT / Stacking Ensemble`).
   - Kolom: Macro-F1 & Accuracy (`mean ± std` 3 seed) pada **Split A** (acak), **Split B** (per film), **Split C** (per aktor), plus **Δ A → B** (inflasi kebocoran adegan) dan **Δ B → C** (inflasi kebocoran identitas).
2. **Tabel Ablasi 7 Kombinasi Modalitas (pada Arsitektur Fusi Terbaik):**
   - `T` saja, `A` saja, `V` saja, `T+A`, `T+V`, `A+V`, `T+A+V` di Split A, B, dan C — membuktikan secara empiris apakah penambahan audio & visual memberi peningkatan signifikan saat kebocoran ditutup.
3. **Audit & Error Analysis:**
   - Jalankan `12_audit.py --predictions ...` untuk menghasilkan `disagreement.csv` dan `audit_report.md`, lalu tampilkan **Confusion Matrix** model terbaik di Split C serta bedah kasus salah prediksi (misal tertukarnya `takut` vs `terkejut` atau `marah` vs `jijik`).

---

## 8. Checklist Pembagian Bahan Bab III untuk Tim

Sambil menunggu `08_agree.py` → `09_select.py` → `10_cut.py` selesai, pembagian penulisan **Bab III** yang sudah bisa dicicil:
- **Alur Penelitian & Diagram (`Maul`):** Flowchart lengkap dari `00_inventory` sampai `12_audit` + alur 2-tahap training (`Offline Feature Caching` + `Fusion Training`).
- **Dataset & Preprocessing (`Kesan` / `Bagus`):** Sumber 24 film, kriteria QC, hasil kesepakatan *kappa* (`agreement_report.md`), kuota kelas akhir (`selection_report.md`), dan skema *Split A/B/C* (`split_report.md`).
- **Rancangan Model, Ablasi, & Hyperparameter Tuning (`Michael`):** Tabel perbandingan *Model Hemat* vs *Model Maksimal* (`Teks`, `Audio`, `Visual`, `Fusi`), justifikasi optimasi VRAM 8 GB (`bf16`, `LoRA`/partial unfreeze, *gradient checkpointing*, *two-stage caching*), strategi 2 gelombang eksekusi (Bagian 6), serta tabel ruang pencarian Optuna di Bagian 5.
