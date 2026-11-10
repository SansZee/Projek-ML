# dataset-maker

Pembangun dataset emosi multimodal (wajah, suara, dialog) dari klip film
berbahasa Inggris. Lima kelas: **senang, sedih, marah, takut, netral**.

Repo ini mengerjakan **tahap data** saja. Pelatihan model ada di repo terpisah
dan hanya membaca `clips.csv` yang dihasilkan di sini.

## Prasyarat

- `uv` (>= 0.5)
- `ffmpeg` dan `ffprobe` di `PATH`
- Python 3.11 (dipatok di `.python-version`, uv yang mengurus unduhannya)

## Mulai

```bash
uv sync --locked
uv run python -m emosi.bootstrap --verify
uv run python scripts/00_inventory.py
```

Hasilnya: `data/films.csv` dan `data/reject_log.csv`.

Untuk uji cepat tanpa deteksi wajah:

```bash
uv run python scripts/00_inventory.py --limit 2 --skip-faces --force
```

## Dependensi opsional

Core install sengaja ringan (~60 MB) supaya anggota tim yang hanya mengerjakan
data tidak perlu mengunduh 3 GB CUDA.

| Extra | Isi | Dipakai tahap |
|---|---|---|
| `asr` | faster-whisper, jiwer | 01 |
| `face` | insightface, onnxruntime-gpu | 04 |
| `label` | streamlit | 06 |
| `stats` | scikit-learn, statsmodels | 07 |
| `ml` | torch, transformers, timm, lightning, optuna | 09–12 |

```bash
uv sync --locked --extra asr --extra face --extra label
```

## Urutan pipeline

| # | Skrip | Masukan → Keluaran | Status |
|---|---|---|---|
| 1 | `00_inventory.py` | `raw/*.mkv` → `films.csv`, `reject_log.csv` | **selesai** |
| 2 | `01_asr.py` | film → `asr/*.json` (word timestamp) | **selesai** |
| 3 | `02_scenes.py` | film → `scenes/<id>.csv` + `scenes.csv` | **selesai** |
| 4 | `03_utterances.py` | ASR + adegan + SRT → `utterances.csv` | **selesai** |
| 5 | `04_face_screen.py` | `utterances` → `faces.csv`, embedding, `actors.csv` | **selesai** |
| 6 | `05_qc.py` | `utterances` → `pool.csv` | belum |
| 7 | `06_annotate.py` | `pool.csv` → `labels_A/B.csv` | belum |
| 8 | `07_agree.py` | labels → `labels_final.csv` + kappa | belum |
| 9 | `08_select.py` | `labels_final` → `selected.csv` (600/kelas) | belum |
| 10 | `09_cut.py` | `selected` → `clips/`, `audio/`, `faces/` | belum |
| 11 | `10_split.py` | `selected` → split A/B/C | belum |
| 12 | `11_audit.py` | labels vs model → `disagreement.csv` | belum |
| 13 | `12_stats.py` | semua → statistik + dataset card | belum |

## Struktur

```
config.yaml            satu versi untuk seluruh tim, di-commit
films_registry.csv     film_id unik global
models.lock.yaml       revisi bobot model
src/emosi/             pustaka: config, media, subtitle, audio, video, bootstrap
scripts/               skrip per tahap, dijalankan dengan urutan nomor
data/                  keluaran tabular (gitignored, pakai DVC)
media/                 klip, audio, wajah (gitignored)
```

## Aturan yang tidak boleh dilanggar

1. **Jangan ubah `config.yaml` di laptop sendiri.** Kalau harus berubah, ubah
   untuk semua orang lalu jelaskan di laporan.
2. **`film_id` harus unik global.** Daftarkan di `films_registry.csv` sebelum
   memproses film baru.
3. **Labeli semua, pilih nanti.** `08_select.py` dijalankan sekali secara
   terpusat, bukan per orang.
4. **Kunci label setelah adjudikasi** (checksum dicatat di `labels_final.csv`).

Panduan langkah demi langkah ada di **`docs/panduan-pemakaian.md`**.
Pembagian kerja empat orang ada di `docs/koordinasi.md`.
