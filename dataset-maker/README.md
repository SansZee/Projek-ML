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
| `label` | streamlit | 07 |
| `stats` | scikit-learn, statsmodels | analisis tambahan (tidak wajib) |
| `ml` | torch, transformers, timm, lightning, optuna | repo pelatihan |

```bash
uv sync --locked --extra asr --extra face --extra label
```

`uv` hanya mempertahankan extra yang disebut: `--extra label` saja akan
mencabut `face` dan `asr`. Sebut semuanya dalam satu perintah.

## Urutan pipeline

| # | Skrip | Masukan → Keluaran | Status |
|---|---|---|---|
| 1 | `00_inventory.py` | `raw/*.mkv` → `films.csv`, `reject_log.csv` | **selesai** |
| 2 | `01_asr.py` | film → `asr/*.json` (word timestamp) | **selesai** |
| 3 | `02_scenes.py` | film → `scenes/<id>.csv` + `scenes.csv` | **selesai** |
| 4 | `03_utterances.py` | ASR + adegan + SRT → `utterances.csv` | **selesai** |
| 5 | `04_face_screen.py` | `utterances` → `faces.csv`, embedding, `actors.csv` | **selesai** |
| 6 | `05_merge.py` | kiriman 4 laptop → `data/merged/` | **selesai** |
| 7 | `06_qc.py` | `merged` → `pool.csv`, rencana adegan | **selesai** |
| 8 | `07_annotate.py` | `pool` → `assignments.csv` + `labels_<nama>.csv` | **selesai** |
| 9 | `08_agree.py` | labels → `labels_final.csv` + kappa + adjudikasi | **selesai** |
| 10 | `09_select.py` | `labels_final` → `selected.csv` (600/kelas) | **selesai** |
| 11 | `10_cut.py` | `selected` → `clips/`, `audio/`, `faces/` | **selesai** |
| 12 | `11_split.py` | `selected` → split A/B/C | **selesai** |
| 13 | `12_audit.py` | labels vs prediksi → `disagreement.csv` | **selesai** |
| 14 | `13_stats.py` | semua → statistik + dataset card | **selesai** |

## Struktur

```
config.yaml            satu versi untuk seluruh tim, di-commit
films_registry.csv     film_id unik global
models.lock.yaml       revisi bobot model
src/emosi/             pustaka: config, media, subtitle, audio, video, tahap 05–13
scripts/               skrip per tahap, dijalankan dengan urutan nomor
data/                  keluaran tabular (gitignored)
data/merged/           katalog gabungan semua anggota (gitignored)
kiriman/               zip serah-terima antar-anggota (gitignored)
media/                 media turunan (gitignored)
media/preview/         preview adegan 480p untuk pelabelan (gitignored)
media/clips/           klip 720p, audio, wajah (gitignored)
```

## Aturan yang tidak boleh dilanggar

1. **Jangan ubah `config.yaml` di laptop sendiri.** Kalau harus berubah, ubah
   untuk semua orang lalu jelaskan di laporan.
2. **`film_id` harus unik global.** Daftarkan di `films_registry.csv` sebelum
   memproses film baru.
3. **Labeli semua, pilih nanti.** `09_select.py` dijalankan sekali secara
   terpusat, bukan per orang.
4. **Kirim lewat `--pack`, bukan salin manual.** `05_merge.py` memeriksa hash
   config dan kelengkapan berkas; kiriman manual biasanya kehilangan `.npy`.
5. **Kunci label setelah adjudikasi** (checksum dicatat di `labels_final.csv`).

Panduan langkah demi langkah ada di **`docs/panduan-pemakaian.md`**.
Pembagian kerja empat orang ada di `docs/koordinasi.md`.
