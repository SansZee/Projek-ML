# Panduan Pemakaian dataset-maker

Panduan ini menjelaskan cara memakai pipeline dari nol sampai data siap
digabung. Tulisannya untuk empat orang yang bekerja di laptop masing-masing,
bukan hanya untuk satu mesin.

Status tahap per tulisan ini:

| Tahap | Skrip | Status | Keterangan |
|---|---|---|---|
| 1 | `00_inventory.py` | **tersedia** | audisi film, `films.csv` |
| 2 | `01_asr.py` | **tersedia** | transkrip + timestamp kata |
| 3 | `02_scenes.py` | **tersedia** | adegan anotasi 2–3 menit |
| 4 | `03_utterances.py` | **tersedia** | ujaran 2–10 detik |
| 5 | `04_face_screen.py` | **tersedia** | face ratio, embedding, klaster aktor |
| 6 | `05_qc.py` | belum | kolam kandidat `pool.csv` |
| 7 | `06_annotate.py` | belum | aplikasi pelabelan |
| 8 | `07_agree.py` | belum | kappa + adjudikasi |
| 9 | `08_select.py` | belum | seleksi 600/kelas |
| 10 | `09_cut.py` | belum | potong klip, audio, wajah |
| 11 | `10_split.py` | belum | split A/B/C |
| 12 | `11_audit.py` + `12_stats.py` | belum | audit dan statistik |

Tahap yang belum tersedia tetap dijelaskan di bagian akhir supaya urutan
kerjanya sudah jelas. Jangan menjalankan perintah untuk skrip yang belum ada.

---

## 1. Yang perlu disiapkan

Per orang:

- film yang kamu pegang, dalam satu folder (boleh di disk eksternal);
- subtitle Inggris untuk film itu (`.srt`), satu folder dengan filmnya;
- `uv`, `ffmpeg`, dan `ffprobe`;
- GPU NVIDIA sangat disarankan untuk `01_asr.py` dan `04_face_screen.py`.
  Tanpa GPU semuanya tetap jalan, tetapi 10–50x lebih lambat.

Satu film butuh sekitar:

| Tahap | GPU (RTX 4050) | CPU saja |
|---|---|---|
| `00_inventory.py` | 5–10 menit | 5–10 menit |
| `01_asr.py` | 15–30 menit | 4–8 jam |
| `02_scenes.py` | ~10 menit | ~10 menit (memang CPU) |
| `03_utterances.py` | < 10 detik | < 10 detik |
| `04_face_screen.py` | 8–12 menit | 1–2 jam |

Angka ini untuk film 2 jam. Empat film per orang berarti sekitar 3–4 jam
kerja GPU, jadi jangan menunda ke hari terakhir.

---

## 2. Instalasi

### 2.1 Pasang uv dan ffmpeg

```bash
# uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# ffmpeg (Arch)
sudo pacman -S ffmpeg
# Ubuntu/Debian
sudo apt install ffmpeg
```

Windows: pakai WSL2. Instalasi native bisa, tetapi `insightface` sering gagal
dikompilasi. Detailnya ada di `docs/environment.md`.

Cek:

```bash
ffmpeg -version | head -1
uv --version
```

### 2.2 Sinkronkan projek

```bash
git clone <repo> dataset-maker
cd dataset-maker
uv sync --locked
```

`--locked` wajib. Tanpa itu uv boleh menulis ulang `uv.lock`, dan empat laptop
bisa punya environment berbeda tanpa ada yang sadar.

Pasang extra sesuai tahap yang kamu kerjakan:

```bash
# data + wajah (tahap 00–05)
uv sync --locked --extra asr --extra face

# kalau nanti melabeli lewat aplikasi
uv sync --locked --extra asr --extra face --extra label
```

Extra `face` menarik `insightface`, `onnxruntime-gpu`, dan pustaka CUDA.
Ukurannya beberapa ratus MB. Di Windows, gunakan WSL2.

Terakhir, pasang pengaman git. Satu perintah, sekali saja per laptop:

```bash
git config core.hooksPath .githooks
```

Hook `pre-commit` menolak berkas ter-stage di atas 5 MB, jadi satu klip video
atau `.npy` yang nyasar tidak akan pernah masuk riwayat. `.gitignore` saja tidak
cukup: ia tidak menolong kalau ada yang memakai `git add -f`, dan berkas besar
yang sudah ter-commit tidak bisa dihapus dari riwayat tanpa menulis ulang
history. Kalau suatu saat memang perlu berkas besar:

```bash
GIT_MAX_FILE_MB=20 git commit -m "..."
```

### 2.3 Verifikasi environment

```bash
uv run python -m emosi.bootstrap --verify
```

Yang harus terlihat:

- `ffmpeg` ada;
- `uv.lock` ada (bukan `TIDAK ADA`);
- `config` menampilkan 16 karakter hash;
- kalau GPU dipakai, `torch cuda=True` (setelah extra `ml`) atau setidaknya
  provider `CUDAExecutionProvider` saat `04_face_screen.py` dijalankan.

Berkas `environment.lock.json` yang dihasilkan **diserahkan** bersama data.
Berkas ini bukti bahwa hash config dan versi pustaka sama dengan koordinator.

---

## 3. Struktur projek dan aturan config

```
config.yaml            satu versi untuk seluruh tim
films_registry.csv     daftar film_id unik global
models.lock.yaml       catatan revisi bobot model
.githooks/pre-commit   pengaman: tolak berkas > 5 MB
src/emosi/             pustaka bersama
scripts/               skrip per tahap, dijalankan berurutan
data/                  keluaran tabular (gitignored)
media/                 audio, klip, wajah (gitignored)
clips/                 klip akhir hasil potong (gitignored)
kiriman/               zip serah-terima antar-anggota (gitignored)
docs/                  panduan dan catatan koordinasi
```

Tiga aturan yang tidak boleh dilanggar:

1. **Jangan mengubah `config.yaml` di laptop sendiri.** Semua ambang ada di
   sana. Kalau harus berubah, ubah untuk semua orang dan catat di laporan.
2. **`film_id` harus unik global.** Daftarkan film di `films_registry.csv`
   sebelum diproses. Contoh:

   ```csv
   film_id,title,year,genre,lead_actors,path_contains
   ABT13,About Time,2013,romansa,Domhnall Gleeson;Rachel McAdams,About Time
   ```

   `path_contains` adalah potongan path yang cukup untuk mengenali film itu di
   laptop siapa pun. Kalau `00_inventory.py` sampai membuat id otomatis, itu
   tanda registri belum cocok — perbaiki registrinya.

3. **Jangan commit data.** `data/`, `media/`, `clips/`, `kiriman/`,
   `environment.lock.json`, dan semua ekstensi video/audio/array sudah
   diabaikan git. Yang di-commit hanya kode dan config. Aturan ini ditegakkan
   `.githooks/pre-commit`; kalau hook menolak, jangan diakali dengan
   `git add -f` — keluarkan berkasnya dengan `git rm --cached <berkas>`.

---

## 4. Alur kerja satu orang

Kerjakan berurutan. Setiap tahap membaca keluaran tahap sebelumnya.

```bash
# 1. daftarkan film di films_registry.csv, lalu:
uv run python scripts/00_inventory.py

# 2. transkrip + timestamp kata (GPU sangat disarankan)
uv run python scripts/01_asr.py

# 3. adegan anotasi
uv run python scripts/02_scenes.py

# 4. ujaran 2–10 detik
uv run python scripts/03_utterances.py

# 5. wajah + embedding + klaster aktor
uv run python scripts/04_face_screen.py
```

Skrip 01–04 punya `--film <film_id>` untuk satu film dan `--force` untuk
menimpa hasil lama. Tanpa `--force`, hasil film yang sudah ada dilewati, jadi
menjalankan ulang skrip tidak menghapus pekerjaan film lain. `00_inventory.py`
berbeda: ia menulis satu `films.csv` untuk semua film, jadi penimpaan memakai
`--force` dan tidak ada opsi `--film`.

### 4.1 `00_inventory.py` — audisi film

```bash
uv run python scripts/00_inventory.py
uv run python scripts/00_inventory.py --limit 1 --force
```

Mengukur durasi, fps, codec, jumlah cue, WPM, SNR, dan face ratio pada lima
jendela 60 detik. Film yang gagal masuk `data/reject_log.csv` beserta alasannya.

Keluaran: `data/films.csv`, `data/reject_log.csv`.

Uji cepat tanpa deteksi wajah (jauh lebih cepat):

```bash
uv run python scripts/00_inventory.py --limit 1 --skip-faces --force
```

### 4.2 `01_asr.py` — transkripsi

```bash
uv run python scripts/01_asr.py
uv run python scripts/01_asr.py --film ABT13 --seconds 180   # uji 3 menit
```

Yang dilakukan:

1. audio diekstrak ke `media/audio/<film_id>.flac`;
2. faster-whisper menulis transkrip dengan timestamp per kata;
3. teks subtitle dibandingkan dengan transkrip (WER, geseran waktu, cakupan).

Keluaran: `data/asr/<film_id>.json`, `data/asr/asr_meta.json`,
`data/asr/srt_check.csv`.

Baca `srt_check.csv`: kalau statusnya `tidak_selaras` atau `meragukan`,
subtitle kemungkinan bukan untuk potongan film itu. Jangan lanjut ke tahap 03
sebelum ini beres; teks dataset diambil dari subtitle.

VAD sengaja dimatikan. Diukur pada About Time, VAD membuang bisikan dan dialog
di bawah musik — justru materi kelas takut — dan menaikkan WER dari 0,13 ke
0,23. Teks karangan Whisper di musik tidak masuk dataset karena tahap 03 hanya
memakai kata yang jatuh di dalam cue subtitle.

### 4.3 `02_scenes.py` — adegan anotasi

```bash
uv run python scripts/02_scenes.py
uv run python scripts/02_scenes.py --film ABT13 --seconds 600 --force
```

Mendeteksi cut gambar dengan filter `scdet` ffmpeg, lalu menggabungkannya
menjadi adegan 2–3 menit yang batasnya jatuh di cut. Adegan adalah satuan
**anotasi**, bukan satuan dataset: anotator menonton satu adegan utuh supaya
label punya konteks cerita.

Keluaran: `data/scenes/<film_id>.csv`, `data/scenes.csv`,
`data/scenes/scenes_meta.json`.

### 4.4 `03_utterances.py` — ujaran 2–10 detik

```bash
uv run python scripts/03_utterances.py
uv run python scripts/03_utterances.py --film ABT13 --force
```

Menggabungkan cue subtitle yang berdekatan sampai kalimatnya utuh dan cukup
panjang, lalu mengambil awal/akhir dari kata Whisper. **Teks dari subtitle,
waktu dari ASR.**

Keluaran: `data/utterances/<film_id>.csv`, `data/utterances.csv`,
`data/utterances/utterances_meta.json`.

Ujaran di bawah 2 detik tidak dibuang di sini; tahap QC yang memutuskan, supaya
keputusannya tercatat. Di kedua film pilot ada 511 ujaran seperti itu.

### 4.5 `04_face_screen.py` — wajah dan aktor

```bash
uv run python scripts/04_face_screen.py
uv run python scripts/04_face_screen.py --film ABT13 --force
uv run python scripts/04_face_screen.py --no-cluster   # lewati klaster aktor
uv run python scripts/04_face_screen.py --recluster    # hitung ulang aktor tanpa dekode
```

Untuk setiap ujaran, tahap ini mendekode frame pada 2 fps (lebar 960 px),
mendeteksi wajah dengan RetinaFace, dan menghitung:

| Kolom | Arti | Dipakai untuk |
|---|---|---|
| `face_ratio` | bagian frame sampel yang memuat wajah layak | QC final: di bawah 0,50 wajah terlalu jarang |
| `face_width_frac` | lebar wajah dominan / lebar frame | QC: wajah terlalu kecil |
| `multi_face_share` | bagian frame dengan ≥ 2 wajah | menandai percakapan berkelompok |
| `dominant_share` | bagian frame wajah yang jelas dominan | keandalan embedding |
| `face_area_ratio` | luas wajah dominan / luas frame | ukuran wajah |
| `det_score` | median skor detektor | QC kualitas deteksi |
| `emb_frames` | jumlah frame yang menyumbang embedding | 0 berarti tanpa embedding |
| `face_ok` | `face_ratio ≥ 0,60` dan minimal 1 frame berwajah | saringan awal yang lebih ketat dari QC |
| `cluster`, `actor_key` | klaster aktor lokal per film | split C |

Embedding 512 dimensi **tidak** masuk CSV. Ia disimpan di
`data/faces/<film_id>.npy`, sejajar baris CSV. Kolom `emb_path` dan `emb_row`
menunjukkan baris mana yang dipakai.

Keluaran:

```
data/faces/<film_id>.csv
data/faces/<film_id>.npy
data/faces.csv
data/actors.csv
data/faces/faces_meta.json
```

`data/actors.csv` berisi satu baris per klaster:

```csv
actor_key,film_id,cluster,n_utterances,median_face_ratio,median_emb_frames,lead_actors,actor_label
ABT13-A01,ABT13,1,493,0.98,6,Domhnall Gleeson;Rachel McAdams,
```

**Isi `actor_label` manual** dengan nama aktor. Ini yang membuat split C bisa
dipakai: aktor yang sama tidak boleh muncul di train dan test. Kalau aktor yang
sama muncul di dua film (mis. Rachel McAdams di About Time dan The Notebook),
pakai ejaan yang **persis sama** di kedua baris.

Klaster adalah usulan, bukan kebenaran. Periksa dengan membuka beberapa ujaran
dari klaster yang sama. Klaster dengan `n_utterances` kecil biasanya wajah salah
orang atau profil buram; biarkan `actor_label` kosong dan jangan dipakai split C.

Hasil pada dua film pilot (RTX 4050): 2.467 ujaran, 28.246 frame sampel,
`face_ok` 78%, embedding tersedia untuk 82% ujaran, dan 16–17 klaster aktor per
film. Prosesnya 8–11 menit per film. Angka ini akan berbeda pada film lain,
terutama film horor yang banyak adegan gelap.

Kalau ambang klaster ingin diubah, jangan mengulang dekode video: cukup
`--recluster`, yang membaca `faces/<film_id>.csv` dan `.npy` lalu menulis ulang
`cluster`, `actor_key`, dan `actors.csv` dalam hitungan detik.

Uji cepat dua adegan tanpa menyentuh data asli:

```bash
uv run python scripts/04_face_screen.py --film ABT13 --scenes 2 \
  --out-dir data/uji --force
```

Hasilnya di `data/uji/`, aman dihapus. Kalau `provider onnxruntime` yang
tercetak `CPUExecutionProvider`, GPU tidak terpakai — lihat bagian
troubleshooting.

---

## 5. Membaca keluaran

### 5.1 `films.csv`

Satu baris per film. Kolom penting: `film_id`, `title`, `year`, `genre`,
`duration_s`, `fps`, `srt_cues`, `wpm`, `snr_db`, `face_ratio`,
`pass_audition`, `reject_reason`.

### 5.2 `utterances.csv`

Satu baris per ujaran:

| Kolom | Arti |
|---|---|
| `utterance_id` | id unik, mis. `ABT13-U00001` |
| `scene_id` | adegan tempat ujaran berada |
| `start_s`, `end_s` | waktu di film, detik |
| `duration_s` | durasi klip |
| `text` | teks dari subtitle, sudah dibersihkan |
| `n_words` | jumlah kata |
| `timing` | `asr` (waktu dari kata Whisper) atau `cue` (mundur ke waktu subtitle) |
| `speaker` | nama pembicara kalau subtitle memberinya |

### 5.3 `faces.csv`

Satu baris per ujaran, urutan kolom dijelaskan di bagian 4.5. Baris dengan
`emb_frames = 0` tetap ada; artinya ujaran itu tidak punya wajah yang bisa
dipakai untuk split C.

### 5.4 `actors.csv`

Daftar klaster aktor per film. `actor_label` diisi manual; `actor_key` tidak
boleh diubah karena dipakai `faces.csv`.

---

## 6. Tahap pelabelan (belum tersedia)

Rencana kerjanya, supaya kamu bisa menyiapkan waktu:

1. `05_qc.py` menyaring `utterances.csv` + `faces.csv` menjadi `pool.csv`:
   durasi, jumlah kata, SNR, face ratio, dan klip yang rusak.
2. `06_annotate.py` menampilkan satu adegan utuh beserta semua ujarannya.
   Anotator memilih satu dari lima kelas: **senang, sedih, marah, takut,
   netral**, atau `ambigu` / `tidak_layak`, dengan keyakinan 1–5.
3. Setiap orang melabeli **150 ujaran tukar silang** dengan anggota lain.
   Tanpa overlap, kappa tidak bisa dihitung.
4. `07_agree.py` menghitung kappa dan menghasilkan `labels_final.csv`.

Aturan pelabelan lengkap ada di `docs/label_guideline.md`. Baca sebelum
melabeli; label yang berubah setelah eksperimen dimulai membatalkan semua
hasil.

---

## 7. Seleksi, potong, dan split (belum tersedia)

Urutannya nanti:

```bash
uv run python scripts/08_select.py    # 600 per kelas, terpusat
uv run python scripts/09_cut.py       # potong klip di laptop pemilik film
uv run python scripts/10_split.py     # split A/B/C, terpusat
```

`08_select.py` dijalankan **sekali** oleh koordinator. Kalau setiap orang
menyeleksi sendiri, kuota kelas habis di film yang sama.

`09_cut.py` dijalankan per orang karena berkas film tetap di laptop
masing-masing. Yang dikirim hanya daftar `clip_id`.

`10_split.py` memakai `actors.csv` untuk varian C. Tiga varian split:

- **A** — acak per ujaran;
- **B** — per film;
- **C** — per aktor.

Ketiganya dilaporkan supaya terlihat berapa besar kebocoran yang disebabkan
aktor yang sama muncul di train dan test.

---

## 8. Serah terima ke koordinator

Kirim hanya berkas tabular dan catatan environment:

```
films.csv
utterances.csv
faces.csv
actors.csv
data/faces/<film_id>.csv
data/faces/<film_id>.npy
environment.lock.json
config.yaml (kalau diminta memastikan hash)
```

Jangan kirim `media/`, `clips/`, atau `audio/`; ukurannya besar dan bisa dibuat
ulang di laptop pemilik film.

Sebelum menyerahkan:

```bash
uv run python -m emosi.bootstrap --verify
sha256sum config.yaml
```

Hash `config` harus sama dengan koordinator. Kalau berbeda, jangan digabung
dulu — cari tahu ambang mana yang berubah.

Koordinator menggabungkan per film, bukan menempel `*.csv` mentah. Potongan
yang aman:

```python
import pandas as pd
from pathlib import Path

bagian = [pd.read_csv(p) for p in Path("kiriman").glob("*/utterances.csv")]
gabung = pd.concat(bagian).drop_duplicates("utterance_id", keep="last")
gabung.to_csv("data/utterances.csv", index=False)
```

Lakukan hal yang sama untuk `faces.csv` dengan kunci `utterance_id`, dan untuk
`actors.csv` dengan kunci `actor_key`. Berkas `.npy` disalin per film ke
`data/faces/`; jangan menggabungkannya, karena `emb_row` merujuk baris di
berkas film masing-masing.

---

## 9. Troubleshooting

| Gejala | Sebab | Tindakan |
|---|---|---|
| `uv sync --locked` gagal "lock is outdated" | `pyproject.toml` berubah tanpa `uv lock` | `uv lock`, commit ulang, minta semua orang sync |
| `ffmpeg tidak ditemukan` | belum terpasang / bukan di PATH | pasang, buka shell baru |
| `01_asr.py` sangat lambat | GPU tidak terpakai | cek `torch.cuda.is_available()`; lihat `docs/environment.md` |
| `provider onnxruntime: CPUExecutionProvider` | pustaka CUDA tidak lengkap | `uv sync --locked --extra face`; cek `nvidia-smi` |
| `CascadeClassifier` kosong / `cv2` error | OpenCV 5 terpasang | `uv sync --locked --extra face` (pin `<5` sudah ada); jangan pasang opencv sendiri |
| `insightface` gagal dikompilasi (Windows) | butuh compiler C++ | pakai WSL2 |
| `faces.csv` kosong | film gelap / wajah jarang / frame gagal didekode | cek `faces_meta.json`, jalankan `--scenes 2` untuk uji |
| `srt_check.csv` `tidak_selaras` | subtitle bukan untuk film itu | cari SRT yang benar sebelum lanjut |
| `film_id` otomatis (`FLM…`) | registri tidak cocok | perbaiki `films_registry.csv`, jalankan `00_inventory.py --force` |
| Dua laptop punya `F01` | id tidak unik global | pakai pola tiga huruf + tahun, mis. `ABT13` |
| Hasil film hilang setelah rerun | menjalankan tanpa `--film` dan tanpa `--force`? | hasil lama memang dipertahankan; cek pesan "dilewati" |
| `commit dibatalkan: ada berkas di atas 5 MB` | hook `pre-commit` bekerja | keluarkan dengan `git rm --cached <berkas>`; jangan `git add -f` |

---

## 10. Checklist per film

- [ ] film terdaftar di `films_registry.csv` dengan `path_contains` yang benar
- [ ] subtitle Inggris ada satu folder dengan film
- [ ] `00_inventory.py` → `pass_audition = True`
- [ ] `01_asr.py` → status `selaras` di `srt_check.csv`
- [ ] `02_scenes.py` → adegan 90–240 detik, tidak ada yang tumpang-tindih
- [ ] `03_utterances.py` → mayoritas ujaran `timing = asr`
- [ ] `04_face_screen.py` → `face_ok` cukup, `provider` GPU
- [ ] `actors.csv` → `actor_label` diisi untuk aktor utama
- [ ] `bootstrap --verify` bersih
- [ ] `config.yaml` hash sama dengan koordinator

## 11. Checklist sebelum menyerahkan data

- [ ] semua film yang kamu pegang selesai tahap 00–04
- [ ] `environment.lock.json` ikut
- [ ] tidak ada `config.yaml` yang diubah sendiri
- [ ] `actors.csv` sudah diisi nama aktor
- [ ] 150 ujaran tukar silang sudah dilabeli (setelah tahap 06 tersedia)
- [ ] berkas yang dikirim sesuai daftar bagian 8
