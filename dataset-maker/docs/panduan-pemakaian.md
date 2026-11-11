# Panduan Pemakaian dataset-maker

Panduan ini menjelaskan cara memakai pipeline dari nol sampai data siap
digabung. Tulisannya untuk empat orang yang bekerja di laptop masing-masing,
bukan hanya untuk satu mesin.

Status tahap per tulisan ini:

| Skrip | Status | Keterangan |
|---|---|---|
| `00_inventory.py` | **tersedia** | audisi film, `films.csv` |
| `01_asr.py` | **tersedia** | transkrip + timestamp kata |
| `02_scenes.py` | **tersedia** | adegan anotasi 2–3 menit |
| `03_utterances.py` | **tersedia** | ujaran 2–10 detik |
| `04_face_screen.py` | **tersedia** | face ratio, embedding, klaster aktor |
| `05_merge.py` | **tersedia** | bungkus kiriman, katalog gabungan |
| `06_qc.py` | **tersedia** | kolam kandidat `pool.csv` + rencana adegan |
| `07_annotate.py` | **tersedia** | penugasan, preview, aplikasi pelabelan |
| `08_agree.py` | **tersedia** | kappa + adjudikasi |
| `09_select.py` | **tersedia** | seleksi 600/kelas |
| `10_cut.py` | **tersedia** | potong klip, audio, wajah |
| `11_split.py` | **tersedia** | split A/B/C |
| `12_audit.py` + `13_stats.py` | **tersedia** | audit prediksi dan dataset card |

Semua skrip di atas tersedia. Jalankan berurutan; setiap tahap membaca
keluaran tahap sebelumnya.

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

Pilih satu jalur dan jangan campur perintahnya: **PowerShell** untuk Windows,
**bash/zsh** untuk Linux, macOS, atau WSL2.

### 2.1 Pasang uv, ffmpeg, dan git

**Linux / macOS / WSL2**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# ffmpeg
sudo pacman -S ffmpeg          # Arch
sudo apt install ffmpeg        # Ubuntu/Debian
brew install ffmpeg            # macOS
```

**Windows (PowerShell, bukan `cmd`)**

```powershell
winget install --id=astral-sh.uv -e
winget install --id=Gyan.FFmpeg -e
winget install --id=Git.Git -e
```

Tutup lalu buka lagi PowerShell setelah instalasi. `winget` menaruh `uv` di
`%USERPROFILE%\.local\bin` dan ffmpeg di folder baru yang masuk `PATH`; tanpa
restart, `uv` atau `ffmpeg` masih "tidak dikenal".

Kalau `winget` tidak tersedia (Windows 10 lama), perbarui **App Installer**
dari Microsoft Store, atau pakai installer manual dari situs uv dan
[gyan.dev](https://www.gyan.dev/ffmpeg/builds/) lalu tambahkan folder `bin` ke
`PATH`.

Cek:

```powershell
ffmpeg -version
ffprobe -version
uv --version
git --version
```

`ffprobe` harus ikut; sebagian paket ffmpeg hanya membawa `ffmpeg`.

Sekali saja, sebelum meng-clone: izinkan path panjang. Folder projek yang
berlapis (mis. `C:\kuliah\semester7\ml\projek\dataset-maker`) bisa melewati
batas 260 karakter Windows.

```powershell
git config --global core.longpaths true
```

> **Jangan taruh projek di folder OneDrive.** Sinkronisasi saat ffmpeg menulis
> ribuan frame memperlambat semuanya dan kadang mengunci berkas di tengah
> proses. Pakai `C:\kuliah\dataset-maker` atau serupa.

### 2.2 Sinkronkan projek

```bash
git clone <repo> dataset-maker
cd dataset-maker
uv sync --locked
```

`--locked` wajib. Tanpa itu uv boleh menulis ulang `uv.lock`, dan empat laptop
bisa punya environment berbeda tanpa ada yang sadar.

Python tidak perlu dipasang sendiri: `.python-version` mematok 3.11 dan uv
mengunduhnya.

Pasang extra sesuai tahap yang kamu kerjakan:

```bash
# data + wajah (skrip 00–05)
uv sync --locked --extra asr --extra face

# kalau nanti melabeli lewat aplikasi
uv sync --locked --extra asr --extra face --extra label
```

Sebut **semua** extra yang dipakai dalam satu perintah. `uv` hanya
mempertahankan extra yang disebut, jadi `--extra label` saja akan mencabut
`face` dan `asr`.

Extra `face` menarik `insightface`, `onnxruntime-gpu`, dan pustaka CUDA,
ukurannya beberapa ratus MB.

#### Windows: mengompilasi `insightface`

`insightface` tidak menyediakan wheel Windows, jadi `--extra face` butuh
compiler C++. Sekali saja per laptop:

```powershell
winget install --id=Microsoft.VisualStudio.2022.BuildTools -e
```

Di installer yang terbuka, centang **Desktop development with C++**, lalu
jalankan ulang `uv sync`. Kalau tetap gagal, kerjakan tahap 04 di WSL2; sisa
pipeline tetap bisa jalan di Windows.

#### Windows: line ending

`.gitattributes` di repo sudah memaksa LF untuk semua berkas teks, jadi hook
`pre-commit` (skrip bash) tidak berubah menjadi CRLF walaupun Git for Windows
memakai `core.autocrlf=true`. Tidak ada yang perlu disetel.

Kalau repo sempat di-clone **sebelum** berkas itu ada dan hook gagal dengan
`bad interpreter`, checkout ulang sekali:

```powershell
git rm --cached -r .
git reset --hard
```

#### Windows: GPU untuk tahap 04

Paket CUDA di `pyproject.toml` sengaja hanya untuk Linux
(`sys_platform == 'linux'`). Di Windows, `onnxruntime-gpu` memakai CUDA dan
cuDNN yang dipasang sendiri:

1. pasang driver NVIDIA terbaru;
2. pasang **CUDA Toolkit 12.6** dan **cuDNN 9.1 untuk Windows**;
3. pastikan folder `bin` CUDA dan `bin` cuDNN ada di `PATH`;
4. cek:

```powershell
uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Hasil yang benar memuat `CUDAExecutionProvider`. Kalau tidak, tahap 04 tetap
jalan di CPU, tetapi 1–2 jam per film, bukan 5–10 menit.

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

Di PowerShell:

```powershell
$env:GIT_MAX_FILE_MB = 20
git commit -m "..."
```

### 2.3 Verifikasi environment

```bash
uv run python -m emosi.bootstrap --verify
```

Yang harus terlihat:

- `ffmpeg` ada;
- `uv.lock` ada (bukan `TIDAK ADA`);
- `config` menampilkan 16 karakter hash;
- kalau GPU dipakai, provider `CUDAExecutionProvider` muncul saat
  `04_face_screen.py` dijalankan (Windows: lihat bagian GPU di atas).

Berkas `environment.lock.json` yang dihasilkan **diserahkan** bersama data.
Berkas ini bukti bahwa hash config dan versi pustaka sama dengan koordinator.

---

## 3. Struktur projek dan aturan config

```
config.yaml            satu versi untuk seluruh tim
films_registry.csv     daftar film_id unik global
models.lock.yaml       catatan revisi bobot model
.gitattributes         LF untuk teks, biner tidak dikonversi
.githooks/pre-commit   pengaman: tolak berkas > 5 MB
src/emosi/             pustaka bersama
scripts/               skrip per tahap, dijalankan berurutan
data/                  keluaran tabular (gitignored)
media/preview/         preview adegan untuk pelabelan (gitignored)
media/clips/           klip 720p hasil potong (gitignored)
media/audio/           audio 16 kHz mono (gitignored)
media/faces/           wajah 224×224 hasil potong (gitignored)
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

# 6. bungkus hasil sendiri jadi satu zip kiriman
uv run python scripts/05_merge.py --pack <namamu>
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

### 4.6 `05_merge.py` — bungkus kiriman sendiri

```bash
uv run python scripts/05_merge.py --pack kesan
```

Membungkus hasil tahap 00–04 menjadi satu zip di
`kiriman/<nama>-<tanggal>.zip`. Isinya tabel, embedding wajah, dan
`handover.json` (pemilik, waktu, hash `config.yaml` dan `uv.lock`) — sekitar
4 MB untuk dua film, sekitar 40 MB untuk delapan film. Film, audio, dan klip
tidak pernah ikut.

Langkah ini wajib sebelum mengirim apa pun. Kiriman manual hampir selalu
kehilangan satu hal: `.npy`, `actors.csv`, atau `handover.json`.

Kalau ada film yang belum punya `faces/<film_id>.npy`, skrip berhenti dan
menyebut film mana yang belum selesai.

---

## 5. Menggabungkan kiriman (koordinator)

Dijalankan sekali oleh satu orang, di laptop yang menyimpan katalog. Anggota
tidak menjalankan bagian ini.

```bash
uv run python scripts/05_merge.py --from kiriman/
```

Isi `kiriman/` boleh zip semuanya, atau folder hasil ekstrak; keduanya
dikenali, dan kiriman yang sama tidak dihitung dua kali.

Yang dilakukan skrip:

1. membaca setiap kiriman, lalu mencocokkan hash `config.yaml` dengan config
   lokal;
2. memeriksa kelengkapan: `film_id` unik, `utterance_id` unik, jumlah baris
   `.npy` sama dengan `faces/<film_id>.csv`, dan setiap ujaran punya baris
   wajah;
3. membuang film yang `pass_audition = False`;
4. menulis katalog gabungan ke `data/merged/`;
5. membuat `merge_report.md` dan `merge_meta.json`.

Keluaran:

```
data/merged/
  films.csv          films.csv + kolom owner, tanpa kolom path
  utterances.csv
  faces.csv
  actors.csv
  scenes.csv
  reject_log.csv
  faces/<film_id>.csv
  faces/<film_id>.npy
  merge_report.md
  merge_meta.json
```

Kolom `path` dan `srt_path` dibuang karena hanya berarti di laptop pemilik
film; penggantinya `owner`. Setiap laptop menerjemahkan `film_id` ke path-nya
sendiri lewat `data/films.csv` lokal, jadi berkas lokal tidak pernah ditimpa.

### 5.1 Pemeriksaan yang menggagalkan penggabungan

Kalau ada satu error saja, **tidak ada berkas yang ditulis**, dan
`data/merged/` yang lama tidak tersentuh.

| Pemeriksaan | Kalau gagal |
|---|---|
| hash `config.yaml` berbeda | berhenti — samakan config, lalu kirim ulang |
| `film_id` datang dari dua anggota | berhenti — atau `--keep <nama>` |
| `utterance_id` ganda | berhenti |
| `.npy` tidak sama dengan baris `faces/<film_id>.csv` | berhenti |
| ujaran tanpa baris wajah | berhenti — tahap 04 belum selesai |
| `uv.lock` berbeda | peringatan |
| embedding dibuat dengan config lama | peringatan |

Film yang sama dari dua orang adalah kesalahan paling mahal: dataset membengkak
dan film itu bisa masuk train dan test sekaligus. Karena itu defaultnya
berhenti, bukan memilih sendiri.

### 5.2 Film ganda

```bash
uv run python scripts/05_merge.py --from kiriman/ --keep budi
```

Dipilih hanya setelah kamu yakin kiriman siapa yang benar. Kalau film ganda
tidak melibatkan `budi`, penggabungan tetap berhenti.

### 5.3 Membaca `merge_report.md`

Isinya ringkasan per kontributor, daftar film di katalog, film yang tidak lolos
audisi, hasil pemeriksaan, dan **kandidat aktor lintas film**.

Bagian terakhir yang perlu perhatian. Ia mendaftar pasangan `actor_key` dari
film berbeda yang wajahnya mirip. Pada dua film pilot, metodenya menemukan satu
pasangan berjarak 0,215 — Rachel McAdams di About Time dan The Notebook. Setelah
diperiksa manual, beri `actor_label` yang **persis sama** di kedua baris; kalau
tidak, split C akan bocor.

Kandidat hanyalah usulan: jarak 0,4 belum tentu orang yang sama, dan aktor yang
sama dengan tata rias berbeda bisa berjarak lebih jauh.

### 5.4 Menjalankan ulang

`data/merged/` selalu ditulis ulang dari nol, jadi jalankan lagi setiap ada
kiriman baru. Yang lama diganti, bukan ditambah, sehingga tidak ada film yang
terhitung dua kali.

### 5.5 `06_qc.py` — kolam kandidat dan rencana adegan

```bash
uv run python scripts/06_qc.py
```

Menyaring `data/merged/` menjadi **`data/pool.csv`**. Tahap ini tidak membuka
media sama sekali: `face_ratio` sudah diukur tahap 04, `snr_db` dan `clip_ratio`
sudah diukur tahap 00 per film.

| Pemeriksaan | Ambang | Tingkat |
|---|---|---|
| teks tidak kosong | — | ujaran |
| bukan lirik | tanpa `#`, `♪`, `♫` | ujaran |
| `n_words` | ≥ `qc.min_words` (2) | ujaran |
| `face_ratio` | ≥ `qc.face_ratio_min` (0,50) | ujaran |
| `snr_db` | ≥ `qc.snr_min_db` (15) | film |
| `clip_ratio` | ≤ `qc.clip_ratio_max` (0,01) | film |

Ambang wajah di sini lebih longgar daripada tahap 04 (0,60). Tahap 04 menyaring
cepat supaya embedding tidak dihitung dari frame buruk; tahap 06 memutuskan apa
yang benar-benar dilabeli, dan wajah di separuh frame masih cukup untuk dinilai
manusia.

**Rencana adegan.** Pelabelan berjalan per adegan utuh, bukan per klip lepas,
jadi yang dibatasi adalah adegan yang diambil. Batasnya diturunkan dari
`selection`: satu film tidak boleh menyumbang lebih dari 15% tiap kelas, jadi
`600 × 0,15 × 5 = 450` ujaran per film. Film horor/thriller mendapat 1,5×
(675), karena di situ kelas takut dan marah paling banyak; tanpa kelonggaran
itu kuota dua kelas tersebut tidak akan terpenuhi.

Adegan dipilih dengan langkah tetap sepanjang film, supaya tidak menumpuk di
babak pertama. Satu adegan bisa melewati batas sedikit: memotong adegan di
tengah menghilangkan konteks yang dibutuhkan anotator.

Keluaran:

```
data/pool.csv          satu baris per ujaran + qc_pass, qc_reason, label_plan
data/pool_scenes.csv   adegan terpilih dan jumlah ujaran layaknya
data/qc_report.md      ringkasan ambang, per film, dan alasan gagal
data/qc_meta.json      ambang, batas, hash config
```

`label_plan = True` berarti ujaran itu lolos QC **dan** adeganya terpilih. Di
aplikasi pelabelan adegan tetap diputar utuh, tetapi ujaran yang gagal QC tidak
perlu diklik karena tidak akan lolos seleksi. Ujaran yang lolos QC tetapi
adegannya tidak terpilih tetap ada di `pool.csv`, jadi rencana bisa diubah
tanpa mengulang tahap 05.

Opsi:

```bash
uv run python scripts/06_qc.py --cap 300        # batas sama untuk semua film
uv run python scripts/06_qc.py --no-cap         # rencanakan semua adegan
uv run python scripts/06_qc.py --rare-multiplier 2.0
```

Hasil pada tiga film (About Time, 500 Days of Summer, The Notebook): 3.459
ujaran diperiksa, 2.608 lolos QC (75%), 1.373 masuk rencana label dalam 66
dari 133 adegan (sekitar 165 menit tontonan), dan 1.246 di antaranya punya
embedding. Yang paling banyak menggugurkan adalah `wajah_jarang` (642),
`kata_sedikit` (174), dan `lirik` (132 — semuanya dari 500 Days of Summer,
yang banyak menyisipkan lagu).

`data/pool.csv` dan `data/pool_scenes.csv` berada di luar `data/merged/`, jadi
tidak ikut terhapus saat penggabungan diulang.

---

## 6. Membaca keluaran

### 6.1 `films.csv`

Satu baris per film. Kolom penting: `film_id`, `title`, `year`, `genre`,
`duration_s`, `fps`, `srt_cues`, `wpm`, `snr_db`, `face_ratio`,
`pass_audition`, `reject_reason`.

### 6.2 `utterances.csv`

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

### 6.3 `faces.csv`

Satu baris per ujaran, urutan kolom dijelaskan di bagian 4.5. Baris dengan
`emb_frames = 0` tetap ada; artinya ujaran itu tidak punya wajah yang bisa
dipakai untuk split C.

### 6.4 `actors.csv`

Daftar klaster aktor per film. `actor_label` diisi manual; `actor_key` tidak
boleh diubah karena dipakai `faces.csv`.

### 6.5 `pool.csv`

Hasil `06_qc.py`: satu baris per ujaran dari semua film, sudah digabung dengan
`face_ratio`, `snr_db`, dan `clip_ratio`. Kolom yang menentukan:

| Kolom | Arti |
|---|---|
| `qc_pass` | `True` kalau lolos semua ambang bagian 5.5 |
| `qc_reason` | alasan gagal, dipisah `;`, mis. `wajah_jarang:0.31;kata_sedikit:1` |
| `label_plan` | `True` kalau perlu dilabeli (lolos QC dan adeganya terpilih) |
| `emb_frames` | `0` berarti tidak ada embedding; masih bisa dipakai untuk audio/teks |

`pool_scenes.csv` memuat satu baris per adegan: `n_qc_pass` (ujaran layak di
adegan itu) dan `selected` (masuk tugas pelabelan). Aplikasi pelabelan membaca
kedua berkas ini, bukan `data/merged/`.

### 6.6 Keluaran seleksi, split, dan laporan

Dijelaskan cara membuatnya di bagian 8; arti berkasnya:

| Berkas | Isi |
|---|---|
| `assignments.csv` | tugas per anotator (`milik`, `gold`, `silang`) |
| `labels_<nama>.csv` | label mentah per anotator, satu baris per klik |
| `labels_final.csv` | label akhir setelah mayoritas/adjudikasi |
| `adjudication.csv` | ujaran tanpa mayoritas + kolom `decision` |
| `selected.csv` | klip terpilih + `clip_id`, label, durasi, aktor |
| `clips.csv` | katalog media: path relatif, ukuran, status potong |
| `splits.csv` | `split_A`, `split_B`, `split_C` untuk tiap klip |
| `disagreement.csv` | label vs prediksi model, urut skor |
| `dataset_card.md` | ringkasan dataset untuk lampiran laporan |

---

## 7. Pelabelan

Pelabelan berjalan **lokal di laptop pemilik film**, bukan di-deploy ke
internet. Aplikasi terpusat harus bisa membaca film, dan itu berarti memindah
ratusan GB ke satu mesin. Yang berpindah hanya tabel 1 MB, preview adegan
beberapa puluh MB untuk tukar silang, dan label 300 KB per orang.

Tiga peran:

| Siapa | Perintah | Kapan |
|---|---|---|
| koordinator | `--assign` | sekali, setelah `06_qc.py` |
| pemilik film | `--prepare`, `--prepare --zip` | sebelum sesi |
| anotator | `streamlit run` | saat sesi |

### 7.1 Membagi tugas

```bash
uv run python scripts/07_annotate.py --assign --people kesan,budi,ani,dina
```

Menulis `data/assignments.csv` dengan tiga jenis tugas:

| Peran | Isi | Tujuan |
|---|---|---|
| `milik` | semua adegan terpilih di film yang ada di laptop sendiri | mengisi kuota |
| `gold` | dua adegan yang dinilai **semua** orang | kappa global |
| `silang` | 150 ujaran milik orang lain | kappa antarpasangan |

Adegan gold dipilih yang cukup berisi (≥ 8 ujaran) dan, kalau ada, dari genre
langka. Tukar silang memakai **satu set per pemilik film**: semua orang yang
menilai film itu memakai 150 ujaran yang sama, jadi kappa bisa dibandingkan
berpasangan.

Ujaran yang sama tidak muncul dua kali untuk satu anotator: prioritasnya
`milik` → `gold` → `silang`.

Hasil pada pilot (tiga film, semuanya di laptop kesan):

| Anotator | milik | gold | silang | total |
|---|---:|---:|---:|---:|
| kesan | 1.373 | 0 | 0 | 1.373 |
| budi | 0 | 54 | 150 | 204 |
| ani | 0 | 54 | 150 | 204 |
| dina | 0 | 54 | 150 | 204 |

Kesan tidak punya tugas silang karena tidak ada film orang lain di katalog.
Kappa tetap bisa dihitung: gold set dinilai semua orang, dan ketiga penilai
lain memakai set 150 ujaran yang sama dari film kesan.

Kirim `data/pool.csv`, `data/pool_scenes.csv`, dan `data/assignments.csv` ke
semua anotator. Ketiganya di bawah 1 MB.

### 7.2 Memotong preview adegan

Browser tidak bisa memutar `.mkv`, jadi tiap adegan dipotong jadi mp4 480p
dengan audio tetap ada.

```bash
uv run python scripts/07_annotate.py --prepare
```

Memotong semua adegan yang dibutuhkan siapa pun dan filmnya ada di laptop ini.
Satu adegan 2–3 menit menjadi 5–6 MB, jadi 66 adegan pilot sekitar 370 MB.

Kalau hanya ingin adegan milik sendiri:

```bash
uv run python scripts/07_annotate.py --prepare --annotator kesan
```

Untuk mengirim preview adegan yang dibutuhkan orang lain (gold dan tukar
silang):

```bash
uv run python scripts/07_annotate.py --prepare --zip budi
```

Hasilnya `kiriman/preview-untuk-budi-<tanggal>.zip`. Penerima mengekstraknya
sehingga isi `preview/` masuk ke `media/preview/`. Satu adegan sekitar 5 MB,
jadi bungkus untuk satu orang biasanya 40–60 MB.

Opsi lain: `--force` (potong ulang), `--limit N` (uji cepat), `--scale 480`,
`--crf 30` (makin besar makin kecil berkasnya).

### 7.3 Membuka aplikasi

```bash
uv sync --locked --extra asr --extra face --extra label
uv run streamlit run scripts/07_annotate.py
```

`uv` hanya mempertahankan extra yang disebut. Menjalankan `--extra label`
saja akan **mencabut** `face` dan `asr` dari laptop yang sama, sehingga
`00_inventory.py` gagal dengan `No module named 'cv2'`. Sebut semuanya.

Aplikasi membaca `pool.csv` dan `assignments.csv` dari `data/`, dan preview
dari `media/preview/`. Alurnya:

1. pilih nama di sidebar;
2. pilih adegan; daftar adegan menampilkan progres tiap adegan;
3. video mulai dari ujaran yang sedang dinilai — toggle **Mulai dari awal
   adegan** untuk menonton konteks lebih dulu;
4. klik satu dari lima kelas, atau `ambigu` / `tidak_layak`;
5. atur keyakinan 1–5 dan catatan kalau perlu. Setelah satu label diklik,
   aplikasi pindah ke ujaran berikutnya yang belum dilabeli.

Label ditulis ke `data/labels_<nama>.csv` **setiap klik**, lewat berkas
sementara lalu diganti, jadi sesi bisa ditutup kapan saja dan dilanjutkan
besok. Kolomnya:

```
annotator, utterance_id, film_id, scene_id, role, label, confidence, notes, labeled_at
```

`role` (`milik` / `gold` / `silang`) ikut disimpan supaya `08_agree.py` bisa
melaporkan kappa per jenis tugas, bukan hanya satu angka gabungan.

Nama berkas memakai nama anotator, bukan nama mesin. Kalau satu orang sempat
memakai dua laptop, `08_agree.py` tetap menggabungkannya lewat kunci
`(annotator, utterance_id)`.

Setelah sesi selesai, preview boleh dihapus: `media/preview/` hanya alat
bantu, dan klip final dibuat `10_cut.py` dari film aslinya.

### 7.4 Urutan sesi

1. **Kalibrasi** — tonton beberapa adegan bersama, sepakati contoh tiap kelas.
   Diskusi boleh.
2. **Melabeli sendiri** — tidak ada diskusi. Ini yang membuat kappa berarti.
   Kerjakan gold set lebih dulu supaya kesepakatan bisa dicek di tengah sesi.
3. **Adjudikasi** — setelah semua selesai, bahas yang beda. `08_agree.py`
   menandai ujaran yang perlu diputuskan.

Aturan pelabelan lengkap ada di `docs/label_guideline.md`. Baca sebelum
melabeli; label yang berubah setelah eksperimen dimulai membatalkan semua
hasil.

### 7.5 Kesepakatan dan adjudikasi

```bash
uv run python scripts/08_agree.py
```

Membaca semua `data/labels_*.csv` yang dikumpulkan dari para anotator, lalu:

1. menggabungkan label dengan kunci `(annotator, utterance_id)` — satu orang
   yang sempat memakai dua laptop tetap dihitung sebagai satu penilai;
2. menghitung **Cohen's kappa** untuk setiap pasangan penilai dan **Fleiss'
   kappa** untuk ujaran yang dinilai tiga orang atau lebih;
3. memutuskan label akhir dengan suara mayoritas;
4. menulis daftar ujaran yang tidak punya mayoritas ke
   `data/adjudication.csv`.

Kappa dikelompokkan per jenis tugas: `gold`, `silang`, `milik`, dan `semua`.
Hanya tugas `gold` dan `silang` yang punya ujaran dinilai lebih dari satu
orang, jadi di situlah kappa benar-benar terukur.

Kappa dihitung dari **label mentah**, bukan setelah adjudikasi. Kalau dihitung
setelah keputusan koordinator, angkanya naik sendiri dan tidak lagi jujur.

Aturan label akhir:

| Status | Arti | Masuk `labels_final.csv` |
|---|---|---|
| `tunggal` | hanya satu penilai | ya |
| `sepakat` | semua penilai sama | ya |
| `mayoritas` | satu label > 50% penilai | ya |
| `adjudikasi` | diputuskan manusia | ya |
| `perlu_adjudikasi` | tidak ada mayoritas | belum |

Keluaran:

```
data/labels_final.csv     utterance_id, label, confidence, n_raters, agreement, status
data/adjudication.csv     ujaran tanpa mayoritas + kolom decision
data/agreement_report.md  cakupan, matriks irisan, tabel kappa
data/agreement_meta.json  angka kappa dan jumlah butir
```

Koordinator menulis laporan kappa dari `agreement_report.md`. Tafsirnya
memakai Landis & Koch: di bawah 0,20 kesepakatan sedikit, 0,20–0,40 cukup,
0,40–0,60 sedang, 0,60–0,80 baik, di atas 0,80 sangat baik.

**Adjudikasi.** Buka `data/adjudication.csv`, isi kolom `decision` dengan salah
satu kelas, `ambigu`, atau `tidak_layak`, isi `decided_by`, lalu jalankan
skrip ini lagi. Keputusan yang sudah diisi tetap tersimpan di
`adjudication.csv`, jadi riwayatnya tidak hilang. Kalau label mentah berubah
dan keputusan lama ingin dibuang, pakai `--reset-decisions`.

Ujaran yang belum diputuskan tidak masuk `labels_final.csv`, sehingga tahap
seleksi tidak akan mengambilnya. Jumlahnya dicetak di akhir skrip.

Kolom `confidence` pada baris `adjudikasi` adalah rata-rata keyakinan para
penilai, bukan keyakinan pemutus; keputusan pemutus dicatat di `decided_by`.

Opsi:

```bash
uv run python scripts/08_agree.py --min-overlap 20   # pasangan kecil tidak diberi kappa
uv run python scripts/08_agree.py --reset-decisions
```

Pasangan dengan kurang dari 10 ujaran bersama ditandai `butir kurang`: angka
kappa dari 5 ujaran terlalu liar untuk dilaporkan.

---

## 8. Seleksi, potong, split, dan laporan

Setelah adjudikasi selesai, empat langkah terakhir tidak memerlukan media di
laptop koordinator.

```bash
uv run python scripts/09_select.py    # 600 per kelas, terpusat
uv run python scripts/10_cut.py       # potong klip di laptop pemilik film
uv run python scripts/11_split.py     # split A/B/C, terpusat
uv run python scripts/12_audit.py --predictions <prediksi.csv>
uv run python scripts/13_stats.py     # dataset card
```

### 8.1 `09_select.py` — seleksi berkuota

```bash
uv run python scripts/09_select.py
```

Membaca `labels_final.csv` + `pool.csv` dan memilih:

- hanya label kelas (bukan `ambigu` / `tidak_layak`);
- `confidence ≥ selection.min_label_conf` (3);
- maksimal 15% tiap kelas dari satu film — **90 ujaran per film per kelas**;
- bergiliran antar film, di dalam film keyakinan tertinggi lalu klip lebih
  panjang, supaya sebarannya merata dan hasilnya deterministik.

Konsekuensi batas 15%: 600 per kelas baru tercapai dengan sekitar tujuh film
per kelas. Dengan tiga film, plafonnya 270 per kelas, dan skrip menuliskan
kekurangannya per kelas — bukan menurunkan batas diam-diam.

Keluaran: `data/selected.csv`, `data/selection_report.md`,
`data/selection_meta.json`.

Opsi untuk eksperimen kecil: `--per-class 20`, `--max-share 1.0`,
`--min-conf 1`.

### 8.2 `10_cut.py` — potong media

Dijalankan **di laptop pemilik film**, sekali untuk semua klip yang ada di
sana.

```bash
uv run python scripts/10_cut.py
uv run python scripts/10_cut.py --limit 3          # uji cepat
uv run python scripts/10_cut.py --no-faces         # tanpa video wajah
```

Tiga keluaran per klip:

| Berkas | Isi | Ukuran rata-rata |
|---|---|---|
| `media/clips/<clip_id>.mp4` | 720p + audio | ~0,4–0,8 MB |
| `media/audio/<clip_id>.wav` | 16 kHz mono | ~100 KB |
| `media/faces/<clip_id>.mp4` | wajah dominan 224×224 @ 5 fps | ~40 KB |

Wajah diambil dari frame yang sama dengan tahap 04, memakai detektor dan
`face.frame_width` yang sama, lalu dipotong persegi dengan margin 1,35 dan
diredam antar frame supaya tidak goyang. `--no-faces` melewati bagian ini.

`data/clips.csv` mencatat path relatif, ukuran, dan status tiap klip
(`ok`, `ada`, `film_tidak_ada`, `gagal`). Repo pelatihan membaca katalog ini,
bukan folder media langsung.

### 8.3 `11_split.py` — tiga varian split

```bash
uv run python scripts/11_split.py
```

| Varian | Aturan | Gunanya |
|---|---|---|
| A | acak per ujaran, stratifikasi kelas | angka optimistis |
| B | per film | menutup kebocoran film |
| C | per aktor | menutup kebocoran aktor |

Keluaran `data/splits.csv` (kolom `split_A`, `split_B`, `split_C`) dan
`data/split_report.md` berisi porsi tiap split, sebaran kelas, serta jumlah
film/aktor yang bocor. Varian A diulang pada `split.n_seeds` seed dan
rentangnya dilaporkan.

Untuk varian C, isi `actor_label` di `data/actors.csv` lebih dulu. Kalau belum,
skrip memakai klaster per film (`ABT13-A01`), sehingga aktor yang sama di dua
film masih bisa bocor; laporan menyebutkan itu.

### 8.4 `12_audit.py` — label vs prediksi model

Skrip ini **tidak** melatih model. Repo pelatihan menghasilkan CSV
`utterance_id, predicted, score`, lalu:

```bash
uv run python scripts/12_audit.py --predictions artifacts/pred_test.csv
```

Keluaran `data/disagreement.csv` (ujaran yang beda, urut skor model),
`data/audit_report.md` (akurasi, matriks konfusi, rekap per kelas), dan
`data/audit_meta.json`. Model yang salah tidak otomatis berarti labelnya salah;
daftar ini untuk diperiksa manusia.

### 8.5 `13_stats.py` — dataset card

```bash
uv run python scripts/13_stats.py
```

Menulis `data/dataset_card.md` dan `data/stats.json`: komposisi kelas, rasio
ketidakseimbangan, kontribusi film, ukuran media, sebaran split A/B/C,
kualitas label, dan keterbatasan. Angkanya diambil dari tabel, bukan dihitung
ulang dari media, jadi bisa dijalankan di laptop mana pun.

Kalau `data/clips.csv` belum mencakup semua klip terpilih, kartu memberi
peringatan bahwa media belum lengkap.

---

## 9. Serah terima ke koordinator

Satu perintah, jangan salin berkas manual:

```bash
uv run python scripts/05_merge.py --pack <namamu>
```

Hasilnya `kiriman/<nama>-<tanggal>.zip`. Kirim zip itu; tidak ada berkas lain
yang perlu disalin. Yang ada di dalamnya:

```
data/films.csv
data/utterances.csv
data/faces.csv
data/actors.csv
data/scenes.csv
data/faces/<film_id>.csv
data/faces/<film_id>.npy
data/*/*_meta.json
handover.json          pemilik, waktu, hash config + uv.lock
```

Sebelum membungkus:

```bash
uv run python -m emosi.bootstrap --verify
uv run pytest -q
```

Hash `config` harus sama dengan koordinator. Kalau berbeda, jangan dibungkus
dulu — samakan config, lalu ulangi tahap yang terpengaruh (biasanya 03 dan 04).

Jangan kirim `media/`, `clips/`, atau `audio/`; ukurannya besar dan bisa dibuat
ulang di laptop pemilik film. Folder `kiriman/` sudah masuk `.gitignore`.

Koordinator tidak menempel `*.csv` mentah. Ia menjalankan
`05_merge.py --from kiriman/`, yang memeriksa hash config, kelengkapan
embedding, dan film ganda sebelum menulis `data/merged/`.

Label **tidak** ikut zip di atas. `07_annotate.py` menulis
`data/labels_<nama>.csv` setiap kali satu label diklik, dan berkas itulah yang
diserahkan setelah sesi selesai — terpisah dari kiriman data. Jangan
menggabungkannya ke `labels_kesan.csv` milik orang lain; `08_agree.py`
menggabungkan berdasarkan kolom `annotator`.

---

## 10. Troubleshooting

| Gejala | Sebab | Tindakan |
|---|---|---|
| `uv sync --locked` gagal "lock is outdated" | `pyproject.toml` berubah tanpa `uv lock` | `uv lock`, commit ulang, minta semua orang sync |
| `ffmpeg tidak ditemukan` | belum terpasang / bukan di PATH | pasang, buka shell baru |
| `01_asr.py` sangat lambat | GPU tidak terpakai | cek `torch.cuda.is_available()`; lihat `docs/environment.md` |
| `provider onnxruntime: CPUExecutionProvider` | pustaka CUDA tidak lengkap | `uv sync --locked --extra face`; cek `nvidia-smi` |
| `CascadeClassifier` kosong / `cv2` error | OpenCV 5 terpasang | `uv sync --locked --extra face` (pin `<5` sudah ada); jangan pasang opencv sendiri |
| `insightface` gagal dikompilasi (Windows) | butuh compiler C++ | pasang Build Tools (§2.2); kalau tetap gagal, WSL2 |
| `faces.csv` kosong | film gelap / wajah jarang / frame gagal didekode | cek `faces_meta.json`, jalankan `--scenes 2` untuk uji |
| `srt_check.csv` `tidak_selaras` | subtitle bukan untuk film itu | cari SRT yang benar sebelum lanjut |
| `film_id` otomatis (`FLM…`) | registri tidak cocok | perbaiki `films_registry.csv`, jalankan `00_inventory.py --force` |
| Dua laptop punya `F01` | id tidak unik global | pakai pola tiga huruf + tahun, mis. `ABT13` |
| Hasil film hilang setelah rerun | menjalankan tanpa `--film` dan tanpa `--force`? | hasil lama memang dipertahankan; cek pesan "dilewati" |
| `commit dibatalkan: ada berkas di atas 5 MB` | hook `pre-commit` bekerja | keluarkan dengan `git rm --cached <berkas>`; jangan `git add -f` |
| `kiriman tidak memuat data/utterances.csv` | zip dibuat manual atau folder yang ditunjuk salah | jalankan `05_merge.py --pack <nama>` |
| `config.yaml berbeda (…… vs ……)` | config diubah di satu laptop | samakan config, ulangi 03 dan 04, bungkus ulang |
| `ABT13 datang dari budi, kesan` | satu film diproses dua orang | pastikan yang benar, lalu `--keep <nama>` |
| `nama kontributor dipakai dua kali` | dua zip dengan nama sama | ganti nama berkas atau foldernya |
| `No module named 'streamlit'` | extra `label` belum dipasang | `uv sync --locked --extra label` |
| `assignments.csv belum ada` | `--assign` belum dijalankan | koordinator menjalankan `07_annotate.py --assign` |
| `Preview … belum ada` | adegan belum dipotong di laptop ini | `07_annotate.py --prepare`; kalau film orang lain, minta zip `--zip` dari pemiliknya |
| `KeyError: ['title'] not in index` | `pool.csv` dibuat sebelum versi 06 terbaru | jalankan ulang `06_qc.py` |
| `Kappa` bertuliskan `butir kurang` | irisan dua penilai < 10 ujaran | tambah gold set atau cek tukar silang tahap 07 |
| `keputusan … bukan label sah` | typo di kolom `decision` | isi kelas, `ambigu`, atau `tidak_layak` |
| `labels_final.csv` kosong | semua ujaran masih `perlu_adjudikasi` | isi `adjudication.csv`, jalankan ulang |
| Seleksi berhenti di 270/kelas | batas 15% per film, film masih tiga | tambah film; jangan turunkan batas |
| `10_cut.py` melaporkan `film_tidak_ada` | film itu bukan milik laptop ini | kirim `selected.csv` ke pemiliknya |
| Split C masih bocor | `actor_label` kosong | isi `data/actors.csv`, jalankan ulang |
| `No module named 'cv2'` setelah memasang streamlit | `uv sync --extra label` mencabut extra lain | sync dengan `--extra asr --extra face --extra label` |
| `OrtCompileApiFlags` / `onnxruntime` gagal diimpor | paket CPU `onnxruntime` menimpa `onnxruntime-gpu` | lihat bagian 10.1 |

---

### 10.1 `onnxruntime` CPU menimpa GPU

`insightface` bergantung pada `onnxruntime` (CPU), sedangkan extra `face`
memasang `onnxruntime-gpu`. Kalau `uv sync` memasang keduanya, berkas Python
versi CPU bisa menimpa berkas GPU dan impor gagal dengan
`cannot import name 'OrtCompileApiFlags'`.

Gejalanya juga terlihat dari `uv pip list`: ada `onnxruntime` **dan**
`onnxruntime-gpu` dengan versi berbeda.

Perbaikan di laptop itu:

```bash
uv pip uninstall onnxruntime onnxruntime-gpu
rm -rf .venv/lib/python3.11/site-packages/onnxruntime*
uv pip install --no-deps onnxruntime-gpu==1.22.0
uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Hasil yang benar memuat `CUDAExecutionProvider`. Periksa lagi setelah
`uv sync` berikutnya; selama `onnxruntime` CPU ikut terpasang, masalah ini
bisa muncul kembali.

## 11. Checklist per film

- [ ] film terdaftar di `films_registry.csv` dengan `path_contains` yang benar
- [ ] subtitle Inggris ada satu folder dengan film
- [ ] `00_inventory.py` → `pass_audition = True`
- [ ] `01_asr.py` → status `selaras` di `srt_check.csv`
- [ ] `02_scenes.py` → adegan 90–240 detik, tidak ada yang tumpang-tindih
- [ ] `03_utterances.py` → mayoritas ujaran `timing = asr`
- [ ] `04_face_screen.py` → `face_ok` cukup, `provider` GPU
- [ ] `actors.csv` → `actor_label` diisi untuk aktor utama
- [ ] `05_merge.py --pack <nama>` → zip kiriman jadi
- [ ] `07_annotate.py --prepare` → preview adegan tersedia
- [ ] `bootstrap --verify` bersih
- [ ] `config.yaml` hash sama dengan koordinator

## 12. Checklist sebelum menyerahkan data

- [ ] semua film yang kamu pegang selesai tahap 00–04
- [ ] `environment.lock.json` ikut
- [ ] tidak ada `config.yaml` yang diubah sendiri
- [ ] `actors.csv` sudah diisi nama aktor
- [ ] 150 ujaran tukar silang sudah dilabeli
- [ ] `labels_<nama>.csv` diserahkan ke koordinator (label tidak ikut zip)
- [ ] `08_agree.py` dijalankan; tidak ada `perlu_adjudikasi` tersisa
- [ ] `09_select.py` dijalankan; kekurangan kuota dijelaskan di laporan
- [ ] `10_cut.py` selesai; `clips.csv` tanpa status `gagal`
- [ ] `11_split.py` dijalankan; kebocoran film/aktor sesuai varian
- [ ] `12_audit.py` dijalankan setelah repo model menghasilkan prediksi
- [ ] `13_stats.py` dijalankan; `dataset_card.md` diperiksa
- [ ] kiriman dibungkus dengan `--pack`, bukan salinan manual
- [ ] yang dikirim sesuai daftar bagian 9
