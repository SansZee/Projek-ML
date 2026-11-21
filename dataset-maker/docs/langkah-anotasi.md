# Langkah Anotasi

Satu halaman untuk fase pelabelan. Yang sudah selesai: 24 film digabung, QC
jalan, dan `data/assignments.csv` sudah dibuat. Tidak ada yang perlu di-merge
atau di-QC lagi.

Aturan lengkap tiap kelas ada di `docs/label_guideline.md`. Panduan teknis
lengkap ada di `docs/panduan-pemakaian.md` §7.

## 1. Siapa mengerjakan apa

Empat orang, nama harus persis: `kesan`, `bagus`, `maul`, `michael`.

| Orang | Tugas | Jumlah klik |
|---|---|---|
| kesan | blok utama + gold + silang | 3.020 |
| maul | blok utama + gold + silang | 3.020 |
| bagus | blok utama + gold + silang | 3.002 |
| michael | blok utama + gold + silang | 3.020 |

Blok utama dibagi **per adegan**, jadi bisa berisi film orang lain. Yang
menentukan bukan siapa pemilik film, melainkan siapa yang mendapat adegan itu.
Adegan tidak pernah dipecah antar-orang.

Gold adalah dua adegan yang sama untuk semua orang (38 ujaran), diambil dari
film Michael dan sengaja tidak dimasukkan ke blok utama siapa pun. Jadi keempat
orang menilai ujaran yang persis sama — itu syarat kappa global.

Pembagian ini sudah final. Kalau di tengah jalan diubah, peran berpindah dan
label yang sudah ada tidak cocok lagi.

## 2. Yang dikirim koordinator

Tiga berkas, taruh di `data/`:

```
data/assignments.csv
data/pool.csv
data/pool_scenes.csv
```

Plus preview untuk adegan yang bukan milikmu. Preview dibuat pemilik film
dengan:

```bash
uv run python scripts/07_annotate.py --prepare --zip <namamu>
```

Hasilnya `kiriman/preview-untuk-<namamu>-<tanggal>.zip` (±40–60 MB per
pengiriman). Ekstrak sehingga isinya masuk ke `media/preview/`.

Blok utama dibagi rata, jadi adeganmu bisa berasal dari beberapa film
sekaligus. Totalnya sekitar 165 adegan (±950 MB), dan pemilik tiap film harus
mengirim zip-nya kepadamu. Minta semua sebelum mulai, supaya sesi tidak
terhenti di tengah.

## 3. Persiapan laptop (sekali saja)

```bash
git clone <repo> dataset-maker
cd dataset-maker
uv sync --locked --extra asr --extra face --extra label
uv run python -m emosi.bootstrap --verify
```

Windows: PowerShell, dan `insightface` butuh Build Tools — lihat
`docs/panduan-pemakaian.md` §2.

Sebut **ketiga** extra. `--extra label` saja mencabut `face` dan `asr`, dan
strip wajah tidak bisa dibuat.

## 4. Siapkan preview dan strip wajah

Kalau filmmu ada di laptop ini, potong preview untuk adegan yang dibutuhkan
orang lain lalu kirim:

```bash
uv run python scripts/07_annotate.py --prepare --zip bagus
```

Untuk dirimu sendiri, setelah semua preview terkumpul di `media/preview/`
(termasuk yang dikirim orang lain):

```bash
uv run python scripts/07_annotate.py --prepare
```

Perintah ini juga membuat **strip wajah** di `media/face_preview/`. Film asli
tidak diperlukan — cukup preview. Biayanya ±0,4 detik per ujaran.

## 5. Buka aplikasi dan mulai

```bash
uv run streamlit run scripts/07_annotate.py
```

Browser terbuka di `http://localhost:8501`. Pilih namamu di sidebar, pilih
adegan, lalu nilai ujaran satu per satu.

Cara kerjanya:

- video di kiri, transkrip dan keyakinan di kanan;
- di bawah video ada **strip 3 wajah** — itulah wajah yang akan diterima model;
- pintasan papan ketik: `1`–`7` kelas, `8` ambigu, `0` tidak layak;
- setiap klik langsung tersimpan dan pindah ke ujaran berikutnya;
- sesi boleh ditutup kapan saja, besok lanjut dari yang belum.

Aturan singkat:

| Tombol | Kapan |
|---|---|
| `1`–`7` | emosi penutur terbaca jelas |
| `8` ambigu | sinyal ada tapi bertentangan, tidak ada yang dominan |
| `0` tidak layak | wajah salah orang/tertutup, audio rusak, subtitle tidak cocok |

Strip yang menampilkan **orang berbeda di tiga tile** berarti wajah terbesar
berpindah di tengah ujaran — itu `0`. Wajah orang lain yang cuma numpang di
pojok crop bukan alasan `0`, selama wajah utama adalah penutur.

## 6. Setelah selesai

Kirim satu berkas ke koordinator:

```
data/labels_<namamu>.csv
```

Jangan menggabungkan dengan berkas orang lain. `08_agree.py` mencocokkan
berdasarkan kolom `annotator`.

## 7. Kalau ada masalah

| Gejala | Penyebab | Perbaikan |
|---|---|---|
| `No module named 'streamlit'` | extra `label` belum ada | `uv sync --locked --extra asr --extra face --extra label` |
| `No module named 'cv2'` | sync tanpa `face` | sync ulang dengan ketiga extra |
| `Preview … belum ada` | zip preview belum diekstrak | ekstrak ke `media/preview/` |
| `Strip wajah belum ada` | `--prepare` belum jalan setelah preview lengkap | ulangi langkah 4 |
| `bad interpreter` | checkout CRLF | `git rm --cached -r .` lalu `git reset --hard` |
| nama tidak muncul di sidebar | salah ketik nama | harus `kesan`, `bagus`, `maul`, `michael` |

## 8. Khusus yang memakai Kaggle

Kaggle hanya untuk tahap 00–04, yang sudah selesai. **Melabeli tidak bisa di
Kaggle** karena Streamlit butuh server yang dibuka di browser.

Sebelum sesi Kaggle ditutup, unduh dari `/kaggle/working`:

- `kiriman/<nama>-<tanggal>.zip` (data),
- zip preview: `kiriman/preview-untuk-*.zip` dari `--prepare --zip`.

Setelah sesi berakhir, isi `/kaggle/working` hilang. Di Kaggle, potong preview
dengan `--prepare --no-faces` (tidak perlu insightface); strip wajah dibuat di
laptop pada langkah 4.
