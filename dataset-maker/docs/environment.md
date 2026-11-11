# Environment

Tujuan: empat laptop menghasilkan `films.csv`, `utterances.csv`, dan label yang
bisa digabung tanpa penjelasan tambahan.

## 1. Pasang uv

Linux, macOS, WSL2:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
winget install --id=astral-sh.uv -e
# atau installer resmi:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Tutup dan buka ulang PowerShell setelah instalasi supaya `uv` masuk `PATH`.

## 2. Pasang ffmpeg

Wajib ada `ffmpeg` **dan** `ffprobe` di `PATH`. Versi minimum 6.0.

```bash
# Arch
sudo pacman -S ffmpeg
# Ubuntu/Debian
sudo apt install ffmpeg
# macOS
brew install ffmpeg
```

Windows (PowerShell):

```powershell
winget install --id=Gyan.FFmpeg -e
```

Cek **keduanya**; sebagian paket ffmpeg hanya membawa `ffmpeg` tanpa
`ffprobe`:

```powershell
ffmpeg -version
ffprobe -version
```

Kalau perintah masih "tidak dikenal", tutup dan buka ulang PowerShell — folder
`bin` baru masuk `PATH` setelah shell dijalankan ulang.

## 3. Sinkronkan projek

```bash
git clone <repo> dataset-maker && cd dataset-maker
uv sync --locked
```

`--locked` penting. Tanpa itu uv menyelesaikan ulang dependensi dan menulis
ulang `uv.lock`, sehingga empat orang bisa punya empat environment berbeda
tanpa menyadarinya.

Untuk tahap data, wajah, dan pelabelan:

```bash
uv sync --locked --extra asr --extra face --extra label
```

Sebut semua extra yang dipakai: `uv` hanya mempertahankan extra yang disebut.
Di Windows, `--extra face` mengompilasi `insightface`; lihat bagian 5.

## 4. CUDA dan PyTorch

`torch` sengaja diambil dari indeks `pytorch-cu126` (lihat `pyproject.toml`,
bagian `[[tool.uv.index]]`). Kalau driver GPU-mu lebih lama dari CUDA 12.6:

```bash
nvidia-smi | grep "CUDA Version"
```

Kalau tertulis 12.4, ubah blok indeks di `pyproject.toml` menjadi `cu124`, lalu:

```bash
uv lock && uv sync --locked --extra ml
```

**Gejala salah indeks:** training berjalan tapi sangat lambat, dan
`python -c "import torch; print(torch.cuda.is_available())"` menghasilkan
`False`. Itu berarti yang terpasang wheel CPU.

## 5. Windows: insightface dan GPU

Tahap `04_face_screen.py` membutuhkan `insightface`, yang tidak punya wheel
Windows dan harus dikompilasi. Ini sumber kegagalan paling sering.

**Native Windows.** Sekali saja per laptop:

```powershell
winget install --id=Microsoft.VisualStudio.2022.BuildTools -e
```

Di installer yang terbuka, centang **Desktop development with C++**, lalu:

```powershell
uv sync --locked --extra asr --extra face --extra label
```

**Kalau kompilasi tetap gagal**, pindah ke WSL2 (Ubuntu 24.04). Di sana semua
perintah sama dengan Linux, media diakses lewat `/mnt/d/Film`, dan GPU memakai
driver NVIDIA untuk WSL. Docker (bagian 6) adalah pilihan terakhir.

**GPU di Windows.** Paket `nvidia-*-cu12` di `pyproject.toml` hanya dipasang di
Linux (`sys_platform == 'linux'`). Di Windows, `onnxruntime-gpu` mencari CUDA
dan cuDNN yang dipasang sendiri:

1. driver NVIDIA terbaru;
2. CUDA Toolkit **12.6** dan cuDNN **9.1** untuk Windows;
3. folder `bin` keduanya masuk `PATH`;
4. cek:

```powershell
uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Hasil yang benar memuat `CUDAExecutionProvider`. Tanpa itu, tahap 04 tetap
jalan di CPU: 1–2 jam per film, bukan 5–10 menit.

**Line ending.** Git for Windows default memakai CRLF, sedangkan hook
`.githooks/pre-commit` adalah skrip bash. Setel di folder projek:

```powershell
git config core.autocrlf false
git config core.longpaths true
```

## 6. Docker (opsional, untuk tim dengan OS campur)

Docker mengunci CUDA, ffmpeg, dan Python sekaligus — satu-satunya cara
menjamin environment identik lintas OS.

```dockerfile
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project
COPY . .
RUN uv sync --locked
```

```bash
docker build -t emosi:cu126 .
docker run --gpus all -v "$PWD":/app -v /path/ke/film:/film emosi:cu126 \
  uv run python scripts/00_inventory.py --raw-dir /film
```

## 7. Verifikasi

```bash
uv run python -m emosi.bootstrap --verify
```

Keluaran yang diharapkan: ffmpeg ada, `uv.lock` ada (bukan "TIDAK ADA"),
`config` menampilkan 16 karakter pertama hash.

Berkas `environment.lock.json` yang dihasilkan **dilampirkan** bersama data yang
kamu serahkan ke koordinator.

## 8. Kalau bermasalah

| Gejala | Sebab | Tindakan |
|---|---|---|
| `uv sync --locked` gagal "lock is outdated" | `pyproject.toml` berubah tanpa `uv lock` | `uv lock`, commit ulang |
| `cuda_available = False` | wheel CPU terpasang | cek indeks `cu126`/`cu124`, hapus `.venv`, `uv sync` ulang |
| `CascadeClassifier` kosong / `cv2` error | OpenCV 5 terpasang (menghapus `CascadeClassifier`) | `uv sync --locked --extra face`; pin `<5` sudah ada di `pyproject.toml` |
| `provider onnxruntime: CPUExecutionProvider` | libcublas/libcudnn/cudart tidak ditemukan | `uv sync --locked --extra face`, cek `nvidia-smi` |
| `ffmpeg tidak ditemukan` | ffmpeg tidak di PATH | pasang, lalu buka ulang shell |
| `winget: command not found` | Windows 10 lama | perbarui App Installer dari Microsoft Store, atau pasang manual |
| `ffmpeg`/`ffprobe` masih tidak dikenal setelah winget | PATH belum dimuat | tutup dan buka ulang PowerShell |
| `bad interpreter: /usr/bin/env bash^M` | berkas di-checkout dengan CRLF | `git config core.autocrlf false`, lalu checkout ulang |
| `Microsoft Visual C++ 14.0 required` saat `--extra face` | Build Tools belum ada | `winget install --id=Microsoft.VisualStudio.2022.BuildTools -e`, centang Desktop development with C++ |
| `CUDAExecutionProvider` tidak muncul di Windows | CUDA/cuDNN tidak di PATH | pasang CUDA 12.6 + cuDNN 9.1, tambahkan `bin` ke `PATH` |
| Frame hitam saat deteksi | HEVC 10-bit | memang dihindari: semua frame lewat pipe ffmpeg, bukan `cv2.VideoCapture` |
