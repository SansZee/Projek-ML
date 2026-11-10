# Environment

Tujuan: empat laptop menghasilkan `films.csv`, `utterances.csv`, dan label yang
bisa digabung tanpa penjelasan tambahan.

## 1. Pasang uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

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

Cek:

```bash
ffmpeg -version | head -1
```

## 3. Sinkronkan projek

```bash
git clone <repo> dataset-maker && cd dataset-maker
uv sync --locked
```

`--locked` penting. Tanpa itu uv menyelesaikan ulang dependensi dan menulis
ulang `uv.lock`, sehingga empat orang bisa punya empat environment berbeda
tanpa menyadarinya.

Untuk tahap ASR dan pelabelan:

```bash
uv sync --locked --extra asr --extra label
```

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

## 5. Windows: insightface

Tahap `04_face_screen.py` membutuhkan `insightface` saat kompilasi. Di Windows
ini sumber kegagalan paling sering.

Urutan pilihan:

1. **WSL2** (paling dekat dengan setup Linux, disarankan).
2. Python 3.11 dari python.org + Visual C++ Build Tools, lalu
   `uv sync --extra face`.
3. Docker (lihat bawah).

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
| Frame hitam saat deteksi | HEVC 10-bit | memang dihindari: semua frame lewat pipe ffmpeg, bukan `cv2.VideoCapture` |
