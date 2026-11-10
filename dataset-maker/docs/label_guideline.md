# Pedoman Anotasi

Versi 1.0 — dibekukan sebelum pelabelan dimulai. Perubahan setelah pelabelan
berjalan berarti label lama harus diperiksa ulang.

## Tugas anotator

Tonton **adegan utuh 2–3 menit** sambil memberi label pada setiap ujaran saat
muncul. Jangan menilai klip terpisah, kecuali untuk memeriksa ulang.

Tombol di aplikasi:

| Tombol | Arti |
|---|---|
| `1` | senang |
| `2` | sedih |
| `3` | marah |
| `4` | takut |
| `5` | netral |
| `6` | ambigu — sinyal saling bertentangan, tidak ada yang dominan |
| `0` | tidak layak — wajah tidak terlihat, audio rusak, bukan dialog |

Setelah memilih, beri keyakinan **1–5**:
1 = menebak, 3 = cukup yakin, 5 = sangat yakin.

## Definisi operasional

| Kelas | Definisi | Contoh | Kasus batas |
|---|---|---|---|
| **Senang** | Ekspresi positif jelas: tawa, senyum terbuka, nada ceria | Kabar baik, reuni, lelucon yang berhasil | Antusiasme datar → netral |
| **Sedih** | Kekecewaan atau kehilangan; tempo melambat, air mata, bahu turun | Perpisahan, kabar duka, penyesalan | Marah yang menahan tangis → sedih bila dominan |
| **Marah** | Konfrontasi atau kemarahan; volume naik, alis turun, rahang menegang | Pertengkaran, ancaman, tuduhan | Kesal ringan tanpa konfrontasi → netral |
| **Takut** | Ancaman atau kecemasan; mata membelalak, tubuh menegang, bisikan | Adegan horor, ketegangan sebelum kejadian | Kaget sesaat → takut bila bertahan lebih dari satu ujaran |
| **Netral** | Tidak ada muatan emosi menonjol; informasi faktual | Dialog transaksional, penjelasan, arahan | **Bukan tempat sampah.** Bila ragu, pakai ambigu |

## Aturan umum

1. **Emosi penutur, bukan emosi kata.** "Bagus sekali" bisa pujian atau
   sarkasme. Nilai maksudnya, bukan kata-katanya.
2. **Sarkasme** mengikuti maksud. Bila maksudnya menyakiti → marah; bila
   bercanda → senang.
3. **Emosi campuran** → label dominan, isi kolom alternatif bila ada.
4. **Kontekstual, bukan potongan.** Adegan sebelum ujaran menjelaskan kenapa
   seseorang bicara begitu. Manfaatkan itu.
5. **Ragu antara dua kelas** → pilih yang lebih kuat, turunkan keyakinan ke 2.
6. **Ragu tanpa arah** → `6` (ambigu). Klip ambigu diperiksa saat adjudikasi,
   tidak dibuang otomatis.
7. **Jangan menebak dari teks saja.** Bila wajah tidak terlihat dan suara datar,
   itu `0` (tidak layak), bukan netral.

## Penanda wicara

Buang dari pertimbangan: `[MUSIC]`, `(laughs)`, lirik, teks layar, kredit.
Bila satu cue memuat dua pembicara dengan emosi berbeda, nilai pembicara yang
**paling menonjol** dan catat di kolom `notes`.

## Yang tidak boleh dilakukan

- Melihat saran model pada pass pertama (anchoring).
- Menyeimbangkan kelas saat melabeli. Penyeimbangan dilakukan sekali setelah
  semua label terkunci.
- Mengubah definisi kelas di tengah jalan.
- Melabeli klip yang wajahnya tidak terlihat hanya agar jumlahnya cukup.

## Uji kesepakatan

- 500 ujaran dilabeli dua anotator berbeda, diambil dari adegan yang berbeda.
- Target Cohen's kappa > 0.6. Nilai 0.4–0.6 dapat diterima dengan catatan.
- Kappa di bawah 0.4 → masalahnya pedoman, bukan film. Revisi definisi, lalu
  uji ulang pada 50 klip yang sama.
