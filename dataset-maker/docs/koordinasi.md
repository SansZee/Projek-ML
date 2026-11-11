# Koordinasi Tim

Empat orang, masing-masing memegang sebagian film, digabung di akhir.

## Pembagian kerja

| Tahap | Siapa | Alasan |
|---|---|---|
| `00`–`04` | per orang, di laptop masing-masing | butuh berkas film yang hanya ada di orang itu |
| `05` merge + `06` QC | koordinator, sekali | katalog gabungan dan rencana adegan harus satu versi |
| `07` pelabelan | per orang + **150 ujaran tukar silang** | tanpa overlap, kappa tidak bisa dihitung |
| `08` kappa + adjudikasi | terpusat | kappa lintas anotator harus satu hitungan |
| `09` seleksi 600/kelas | terpusat, sekali saja | kalau per orang, semua orang menyerobot kelas yang sama |
| `10` potong klip | per orang, setelah daftar seleksi dibagi | berkas film tetap di laptop masing-masing |
| `11` split A/B/C | terpusat | butuh `actors.csv` dari semua film |
| `12` audit + `13` statistik | terpusat | butuh seluruh 3000 klip terkumpul |

## Aturan `film_id`

Harus unik secara global. Daftarkan **sebelum** memproses film:

```csv
film_id,title,year,genre,lead_actors,path_contains
ABT13,About Time,2013,romansa,Domhnall Gleeson;Rachel McAdams,About Time
```

Pola yang dipakai: tiga huruf awal + dua digit tahun (`ABT13`). Jangan pakai
`F01`, `F02` — dua laptop akan sama-sama punya `F01`.

`path_contains` adalah potongan path yang cukup untuk mengenali film itu di
laptop siapa pun. Kalau tidak cocok, `00_inventory.py` membuat id otomatis dan
**itu tanda bahaya**: perbaiki registri, jangan pakai id otomatis.

## Tukar silang 150 ujaran

Tanpa ini, bagian "uji kesepakatan antar-anotator" di laporan kosong.

1. Koordinator menjalankan `07_annotate.py --assign --people <empat nama>`.
   Skrip memilih dua adegan gold dan satu set 150 ujaran per pemilik film.
2. Tiap pemilik film menjalankan `--prepare` untuk preview adegan sendiri, lalu
   `--prepare --zip <nama>` untuk adegan yang dibutuhkan orang lain.
3. Anotator melabeli gold set lebih dulu, lalu 150 ujaran silang.
4. Serahkan `labels_<nama>.csv` ke koordinator. Jangan menyalin label orang
   lain ke berkas sendiri.

Lakukan di minggu yang sama, jangan ditunda ke akhir.

## Berkas yang diserahkan per orang

```
kiriman/<nama>-<tanggal>.zip   hasil 05_merge.py --pack
```

Isi zip itu sudah mencakup `films.csv`, `utterances.csv`, `faces.csv`,
`faces/<film_id>.npy`, `actors.csv`, `scenes.csv`, dan `handover.json`. Tidak
ada berkas lain yang perlu disalin manual. `actors.csv` di dalamnya harus sudah
diisi `actor_label`.

Label tidak ikut zip: `labels_<nama>.csv` diserahkan terpisah setelah sesi
pelabelan, karena isinya baru ada di tahap `07`.

`clips/`, `audio/`, `faces/` **tidak** dikirim. Yang dikirim daftar `clip_id`
hasil seleksi terpusat; potongan klip dibuat di laptop masing-masing lalu
dikumpulkan sesuai kebutuhan eksperimen.

## Checklist sebelum menyerahkan

- [ ] `uv run python -m emosi.bootstrap --verify` bersih
- [ ] `config.yaml` tidak diubah (bandingkan hash dengan koordinator)
- [ ] kiriman dibungkus `05_merge.py --pack <nama>`, bukan salinan manual
- [ ] `film_id` terdaftar di registri
- [ ] `labels_<nama>.csv` diserahkan, `labels_final.csv` dikunci koordinator
- [ ] 150 ujaran tukar silang sudah dilabeli
- [ ] `actors.csv` untuk film yang dipegang, `actor_label` sudah diisi

## Kesalahan yang paling sering terjadi

1. **Config berbeda** → dataset tidak bisa digabung. Cek hash.
2. **`film_id` bertabrakan** → baris dari empat laptop tercampur tanpa jejak.
3. **Seleksi dilakukan per orang** → kuota kelas habis di film yang sama.
4. **Label berubah setelah eksperimen dimulai** → semua hasil harus diulang.
5. **Tukar silang ditunda** → kappa tidak ada, bab evaluasi kehilangan satu bagian.
