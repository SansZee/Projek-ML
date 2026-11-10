# Koordinasi Tim

Empat orang, masing-masing memegang sebagian film, digabung di akhir.

## Pembagian kerja

| Tahap | Siapa | Alasan |
|---|---|---|
| `00` inventory → `05` QC | per orang, di laptop masing-masing | butuh berkas film yang hanya ada di orang itu |
| `06` pelabelan | per orang + **150 ujaran tukar silang** | tanpa overlap, kappa tidak bisa dihitung |
| `07` kappa + adjudikasi | terpusat | kappa lintas anotator harus satu hitungan |
| `08` seleksi 600/kelas | terpusat, sekali saja | kalau per orang, semua orang menyerobot kelas yang sama |
| `09` potong klip | per orang, setelah daftar seleksi dibagi | berkas film tetap di laptop masing-masing |
| `10` split A/B/C | terpusat | butuh `actors.csv` dari semua film |
| `11` audit + `12` statistik | terpusat | butuh seluruh 3000 klip terkumpul |

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

1. Setelah `pool.csv` siap, ekspor 150 ujaran acak **beserta video adegannya**.
2. Kirim ke satu anggota lain, terima 150 dari anggota ketiga.
3. Labeli, lalu serahkan `labels_cross.csv` ke koordinator.
4. Lakukan di minggu yang sama, jangan ditunda ke akhir.

## Berkas yang diserahkan per orang

```
films.csv            hasil 00_inventory
asr/*.json           opsional, besar — cukup simpan di laptop masing-masing
utterances.csv
faces.csv            hasil 04_face_screen
faces/<film_id>.npy  embedding wajah, sejajar baris faces/<film_id>.csv
actors.csv           daftar aktor per film (input split C), actor_label diisi
pool.csv
labels_final.csv
environment.lock.json
```

`clips/`, `audio/`, `faces/` **tidak** dikirim. Yang dikirim daftar `clip_id`
hasil seleksi terpusat; potongan klip dibuat di laptop masing-masing lalu
dikumpulkan sesuai kebutuhan eksperimen.

## Checklist sebelum menyerahkan

- [ ] `uv run python -m emosi.bootstrap --verify` bersih
- [ ] `config.yaml` tidak diubah (bandingkan hash dengan koordinator)
- [ ] `film_id` terdaftar di registri
- [ ] `labels_final.csv` sudah dikunci (checksum dicatat)
- [ ] 150 ujaran tukar silang sudah dilabeli
- [ ] `actors.csv` untuk film yang dipegang, `actor_label` sudah diisi

## Kesalahan yang paling sering terjadi

1. **Config berbeda** → dataset tidak bisa digabung. Cek hash.
2. **`film_id` bertabrakan** → baris dari empat laptop tercampur tanpa jejak.
3. **Seleksi dilakukan per orang** → kuota kelas habis di film yang sama.
4. **Label berubah setelah eksperimen dimulai** → semua hasil harus diulang.
5. **Tukar silang ditunda** → kappa tidak ada, bab evaluasi kehilangan satu bagian.
