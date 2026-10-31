# Data Catalog

Katalog sumber data proyek. Setiap entri memuat sumber, versi, lisensi, resolusi, tanggal unduh, dan keterbatasan. Entri dilengkapi seiring fitur F-2.x selesai.

## 1. Batas administrasi — OCHA COD-AB Indonesia

| Item | Nilai |
|---|---|
| Sumber | OCHA Field Information Services Section, data dasar Badan Pusat Statistik (BPS) |
| Dataset | HDX `cod-ab-idn`, "Indonesia - Subnational Administrative Boundaries" |
| Versi | `v01`, `valid_on` 2020-04-01; dataset dimutakhirkan di HDX 2026-06-23 |
| Lisensi | Creative Commons Attribution for Intergovernmental Organisations (CC BY-IGO) |
| Format mentah | FileGDB terkompresi (`idn_admin_boundaries.gdb.zip`, 218 MB), layer `idn_admin2` |
| CRS | EPSG:4326 |
| Tanggal unduh | 2026-10-31 (lihat `data/raw/admin/*.meta.json`, sha256 tercatat) |
| Output | `data/processed/admin/adm2.parquet` (10 kabupaten/kota), `adm1.parquet` (provinsi, hasil dissolve) |

Kolom terstandar: `adm2_name`, `adm2_pcode` (kode BPS berawalan `ID64`), `adm1_name`, `geometry`.

Keterbatasan: sumber mengeja Mahakam Ulu sebagai "Mahakam Hulu" (pcode `ID6411`); nama dibiarkan apa adanya. Batas laut/pulau kecil mengikuti BPS 2020, belum tentu sama dengan peta BIG terbaru.

## 2. Hotspot — NASA FIRMS VIIRS S-NPP 375 m (standard processing)

Ringkasan hasil unduhan (bbox 113,7359–119,0891 BT, −2,5052–2,7263 LS/LU): **366.015 hotspot** semua tingkat confidence, 2012-01-24 s.d. 2026-07-27 (tanggal lokal). Tahun puncak 2015 (118.033) dan 2019 (53.302); terendah 2022 (3.867). Rincian per tahun dan bulan di `reports/firms/`.

| Item | Nilai |
|---|---|
| Sumber | NASA FIRMS, produk VNP14IMGT (VIIRS I-band 375 m Active Fire, standard quality) |
| Akses | FIRMS area API: `/api/area/csv/{MAP_KEY}/VIIRS_SNPP_SP/{bbox}/{1..5}/{YYYY-MM-DD}`; maks 5 hari per permintaan, kuota 5.000 transaksi per 10 menit per MAP_KEY |
| Cakupan temporal | 20 Januari 2012 – sekarang (data standar tersedia dengan jeda sekitar 2 bulan setelah akuisisi) |
| Cakupan spasial | bbox Kaltim + buffer `region.bbox_buffer_deg` (0,1°) |
| Lisensi | Data NASA, domain publik; FIRMS meminta sitasi "NASA FIRMS" |
| Format mentah | CSV per potongan ≤5 hari di `data/raw/firms/VIIRS_SNPP_SP/<tahun>/`, manifest `manifest.jsonl` (tanpa MAP_KEY) |
| Tanggal unduh | 2026-11-01 (1.060 potongan 5 hari, 2012-01-23 s.d. 2026-07-27; endpoint `data_availability` melaporkan data standar tersedia sampai 2026-07-31, potongan yang belum tersedia penuh tidak di-cache). Manifest: `data/raw/firms/VIIRS_SNPP_SP/manifest.jsonl` |
| Output | `data/processed/firms/hotspots_viirs_snpp_sp.parquet`; ringkasan `reports/firms/hotspots_per_year.csv`, `hotspots_per_month.csv`; gambar `reports/figures/firms_summary.png` |

### Kolom mentah (dari dokumentasi resmi FIRMS/MODAPS)

| Kolom | Arti |
|---|---|
| `latitude`, `longitude` | Pusat piksel kebakaran nominal 375 m |
| `bright_ti4` | Suhu kecerahan kanal I-4 (Kelvin) |
| `bright_ti5` | Suhu kecerahan kanal I-5 (Kelvin) |
| `frp` | Fire radiative power terintegrasi piksel (MW) |
| `scan`, `track` | Ukuran piksel aktual sepanjang scan dan track (km) |
| `acq_date`, `acq_time` | Tanggal dan jam akuisisi satelit, UTC, `acq_time` dalam format HHMM |
| `satellite` | `N` = Suomi NPP |
| `instrument` | `VIIRS` |
| `confidence` | `l` low, `n` nominal, `h` high (huruf dikonfirmasi dari respons API nyata, 2026-11-01). Low = piksel siang dengan anomali I4 < 15 K, biasanya sun glint; nominal = bebas sun glint, anomali > 15 K; high = piksel jenuh siang/malam |
| `version` | Koleksi dan sumber pemrosesan. Respons API `VIIRS_SNPP_SP` saat ini memberi `2` (Collection 2 standar); dokumentasi lama menyebut `1.0` / `1.0NRT` |
| `daynight` | `D` siang, `N` malam |
| `type` | Jenis hotspot terinferensi, **hanya ada di data standard processing** (dikonfirmasi ada di respons `VIIRS_SNPP_SP`): `0` presumed vegetation fire, `1` active volcano, `2` other static land source, `3` offshore (semua deteksi di atas air) |

### Kolom turunan (ditambahkan `karhutla.ingest.firms.clean_hotspots`)

| Kolom | Arti |
|---|---|
| `acq_datetime_utc` | Gabungan `acq_date` + `acq_time` sebagai timestamp UTC |
| `acq_date_local` | Tanggal lokal WITA (Asia/Makassar, UTC+8). Overpass malam S-NPP (sekitar 17:00–18:30 UTC) jatuh di dini hari tanggal berikutnya waktu lokal |
| `week_start` | Senin minggu ISO dari `acq_date_local`; kunci waktu seluruh proyek |
| `year`, `month` | Dari tanggal lokal |

Pembersihan: baris di luar bbox dibuang, duplikat pada (`latitude`, `longitude`, `acq_date`, `acq_time`, `satellite`) dibuang. **Semua tingkat `confidence` dipertahankan** di Parquet; penyaringan ke `n` + `h` dilakukan saat membangun label (Fase 4) sesuai `labels.confidence_keep`.

Keterbatasan: hotspot adalah anomali termal, bukan kebakaran terkonfirmasi. Deteksi bergantung pada tutupan awan dan sudut pandang. Data NRT tidak dipakai untuk arsip agar atribut `type` dan geolokasi berkualitas standar; untuk minggu terbaru (F-2.2) sumber NRT dipakai dan tidak memiliki `type`. Sensor tunggal S-NPP dipertahankan agar laju deteksi konsisten antar-tahun; S-NPP dijadwalkan berhenti dikirim 2 November 2026 (catatan NASA), sehingga simulasi 2026 memakai data yang tersedia sampai saat itu.
