# QC Suite Python

Greenfield QC suite in a separate folder, built as:

- `backend/`: Flask API for auth, templates, deployments, inspection sessions, workstation, and dashboard
- `client_tk/`: Tkinter desktop shell with role-based screens for Operator and Admin
- `shared/`: contracts and enums shared across backend and client
- `scripts/`: run and smoke-test helpers

Default runtime is local-first desktop: the client uses the embedded local transport by default, and split deployment is only needed for compatibility or remote access.

No default users are seeded. On `postgresql`/`sqlserver` backends, accounts live in the factory's own `operator` table (see `deploy/.env.example`); on the `local` backend, create the first user directly in `data/json_store/users.json` or via another already-authenticated admin session.

## Role Screens

- `Operator`: login, local camera, active deployment lookup, ROI update, live decision, DB write status
- `Admin`: templates, deployments, users, inspection results, dashboard, model registry (import zip / upload / export / delete), machine settings. Training is done in other software — this app only imports finished models.

## Cara Menjalankan

README ini memakai contoh Windows + PowerShell.

### 1. Install prasyarat

- Install Python `3.11`
- Pastikan `pip` aktif
- Untuk local-only desktop, tidak perlu database server tambahan
- Jika akan pakai PostgreSQL, siapkan server PostgreSQL yang bisa dijangkau backend
- Jika masih memakai SQL Server sebagai kompatibilitas sementara, install driver ODBC yang sesuai, default repo ini memakai `ODBC Driver 17 for SQL Server`

### 2. Install dependency project

```powershell
cd qc-suite-python
py -3.11 -m pip install -e .
```

### 3. Siapkan environment

Salin `deploy/.env.example` menjadi `.env`, lalu isi sesuai kebutuhan environment Anda.

Minimal untuk local-only desktop:

```env
QC_SUITE_LOCAL_ONLY=1
QC_SUITE_DATABASE_BACKEND=local
QC_SUITE_SERVER_URL=local://embedded
QC_SUITE_SECRET_KEY=ganti-secret-anda
```

Jika ingin memakai PostgreSQL sebagai backend relasional:

```env
QC_SUITE_DATABASE_BACKEND=postgresql
POSTGRESQL_HOST=127.0.0.1
POSTGRESQL_PORT=5432
POSTGRESQL_DATABASE=qc_suite
POSTGRESQL_USERNAME=qc_suite_user
POSTGRESQL_PASSWORD=secret
POSTGRESQL_SCHEMA=public
POSTGRESQL_SSLMODE=prefer
```

Jika masih perlu SQL Server untuk kompatibilitas sementara, isi juga:

```env
QC_SUITE_DATABASE_BACKEND=sqlserver
MSSQL_SERVER=...
MSSQL_DATABASE=...
MSSQL_USERNAME=...
MSSQL_PASSWORD=...
MSSQL_DRIVER=ODBC Driver 17 for SQL Server
```

`.env` hanya berisi rahasia dan bootstrap (secret key, kredensial DB, data root, host/port,
local-only). Semua setting operasional — koneksi & alamat I/O PLC, timing inspeksi, mode
inferensi / model / device — ada di `data/json_store/machine_settings.json` dan diedit dari
**Admin → Machine Settings**. Tidak ada satupun yang dibaca dari `.env`. PC baru memakai
default (PLC off, dry-run, inference `auto`) sampai disimpan dari tab itu.

Jika model production belum siap, untuk smoke test lokal set **Machine Settings → Inference →
Mode** ke `classic`.

### 4. Jalankan aplikasi

Untuk sekali jalan yang langsung membuka backend dan frontend dalam satu perintah, pakai launcher desktop:

```powershell
cd qc-suite-python
py -3.11 scripts/run_desktop.py
```

Mode default local-only akan memakai backend embedded di proses yang sama. Jika ingin memaksa backend terpisah pada mesin yang sama, gunakan `--split`.

Untuk mode split deployment penuh atau remote client, jalankan backend di terminal lain:

```powershell
cd qc-suite-python
py -3.11 scripts/run_backend.py
```

Lalu set `QC_SUITE_LOCAL_ONLY=0` dan `QC_SUITE_SERVER_URL=http://IP:8100` pada client sebelum menjalankan `scripts/run_client.py`.

### 5. Login ke aplikasi

Login page mendukung dua cara:

- Username/password yang lama masih bisa dipakai seperti sebelumnya.
- RFID lewat Ajfwm RFID Reader mode USB keyboard HID. Pastikan cursor berada di field `RFID Card`, lalu scan kartu. UID kartu dikirim ke backend, dicocokkan dengan binding user, dan session/token tetap memakai mekanisme auth yang sama.

Kalau ingin login langsung dengan kartu, fokuskan field RFID lalu scan. Kalau ingin login manual, gunakan field username dan password di halaman yang sama.

### 6. Smoke check lokal

Smoke-check backend tanpa menyentuh runtime lama:

```powershell
cd qc-suite-python
py -3.11 scripts/smoke_api.py
py -3.11 -m unittest backend.tests.test_api_smoke
```

Untuk test client non-UI:

```powershell
cd qc-suite-python
py -3.11 -m pytest client_tk/tests/test_async_bridge.py client_tk/tests/test_frame_upload.py -q
```

## PLC / Remote I/O (Modbus TCP, RTU, atau Mitsubishi FX)

Semua konfigurasi PLC ada di **Admin → Machine Settings** (`data/json_store/machine_settings.json`):

- **Connection / Transport** — `enabled`, `dry_run`, transport `tcp` / `rtu` / `fx`, host/port
  atau serial port + baudrate, unit id, timeout. Perubahan section ini **butuh restart backend**:
  setelah Save, UI menawarkan restart otomatis (tombol **Restart Backend** juga ada di tab) —
  aplikasi ditutup, kamera & port PLC dilepas, lalu dijalankan ulang dengan perintah yang sama.
  Untuk backend remote (`run_client.py` ke server lain) restart harus dilakukan di server.
- **I/O Addresses** — alamat relay (CH3 Clamp, CH2 OK Light+Buzzer, CH1 Enji Buzzer), input
  (IN1 Release, IN2 Template Cycle, IN3 Clamp Feedback), accept pulse, guard reclamp,
  debounce release. Berlaku langsung saat Save.
- **Timer / Inspection Policy** dan **Inference / Model** — lihat tab.

Catatan singkat:

- Output ditulis sebagai coil ON/OFF (FC05); input dibaca sebagai discrete input (FC02). Tidak ada mode holding-register.
- `Dry Run` tetap aman untuk simulasi, karena hanya log command. PC baru default `enabled=false, dry_run=true`.
- Reject tidak lagi ditulis ke repository hasil inspeksi utama; alasan reject disimpan ke `data/json_store/reject_log.jsonl` dan bisa dilihat lewat `GET /inspection/reject-logs` untuk admin.
- Untuk counter accept-only di dashboard, kirim `decision_code=ACCEPT` ke `/dashboard/summary` dan `/dashboard/buckets`.

Setelah itu, cek status PLC lewat `GET /inspection/plc/status` dan pastikan pulsa accept terkirim saat inspeksi commit terjadi.

## Mode Split Deployment

Mode ini hanya diperlukan bila Anda ingin memisahkan backend dan client ke mesin berbeda.

- backend berjalan di PC server
- client Tkinter berjalan di PC operator/admin
- kamera tetap dibuka di sisi client
- frame dikirim dari client ke backend lewat HTTP

### 1. Siapkan PC server

Set environment backend di PC server:

```env
QC_SUITE_LOCAL_ONLY=0
QC_SUITE_HOST=0.0.0.0
QC_SUITE_PORT=8100
QC_SUITE_DEBUG=0
QC_SUITE_SECRET_KEY=ganti-secret-produksi
QC_SUITE_DATA_ROOT=D:\qc-suite-data
```

Model sticker dan mode inferensi diatur dari **Admin → Machine Settings → Inference**
(atau per template lewat `vision.model_path`), bukan dari `.env`.

Jika PostgreSQL dipakai:

```env
QC_SUITE_DATABASE_BACKEND=postgresql
POSTGRESQL_HOST=...
POSTGRESQL_PORT=5432
POSTGRESQL_DATABASE=...
POSTGRESQL_USERNAME=...
POSTGRESQL_PASSWORD=...
POSTGRESQL_SCHEMA=public
POSTGRESQL_SSLMODE=prefer
```

Jika SQL Server dipakai:

```env
QC_SUITE_DATABASE_BACKEND=sqlserver
MSSQL_SERVER=...
MSSQL_DATABASE=...
MSSQL_USERNAME=...
MSSQL_PASSWORD=...
MSSQL_DRIVER=ODBC Driver 17 for SQL Server
```

Relational backend hanya dipakai untuk `qc_user_accounts` dan `qc_inspection_push`.
Auth audit tetap lokal di `data/json_store/auth_audit.jsonl`, dan session auth
memory-only sehingga login aktif akan reset saat backend restart.

### 2. Jalankan backend di PC server

Untuk development/internal testing:

```powershell
cd qc-suite-python
py -3.11 scripts/run_backend.py
```

Untuk deploy yang lebih layak produksi, jangan pakai Flask dev server. Jalankan backend lewat `waitress`:

```powershell
cd qc-suite-python
py -3.11 -m pip install waitress
waitress-serve --host 0.0.0.0 --port 8100 backend.app.main:app
```

### 3. Buka akses jaringan

- pastikan PC server punya IP yang bisa diakses client, misalnya `192.168.1.10`
- buka firewall Windows untuk port `8100`
- pastikan client dan server berada di jaringan yang saling terhubung

### 4. Siapkan PC client

Set `QC_SUITE_LOCAL_ONLY=0` dan `QC_SUITE_SERVER_URL` di PC client agar mengarah ke IP server:

```env
QC_SUITE_LOCAL_ONLY=0
QC_SUITE_SERVER_URL=http://192.168.1.10:8100
```

Lalu jalankan client:

```powershell
cd qc-suite-python
py -3.11 -m pip install -e .
py -3.11 scripts/run_client.py
```

### 5. Verifikasi koneksi

Setelah backend dan client aktif:

- login dari client
- buka screen `Admin` untuk cek data terbaca
- buka screen `Operator`, load deployment, lalu start camera
- pastikan frame dari client bisa diproses backend server

## Notes

- Existing runtime in the repo is untouched.
- This project uses JSON/file storage by default, with optional PostgreSQL or SQL Server persistence selected via `QC_SUITE_DATABASE_BACKEND`.
- `local` is the default backend for desktop-only use, `postgresql` is the recommended relational backend for new deployments, and `sqlserver` is retained for compatibility.
- Inference mode is set in Machine Settings → Inference:
  - `auto`: pick the backend from the model file extension (Ultralytics / ONNX / OpenVINO / TFLite), fallback to classic contour inference
  - `ultralytics`: require the YOLO runtime and fail if unavailable
  - `classic`: deterministic fallback for smoke tests and local debugging
- **Import model** (Admin → Models → *Import Model Archive (.zip)*): zip berisi satu model — folder hasil export OpenVINO dari Ultralytics (`<nama>_openvino_model/` dengan `.xml` + `.bin` + `metadata.yaml`), `.pt`, `.onnx`, `.tflite`, atau hasil *Export* dari aplikasi ini. Model otomatis masuk ke `data/models/<nama>/` beserta `<nama>.meta.json` (class names diambil dari `metadata.yaml`); tanpa `metadata.yaml` import tetap jalan tapi label deteksi jadi angka. *Export* menyimpan zip yang bisa di-import lagi di PC lain.
- The only validation mode is QC Sticker (presence / class / position). `WRONG_TYPE` is the only terminal reject; everything else keeps inferring until ACCEPT or `COMMIT_TIMEOUT`.
- Dashboard summary and time buckets now aggregate persisted Phase 5 fields, including `station_id`, `sticker_backend`, `total_part_ready`, `avg_sticker_confidence`, and `avg_part_ready_match_ratio`.
- `GET /deployments/active` returns `{ "deployment": ... }` so the client can distinguish between no deployment and a valid deployment deterministically.
