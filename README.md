# QC Suite Python

Greenfield QC suite in a separate folder, built as:

- `backend/`: Flask API for auth, templates, deployments, inspection sessions, workstation, and dashboard
- `client_tk/`: Tkinter desktop shell with role-based screens for Operator, Admin, and Engineer
- `shared/`: contracts and enums shared across backend and client
- `docs/`: product, screens, API, and deployment notes
- `scripts/`: run and smoke-test helpers

Default runtime is local-first desktop: the client uses the embedded local transport by default, and split deployment is only needed for compatibility or remote access.

Default seeded users:

- `admin / admin123`
- `operator / operator123`
- `engineer / engineer123`

## Role Screens

- `Operator`: login, local camera, active deployment lookup, ROI update, live decision, tilt hard-reject status, DB write status
- `Admin`: templates, deployments, users, inspection results, dashboard
- `Engineer`: dataset upload, annotations, augment jobs, training jobs, model registry, portable model export/import, color calibration

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

Jika model production belum siap, untuk smoke test lokal Anda bisa memakai:

```env
QC_SUITE_STICKER_INFERENCE_MODE=classic
```

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

Untuk smoke test UI:

```powershell
cd qc-suite-python
py -3.11 -m unittest client_tk.tests.test_ui_smoke
```

## Remote I/O via Modbus TCP or RTU

Pilih transport secara manual di `.env`: `tcp` untuk Modbus TCP gateway / Ethernet remote I/O, atau `rtu` untuk relay module serial RS232/RS485.

Contoh TCP minimal:

```env
QC_SUITE_PLC_ENABLED=1
QC_SUITE_PLC_DRY_RUN=0
QC_SUITE_PLC_TRANSPORT=tcp
QC_SUITE_PLC_HOST=192.168.1.50
QC_SUITE_PLC_PORT=502
QC_SUITE_PLC_MODBUS_UNIT_ID=1
QC_SUITE_PLC_MODBUS_COMMAND_MODE=coil
QC_SUITE_PLC_MODBUS_HOLD_ADDRESS=0
QC_SUITE_PLC_MODBUS_RELEASE_ADDRESS=0
QC_SUITE_PLC_MODBUS_ZERO_BASED_ADDRESSING=1
```

Contoh RTU untuk relay module serial (misalnya slave id 255, coil 0/1):

```env
QC_SUITE_PLC_ENABLED=1
QC_SUITE_PLC_DRY_RUN=0
QC_SUITE_PLC_TRANSPORT=rtu
QC_SUITE_PLC_SERIAL_PORT=COM3
QC_SUITE_PLC_SERIAL_BAUDRATE=9600
QC_SUITE_PLC_SERIAL_PARITY=N
QC_SUITE_PLC_SERIAL_BYTESIZE=8
QC_SUITE_PLC_SERIAL_STOPBITS=1
QC_SUITE_PLC_MODBUS_UNIT_ID=255
QC_SUITE_PLC_MODBUS_COMMAND_MODE=coil
QC_SUITE_PLC_MODBUS_HOLD_ADDRESS=0
QC_SUITE_PLC_MODBUS_RELEASE_ADDRESS=1
QC_SUITE_PLC_MODBUS_ZERO_BASED_ADDRESSING=1
```

Catatan singkat:

- `coil` cocok kalau output remote I/O berupa coil ON/OFF.
- `holding_register` cocok kalau gateway Anda menulis nilai register tertentu.
- `QC_SUITE_PLC_DRY_RUN=1` tetap aman untuk simulasi, karena hanya log command.
- Jika perangkat Anda punya status balik, aktifkan `QC_SUITE_PLC_MODBUS_READBACK_ENABLED=1` dan isi alamat readback yang benar.
- Reject tidak lagi ditulis ke repository hasil inspeksi utama; alasan reject disimpan ke `data/json_store/reject_log.jsonl` dan bisa dilihat lewat `GET /inspection/reject-logs` untuk admin.
- Untuk counter accept-only di dashboard, kirim `decision_code=ACCEPT` ke `/dashboard/summary` dan `/dashboard/buckets`.

Setelah itu, cek status PLC dari backend admin endpoint dan pastikan satu cycle hold/release terkirim saat inspeksi commit terjadi.

## Mode Split Deployment

Mode ini hanya diperlukan bila Anda ingin memisahkan backend dan client ke mesin berbeda.

- backend berjalan di PC server
- client Tkinter berjalan di PC operator/admin/engineer
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
QC_SUITE_STICKER_INFERENCE_MODE=auto
QC_SUITE_DEFAULT_STICKER_MODEL_PATH=D:\qc-suite-data\models\sticker.pt
QC_SUITE_DEFAULT_STICKER_MODEL_META_PATH=D:\qc-suite-data\models\sticker.meta.json
```

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
- Sticker inference Phase 3 is wired to `D:\ProjectMagang\akh.pt` with metadata from `D:\ProjectMagang\ds-43598c556c__yolov5mu__20260402-085412.meta.json`.
- Runtime mode is controlled by `QC_SUITE_STICKER_INFERENCE_MODE`:
  - `auto`: try Ultralytics first, fallback to classic contour inference
  - `ultralytics`: require the YOLO runtime and fail if unavailable
  - `classic`: deterministic fallback for smoke tests and local debugging
- The engineer screen now supports portable model export/import from the Model Registry.
- Sticker tilt is a hard reject gate; when the configured threshold is exceeded, the operator sees `OUT_OF_ANGLE`.
- Dashboard summary and time buckets now aggregate persisted Phase 5 fields, including `station_id`, `sticker_backend`, `total_part_ready`, `avg_sticker_confidence`, and `avg_part_ready_match_ratio`.
- `GET /deployments/active` returns `{ "deployment": ... }` so the client can distinguish between no deployment and a valid deployment deterministically.
