# Windows Service Backend

Jalankan backend sebagai service Windows dengan wrapper seperti NSSM atau WinSW.

Command target:

```powershell
py -3.11 d:\ProjectMagang\aiflow\aski-flow\qc-suite-python\scripts\run_backend.py
```

Set environment sebelum service dijalankan:

- `QC_SUITE_DATA_ROOT`
- `QC_SUITE_HOST`
- `QC_SUITE_PORT`
- optional `MSSQL_*`

Pastikan service account punya akses ke:

- folder data root
- model files
- SQL Server jika diaktifkan
