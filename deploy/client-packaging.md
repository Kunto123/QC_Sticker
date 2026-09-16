# Client Packaging

## Development Run

```powershell
py -3.11 d:\ProjectMagang\aiflow\aski-flow\qc-suite-python\scripts\run_client.py
```

## Onefile Packaging Example

```powershell
py -3.11 -m pip install pyinstaller
py -3.11 -m PyInstaller --noconfirm --windowed --name qc-suite-client d:\ProjectMagang\aiflow\aski-flow\qc-suite-python\scripts\run_client.py
```

Set environment variable `QC_SUITE_SERVER_URL` di workstation agar client mengarah ke backend yang benar.
