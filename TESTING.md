# Panduan Testing

## Cara menjalankan

Dari repo root:

```sh
scripts/run_tests.sh            # seluruh suite backend
scripts/run_tests.sh -k contract   # teruskan argumen tambahan ke pytest
```

atau langsung:

```sh
python -m pytest backend/tests -q
python -m pytest backend/tests/test_evaluators.py -q          # satu file
python -m pytest backend/tests/test_evaluators.py -q -k Counter   # satu class/pattern
```

`conftest.py` di root dan `pyproject.toml [tool.pytest.ini_options]` membuat
suite bisa dijalankan dari repo root: keduanya membetulkan `sys.path`, mengekspor
`QC_SUITE_DEFAULT_STICKER_MODEL_PATH` dari `.pt` di dalam repo (cuma dipakai
`test_api_smoke.py`, yang menyalinnya ke `machine_settings.json` mesin test), dan
mendaftarkan marker custom. Setting runtime untuk test datang dari
`machine_settings.json` di test data root, tidak pernah dari env var.

### Kondisi yang diharapkan

Per 2026-09-18 (setelah penghapusan Data/Training, HANDOFF.md §9):
**176 passed, 2 failed, 7 skipped** (`pytest backend/tests`); client
`test_async_bridge.py test_frame_upload.py test_app_restart.py`: 9 passed;
`test_ui_smoke.py`: 12 passed / 8 failed (drift yang sudah ada sebelumnya, lihat
`.claude/CLAUDE.md`). Dua kegagalan itu memang sudah ada sebelumnya dan
didokumentasikan di `.claude/CLAUDE.md` (`test_00b` butuh `.pt` untuk seed
registry; `test_04d` adalah drift test/code pada `part_ready_ema_ratio`). Skip-nya
butuh model sticker terlatih yang asli (di luar repo); lihat `HANDOFF.md` section
2b untuk un-skip secara lokal.

(Sejarah: lima klaster test RED dari refactor multi-ronde sudah diputuskan
orchestrator sebagai redesign yang disengaja dan sudah diselesaikan — rewrite
adapter/worker PLC, global-binding deployment, akses operator `/plc/status`,
penghapusan OCR. Lihat `HANDOFF.md` section 3 "RESOLVED" dan section 6 "FASE 1
dead-code cleanup".)

Kalau nanti kamu mengubah apa yang *dilakukan* kode, jangan bungkam test yang
gagal dengan mengedit test-nya supaya lolos — pilih salah satu: kembalikan
behavior-nya (test jadi hijau) atau hapus/tulis ulang test yang sudah usang
**dan** update `HANDOFF.md`.

### Isolasi data root (baca sebelum menambah file test)

`conftest.py` → `backend/tests/_test_env.py::ensure_test_data_root()` memberi
seluruh run `QC_SUITE_DATA_ROOT` yang sekali pakai dengan `machine_settings.json`
test **sebelum modul test mana pun di-import**, dan menimpa nilai yang diwarisi
apa pun. Jangan pernah set `QC_SUITE_DATA_ROOT` di dalam modul test: `config.py`
mengunci `DATA_ROOT` di import pertama, jadi importer pertama yang menang — pada
2026-09-18 sebuah file baru yang urutan namanya sebelum `test_api_smoke.py` sempat
menjalankan tiga suite melawan `data/` asli milik developer (HANDOFF.md §9f).
Modul yang juga harus jalan lewat `python -m unittest` memanggil helper yang sama
(idempotent).

### Dihapus: datasets / annotation / augment / training (disengaja, 2026-09-18)

Training dilakukan di software lain. Repository, worker, route, tab Admin dataset,
annotation, augment, dan training beserta test-nya sudah hilang (HANDOFF.md §9).
Jangan ditambahkan lagi. Model *import* adalah yang tersisa dan dicakup
`backend/tests/test_model_export_import.py` (zip OpenVINO → `data/models/<name>/` +
`.meta.json`, export legacy, round-trip, purge).

### Dihapus: validasi sticker OCR (disengaja)

Validasi sticker berbasis OCR **dihapus dengan sengaja**. Mode sticker sekarang
memvalidasi **presence / position / tilt**, BUKAN kode/konten. Test khusus OCR
sudah dihapus: `OcrAnchorPrimaryGateTest`, kasus OCR dari `StickerOnlyOcrGateTest`
(di `test_sticker_detection_gates.py`), dan test payload `_augment_with_*_ocr` (di
`test_sticker_inference.py`). Test non-OCR (tilt gate, geometry/position, helper
normalisasi teks OCR yang masih ada) dipertahankan. JANGAN tambahkan lagi test yang
bergantung pada field OCR `StickerRule`/`VisionConfig` yang sudah dihapus
(`use_ocr`, `ocr_expected_code`, `ocr_mode`, `ocr_engine`, `expected_dot_x/y`,
`max_anchor_offset`). Kode helper OCR terakhir dan test-nya dihapus pada
2026-09-18 (HANDOFF.md §7).

## Aturannya

**Perubahan behavior apa pun harus dikirim bersama test-nya dalam commit yang sama.**

Kalau kamu mengubah apa yang *dilakukan* kode (field baru, param yang dihapus,
keputusan yang berbeda, endpoint baru, key contract yang diganti nama), commit
yang mengubahnya juga harus menambah atau memperbarui test yang membuktikan
behavior baru itu. PR yang mengubah behavior tanpa test dianggap belum lengkap.
Ini persis failure mode yang dibereskan FASE 0: lima ronde refactor membuat kode
melenceng dari test-nya, menyisakan ~49 kegagalan diam-diam dan nol coverage di
package evaluator.

## Cara menambah test

Test memakai gaya stdlib `unittest` (suite-nya berbasis `unittest`; pytest yang
menjalankannya). Ikuti idiom yang sudah ada:

1. Taruh di `backend/tests/test_<area>.py`. Satu subclass `unittest.TestCase` per
   unit logis; nama method `test_*`. Pakai `self.subTest(...)` untuk kasus table-driven.
2. **Utamakan unit test dibanding integration.** Import unit-nya langsung:
   - Contracts: `from shared.contracts.templates import template_from_dict, ...`
   - Evaluators: `from backend.app.services.evaluators.counter import CounterEvaluator`
     Bangun `EvalContext` minimal (lihat `test_evaluators.py::_ctx`) dan
     `SessionState` minimal (lihat `_make_state`).
3. **Stub dependency eksternal**, jangan sentuh hardware/model/DB asli:
   - Anomaly scorer: `mock.patch("backend.app.services.evaluators.defect.get_scorer",
     return_value=_StubScorer(...))`.
   - Client PLC: patch `backend.app.services.plc_adapter.ModbusTcpClient` dengan
     `MagicMock` (pola di `test_plc_modbus_adapter.py::_make_mock_client`).
   - HTTP/API: pakai `create_app().test_client()` (pola di `test_api_smoke.py`).
4. **Pastikan JSON safety** untuk apa pun yang jadi `Decision.details` / payload
   API: `json.dumps(payload, allow_nan=False)` tidak boleh error. `Infinity`/`NaN`
   diam-diam merusak layer WebSocket/HTTP.
5. Kalau satu test memang butuh infra asli (model terlatih, hardware PLC, DB
   live), tandai dengan `@unittest.skip("alasan ... lihat HANDOFF.md")` atau marker
   `requires_real_sticker_model` / `requires_plc_hardware` — jangan pernah
   dibiarkan jadi kegagalan diam-diam.

## Golden fixture

`backend/tests/fixtures/golden_template_{sticker,counter,defect}.json` adalah
template contract yang dibekukan. `test_golden_templates.py` memastikan mereka
ter-parse, tervalidasi, dan bisa dispatch ke `Decision`. Perlakukan sebagai
append-only: kalau bentuk field template berubah, update fixture-nya **dan**
jelaskan alasannya di commit — fase-fase berikutnya bergantung pada fixture ini
tetap stabil.
