# FASE 0 — Regression Safety Net: Handoff

Author: SUBAGENT NET. Branch: `rev1`.
Scope of my edits: `backend/tests/`, `scripts/`, `conftest.py`, `pyproject.toml`
`[tool.pytest]`, `backend/tests/fixtures/`, this file, `TESTING.md`. **No production
code was modified** (nothing under `backend/app/`, `shared/`, `client_tk/app/`).

Reproduce the suite:

```sh
scripts/run_tests.sh
# or
python -m pytest backend/tests -q
```

---

## 1. Final suite state

| Metric | Baseline | After triage | After adjudication (final) |
| --- | --- | --- | --- |
| passed | 229 | 292 | **315** |
| failed | 49 | 37 | **0** |
| skipped | 1 | 10 | **10** (real-model integration tests, see 2b) |

The suite is now **FULLY GREEN** (0 failed). The 10 skips are the api_smoke
integration tests that require the production sticker model (outside the repo) —
that's acceptable and expected; see 2b.

**History:** After my initial triage the 5 regression clusters R1–R5 were kept RED
pending a human decision. The human orchestrator then ADJUDICATED all five as
INTENTIONAL redesigns (not accidental regressions), so those tests were obsolete
tests for deliberately-removed/changed features. I resolved them per the decisions
in section 3 (RESOLVED) — rewriting to cover the NEW behavior where coverage
mattered (PLC adapter/worker, deployment, plc/status auth) and deleting only the
truly-dead OCR tests. Production dead-code cleanup that these decisions imply is
out of my region and is captured in section 6 (FASE 1 handoff).

---

## 2. Triage buckets

### 2a. Fixed as ENV/INFRA (test harness / seeding)

| Test | Root cause | Fix |
| --- | --- | --- |
| `test_api_smoke::test_00b_seeded_model_registry_contains_default_model` | `models_repository._default_models_payload()` seeds an EMPTY registry when `QC_SUITE_DEFAULT_STICKER_MODEL_PATH` is blank (unset in a bare checkout). | `conftest.py` now sets that env var to an in-repo `.pt` (`yolov5su.pt`) if unset. Registry becomes non-empty; test passes. |

### 2b. Skipped as ENV/INFRA — need the production sticker model (outside repo)

These 10 integration tests drive the full inspection pipeline and expect an
`ACCEPT` + DB commit on a synthetic white-rectangle image. That only happens with
the real trained **"AKH Sticker Detector"** model, which lives OUTSIDE the repo
(`QC_SUITE_DEFAULT_STICKER_MODEL_PATH=D:\qc-suite-data\models\sticker.pt`, per
README). A generic `yolov5su.pt` cannot detect the synthetic sticker → decision
stays `REJECT`, nothing commits, and every downstream assertion (`count_committed`,
`result_id`, `total_inspections`, settle-timing) fails. NOT a code regression.

Marked with `@unittest.skip(_REQUIRES_REAL_STICKER_MODEL)` in `test_api_smoke.py`:

- `test_01_operator_flow_accepts_centered_detection`
- `test_02_part_ready_color_gate_blocks_commit_until_match`
- `test_08_engineer_metadata_roundtrip_and_filtered_queries`
- `test_08a_training_job_request_records_metadata`
- `test_10a_admin_can_patch_inspection_with_audit_trail`
- `test_10b_admin_can_delete_inspection_with_audit_trail`
- `test_13b_settle_zero_bypasses_debounce`
- `test_13g_commit_stable_frames_does_not_override_settle_ms`
- `test_13h_settle_ms_controls_commit_after_settle_window`
- `test_15g_rejects_are_logged_locally_and_not_persisted_to_results_db`

**To un-skip:** point `QC_SUITE_DEFAULT_STICKER_MODEL_PATH` /
`QC_SUITE_DEFAULT_STICKER_MODEL_META_PATH` at the real sticker model + meta and
remove the decorators. A later phase should provide a small checked-in test model
(or a deterministic fake detection backend) so these run in CI.

### 2c. No genuinely-stale tests were silently rewritten-to-pass during triage

During triage I did not paper over any failure. Every RED test was either an
env/infra skip or was escalated to the orchestrator as a suspected regression
(section 3). The classic "stale drift" symptom (`ModbusTcpClient('10.0.0.5', ...)`
positional vs `host=` keyword) was part of the PLC-adapter rewrite cluster (R1) and
was resolved only after the orchestrator confirmed the rewrite was intentional.

---

## 3. RESOLVED — orchestrator decided (intentional redesign)

The human orchestrator adjudicated all five clusters as INTENTIONAL redesigns. The
RED tests were therefore obsolete tests for deliberately-removed/changed features. I
resolved each below — keeping coverage of the NEW behavior wherever it mattered, and
deleting only genuinely-dead tests. All are now GREEN.

### R1 — PLC adapter: minimal `write_coil`/`read_inputs`/`slave_id` design is intended
**Decision:** the minimal adapter is the new design (clamp/readback/command-mode API
intentionally removed).
**Action taken:** REPLACED `backend/tests/test_plc_modbus_adapter.py` — retired the 23
obsolete old-API tests and wrote a lean suite (19 tests) for the API that exists now:
`DryRunPlcAdapter` lifecycle/status/read_inputs; `ModbusTcpPlcAdapter` host/port/timeout
wiring + lazy-connect + `write_coil(addr,val,device_id=slave_id)` + FC02 read + error
raise; `ModbusRtuPlcAdapter` constructor + write; `build_plc_adapter` selection
(dry-run/tcp/rtu/unknown→dry-run). PLC adapter coverage did not drop to zero.
**Dead production code this leaves → FASE 1:** see section 6.

### R2 — PLC worker: accept-pulse + input-polling model is intended (no `hold_ms`)
**Decision:** clamp hold/release + `hold_ms` intentionally replaced.
**Action taken:** REWROTE the three worker tests for the new behavior:
- `test_15e` → `test_15e_plc_worker_notify_decision_enqueues_once`: asserts
  `worker.notify_decision(...)` enqueues one command per decision (dry-run).
- `test_15f`: asserts `DryRunPlcAdapter` write/read/all_off + status (was send_clamp_*).
- The `test_plc_modbus_adapter` worker test → `PlcWorkerInputPollingTest`: IN1
  manual-release needs stable debounce before all-off; IN2 template-cycle once per
  debounce; `notify_decision` enqueues. Worker coverage preserved.

### R3 — `/inspection/plc/status` operator access is intended
**Decision:** operators legitimately need to see PLC status.
**Action taken:** RENAMED `test_15b` → `test_15b_plc_status_allows_operator_but_rejects_anonymous`;
now asserts operator and admin get `200` and an unauthenticated caller gets `401`.

### R4 — Deployment is a single global active binding (no line/station slots)
**Decision:** the global-binding redesign is intended.
**Action taken:** REWROTE the two deployment tests to the new semantics:
- `test_11b`: admin PUT re-binds the deployment to a new template version;
  `/deployments/active` (no query params) returns the single active binding reflecting
  the re-bound version; operator PUT still `403`.
- `test_11d` → `test_11d_get_active_returns_latest_of_multiple_active_deployments`:
  deploying twice leaves both active; `get_active` returns the latest; both created IDs
  appear active in the list (scoped by created IDs, not by removed slot fields).

### R5 — OCR sticker validation fully removed by design
**Decision:** OCR is fully removed; sticker now validates presence/position/tilt, NOT
code/content.
**Action taken:** DELETED only the OCR-specific tests and PRESERVED the rest:
- `test_sticker_detection_gates.py`: deleted `OcrAnchorPrimaryGateTest` and the OCR
  cases of `StickerOnlyOcrGateTest`; kept the non-OCR tilt-normalization test (class
  renamed `TiltNormalizationTest`); all tilt-gate / observability / backward-compat
  classes untouched.
- `test_sticker_inference.py`: deleted the two `_augment_with_*_ocr` payload tests; kept
  the OCR text-normalization utility tests (`_normalize_ocr_text`, `_parse_unique_code`,
  flip-fallback) which test helpers that still exist.
A retirement note was added to `TESTING.md`.
**Dead production code this leaves → FASE 1:** see section 6.

---

## 4. Other code-smells found (non-blocking, for later cleanup)

- **Defect evaluator dead branch / inf risk:** `DefectEvaluator`'s `w <= 0`
  "empty crop" branch is effectively unreachable because `_parse_geometry` clamps
  `w`/`h` to `>= 1` (`max(1, ...)`). Separately, `_aggregate_score` returns
  `float("inf")` for an empty slice — if that value ever reaches `Decision.details`,
  `json.dumps(..., allow_nan=False)` raises. Pinned by
  `test_evaluators.py::DefectEvaluatorTest::test_aggregate_score_empty_slice_is_infinite_and_leaks_to_json`.
- **Sticker has no wired evaluator:** `registry.py` leaves `StickerEvaluator`
  commented out (TODO B5); sticker still runs an inline path in
  `InspectionSessionService._validate_sticker`. Pinned by
  `test_golden_templates.py::...::test_sticker_mode_normalizes_but_has_no_wired_evaluator`.
  When B5 lands, update that test + this note.
- **Cross-file test state pollution:** `test_07_admin_...` passes alone and in the
  api-file-only run but was sensitive to full-suite ordering during triage. The api
  suite shares a JSON store under `QC_SUITE_DATA_ROOT`; deployment/inspection tests
  accumulate state. Not fixed here (would need per-test isolation across the 152 KB
  file). Watch for flakiness.

---

## 5. New tests added (the safety net)

| File | Count | What it locks in |
| --- | --- | --- |
| `backend/tests/test_templates_contract.py` | 28 | round-trip idempotency (all modes), legacy-only + criteria-only parse, `normalize_mode` aliases, min/max count semantics, `validate_criteria` messages |
| `backend/tests/test_evaluators.py` | 25 | first direct coverage of `evaluators/*`: counter (in/out/foreign/multi-ROI), defect w/ scorer stub (all-OK/one-NG/model-fail/no-frame), sticker (pass/wrong-type/low-conf/disabled/not-found), registry unknown-mode, JSON-safety on every branch |
| `backend/tests/test_golden_templates.py` | 7 | 3 golden fixtures parse+validate+dispatch; `Decision -> validation_details` intact + JSON-safe |
| `backend/tests/fixtures/golden_template_{sticker,counter,defect}.json` | 3 | frozen template contracts for later phases |

Total new: **60 tests** (all green). After adjudication the PLC adapter/worker and
deployment/plc-status tests were rewritten to the new behavior (section 3), so the
final suite is **315 passed / 0 failed / 10 skipped**.

---

## 6. FASE 1 DEAD-CODE CLEANUP (production — OUT OF MY REGION)

The orchestrator's "intentional redesign" decisions leave dead production code that I
must NOT touch (it lives under `backend/app/` / `shared/`). Capture for FASE 1:

### 6a. CONTRACT / RUNTIME — dead PLC Modbus settings (from R1)
`backend/app/core/config.py` still DEFINES, and
`backend/app/repositories/machine_settings_repository.py` still PERSISTS, settings the
new minimal adapter ignores entirely:
- `plc_modbus_command_mode`
- `plc_modbus_zero_based_addressing`
- `plc_modbus_readback_mode`
- hold/release addresses + expected hold/release values (readback pair)

These are now NO-OP settings. They will still render in the admin UI as if they do
something. **FASE 1 action:** remove/reconcile these config fields + their persistence
+ any admin-UI widgets that expose them, so the settings surface matches the adapter.

### 6b. RUNTIME — dead OCR reads in sticker inference (from R5)
`backend/app/services/sticker_inference.py` still READS removed fields via
`getattr(sticker_rule, "use_ocr", False)`, `getattr(..., "ocr_expected_code", ...)`,
`getattr(..., "expected_dot_x/y", ...)` — always the defaults now, since `StickerRule`
dropped those fields and `templates._VALID_STICKER_FIELDS` strips them on parse. And
`_resolve_ocr_engine()` is hardcoded `return "disabled"`. The
`_augment_with_anchor_ocr` / `_augment_with_ocr_only` code paths (and any OCR path in
`InspectionSessionService._validate_sticker`) are now dead. **FASE 1 action:** delete
the dead OCR reads/methods now that OCR is officially removed. (`StickerEvaluator` also
still has OCR-shaped `additional` handling that is moot — see it when wiring B5.)

### 6c. RUNTIME — PLC hardening targets the NEW design (note for FASE 1)
The "Ketahanan PLC" / PLC-resilience hardening must target the NEW minimal
accept-pulse + input-polling model (`ModbusTcpPlcAdapter.write_coil`/`read_inputs`,
`PlcWorker` accept-pulse + `_poll_inputs` + strategy). **There is no clamp/hold/release
or readback API to harden** — do not design hardening around the removed API.

### 6d. Non-blocking code-smells still open (from section 4)
- `DefectEvaluator` `w<=0` empty-crop branch is dead (geometry clamps to ≥1px);
  `_aggregate_score` returns `float("inf")` on empty slices (JSON-unsafe if it ever
  reaches `Decision.details`). Pinned by a test.
- `StickerEvaluator` is still not wired into `registry.py` (TODO B5); sticker uses the
  inline `_validate_sticker` path. Pinned by a test.
- api_smoke cross-file shared state under `QC_SUITE_DATA_ROOT` — watch for flakiness.
