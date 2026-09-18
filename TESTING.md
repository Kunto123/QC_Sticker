# Testing Guide

## How to run

From the repo root:

```sh
scripts/run_tests.sh            # whole backend suite
scripts/run_tests.sh -k contract   # forward extra args to pytest
```

or directly:

```sh
python -m pytest backend/tests -q
python -m pytest backend/tests/test_evaluators.py -q          # one file
python -m pytest backend/tests/test_evaluators.py -q -k Counter   # one class/pattern
```

The root `conftest.py` and `pyproject.toml [tool.pytest.ini_options]` make the
suite runnable from the repo root: they fix `sys.path`, export
`QC_SUITE_DEFAULT_STICKER_MODEL_PATH` from an in-repo `.pt` (consumed only by
`test_api_smoke.py`, which copies it into the test machine's
`machine_settings.json`), and register custom markers. Runtime settings for tests
come from a `machine_settings.json` in the test data root, never from env vars.

### Expected state

As of 2026-09-18 (after Data/Training removal, HANDOFF.md §9):
**170 passed, 2 failed, 7 skipped** (`pytest backend/tests`); client
`test_async_bridge.py test_frame_upload.py test_app_restart.py`: 9 passed;
`test_ui_smoke.py`: 12 passed / 8 failed (pre-existing drift, see `.claude/CLAUDE.md`).
The two failures are pre-existing and documented in `.claude/CLAUDE.md`
(`test_00b` needs a `.pt` for the registry seed; `test_04d` is test/code drift on
`part_ready_ema_ratio`). The skips need the real trained sticker model (outside
the repo); see `HANDOFF.md` section 2b to un-skip them locally.

(History: five clusters of RED tests from a multi-round refactor were adjudicated by
the orchestrator as intentional redesigns and resolved — PLC adapter/worker rewrite,
deployment global-binding, `/plc/status` operator access, OCR removal. See
`HANDOFF.md` section 3 "RESOLVED" and section 6 "FASE 1 dead-code cleanup".)

If you later change what the code *does*, do not silence a failing test by editing it
to pass — either restore the behavior (test goes green) or delete/rewrite the obsolete
test **and** update `HANDOFF.md`.

### Retired: datasets / annotation / augment / training (by design, 2026-09-18)

Training happens in other software. The dataset, annotation, augment and training
repositories, workers, routes, Admin tabs and their tests are gone (HANDOFF.md §9).
Do not re-add them. Model *import* is what remains and is covered by
`backend/tests/test_model_export_import.py` (OpenVINO zip → `data/models/<name>/` +
`.meta.json`, legacy export, round-trip, purge).

### Retired: OCR sticker validation (by design)

OCR-based sticker validation was **removed on purpose**. Sticker mode now validates
**presence / position / tilt**, NOT code/content. The OCR-specific tests were retired:
`OcrAnchorPrimaryGateTest`, the OCR cases of `StickerOnlyOcrGateTest` (in
`test_sticker_detection_gates.py`), and the `_augment_with_*_ocr` payload tests (in
`test_sticker_inference.py`). Non-OCR tests (tilt gates, geometry/position, OCR
text-normalization helpers that still exist) were preserved. Do NOT re-add tests that
depend on the removed `StickerRule`/`VisionConfig` OCR fields (`use_ocr`,
`ocr_expected_code`, `ocr_mode`, `ocr_engine`, `expected_dot_x/y`, `max_anchor_offset`).
The last OCR helper code and its tests were deleted on 2026-09-18 (HANDOFF.md §7).

## The rule

**Any behavior change ships with its test in the same commit.**

If you change what the code *does* (a new field, a dropped param, a different
decision, a new endpoint, a renamed contract key), the commit that changes it must
also add or update the test that proves the new behavior. A PR that changes behavior
without a test is incomplete. This is exactly the failure mode FASE 0 cleaned up:
five refactor rounds drifted the code away from its tests, leaving ~49 silent
failures and zero coverage on the evaluator package.

## How to add a test

Tests use the stdlib `unittest` style (the suite is `unittest`-based; pytest runs
it). Match the existing idioms:

1. Put it in `backend/tests/test_<area>.py`. One `unittest.TestCase` subclass per
   logical unit; name methods `test_*`. Use `self.subTest(...)` for table-driven cases.
2. **Prefer unit tests over integration.** Import the unit directly:
   - Contracts: `from shared.contracts.templates import template_from_dict, ...`
   - Evaluators: `from backend.app.services.evaluators.counter import CounterEvaluator`
     Build a minimal `EvalContext` (see `test_evaluators.py::_ctx`) and a minimal
     `SessionState` (see `_make_state`).
3. **Stub external dependencies**, don't hit real hardware/models/DB:
   - Anomaly scorer: `mock.patch("backend.app.services.evaluators.defect.get_scorer",
     return_value=_StubScorer(...))`.
   - PLC client: patch `backend.app.services.plc_adapter.ModbusTcpClient` with a
     `MagicMock` (pattern in `test_plc_modbus_adapter.py::_make_mock_client`).
   - HTTP/API: use `create_app().test_client()` (pattern in `test_api_smoke.py`).
4. **Assert JSON safety** for anything that becomes a `Decision.details` / API
   payload: `json.dumps(payload, allow_nan=False)` must not raise. `Infinity`/`NaN`
   silently break the WebSocket/HTTP layer.
5. If a test genuinely needs real infra (a trained model, PLC hardware, a live DB),
   mark it `@unittest.skip("reason ... see HANDOFF.md")` or with the
   `requires_real_sticker_model` / `requires_plc_hardware` marker — never leave it
   as a silent failure.

## Golden fixtures

`backend/tests/fixtures/golden_template_{sticker,counter,defect}.json` are frozen
template contracts. `test_golden_templates.py` asserts they parse, validate, and
dispatch to a `Decision`. Treat them as append-only: if a template field changes
shape, update the fixture **and** explain why in the commit — later phases depend on
these staying stable.
