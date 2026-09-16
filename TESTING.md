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
suite runnable from the repo root: they fix `sys.path`, seed
`QC_SUITE_DEFAULT_STICKER_MODEL_PATH` from an in-repo `.pt` so the model registry
seeds non-empty, and register custom markers.

### Expected state

The suite is **fully green: 0 failed**. Ten `test_api_smoke` tests are skipped
because they need the real trained sticker model (outside the repo); that is
expected — see `HANDOFF.md` section 2b to un-skip them locally.

(History: five clusters of RED tests from a multi-round refactor were adjudicated by
the orchestrator as intentional redesigns and resolved — PLC adapter/worker rewrite,
deployment global-binding, `/plc/status` operator access, OCR removal. See
`HANDOFF.md` section 3 "RESOLVED" and section 6 "FASE 1 dead-code cleanup".)

If you later change what the code *does*, do not silence a failing test by editing it
to pass — either restore the behavior (test goes green) or delete/rewrite the obsolete
test **and** update `HANDOFF.md`.

### Retired: OCR sticker validation (by design)

OCR-based sticker validation was **removed on purpose**. Sticker mode now validates
**presence / position / tilt**, NOT code/content. The OCR-specific tests were retired:
`OcrAnchorPrimaryGateTest`, the OCR cases of `StickerOnlyOcrGateTest` (in
`test_sticker_detection_gates.py`), and the `_augment_with_*_ocr` payload tests (in
`test_sticker_inference.py`). Non-OCR tests (tilt gates, geometry/position, OCR
text-normalization helpers that still exist) were preserved. Do NOT re-add tests that
depend on the removed `StickerRule`/`VisionConfig` OCR fields (`use_ocr`,
`ocr_expected_code`, `ocr_mode`, `ocr_engine`, `expected_dot_x/y`, `max_anchor_offset`).

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
