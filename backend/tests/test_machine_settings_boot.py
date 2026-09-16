"""Machine settings: env seeds once, then the DB is what the runtime uses.

Locks in the contract decided on 2026-09-16:
- Fresh PC (empty store): MachineSettings is seeded from env (AppConfig).
- After that: the persisted JSON wins. seed_from_env(force=False) must not
  overwrite a user-edited store; only an explicit force re-seed does.
- Boot builds the PLC adapter from settings.connection, and applies the
  sticker section + guards to PlcWorker and the timing/clamp-feedback section
  to InspectionSessionService — the pieces container.py wires together.
- PUT /machine-settings must not flip PlcWorker dry_run (configure_guards
  without dry_run keeps the current value).
"""
from __future__ import annotations

import dataclasses
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import AppConfig
from backend.app.models.machine_settings import MachineSettings, PlcConnectionConfig
from backend.app.repositories.machine_settings_repository import MachineSettingsRepository
from backend.app.services.inspection_session import InspectionSessionService
from backend.app.services.plc_adapter import (
    DryRunPlcAdapter,
    ModbusRtuPlcAdapter,
    ModbusTcpPlcAdapter,
    build_plc_adapter_from_connection,
)
from backend.app.workers.plc_worker import PlcWorker


def _make_mock_client(*, input_bits=None) -> mock.MagicMock:
    client = mock.MagicMock()
    client.connected = False

    def _connect() -> bool:
        client.connected = True
        return True

    client.connect.side_effect = _connect
    ok = mock.MagicMock()
    ok.isError.return_value = False
    client.write_coil.return_value = ok
    di = mock.MagicMock()
    di.isError.return_value = False
    di.bits = list(input_bits) if input_bits is not None else [False] * 8
    client.read_discrete_inputs.return_value = di
    return client


class _TempStoreMixin:
    """Redirect JsonRepository's store dir to a temp folder so these tests never
    touch the real data/json_store/machine_settings.json."""

    def setUp(self) -> None:
        super().setUp()
        self._tmp = Path(tempfile.mkdtemp(prefix="qc-ms-"))
        self._patch = mock.patch(
            "backend.app.repositories.base_json.JSON_STORE_DIR", self._tmp
        )
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.addCleanup(lambda: shutil.rmtree(self._tmp, ignore_errors=True))


class SeedSemanticsTest(_TempStoreMixin, unittest.TestCase):
    def test_fresh_store_seeds_from_env(self) -> None:
        cfg = dataclasses.replace(AppConfig(), plc_transport="rtu", plc_serial_port="COM9", commit_grace_ms=4242)
        repo = MachineSettingsRepository()

        self.assertTrue(repo.seed_from_env(cfg, force=False))

        loaded = repo.load_settings()
        self.assertTrue(loaded.seeded_from_env)
        self.assertEqual(loaded.connection.transport, "rtu")
        self.assertEqual(loaded.connection.serial_port, "COM9")
        self.assertEqual(loaded.timing.commit_grace_ms, 4242)

    def test_existing_store_is_not_overwritten_on_reboot(self) -> None:
        cfg = dataclasses.replace(AppConfig(), commit_grace_ms=1500)
        repo = MachineSettingsRepository()
        repo.seed_from_env(cfg, force=False)

        # User edits via UI (PUT marks seeded_from_env=False)
        edited = repo.load_settings()
        edited.timing.commit_grace_ms = 999
        edited.connection.transport = "tcp"
        edited.seeded_from_env = False
        repo.save_settings(edited)

        # Next boot with a *different* env must not clobber the edit
        cfg_next_boot = dataclasses.replace(cfg, commit_grace_ms=1, plc_transport="fx")
        self.assertFalse(repo.seed_from_env(cfg_next_boot, force=False))

        loaded = repo.load_settings()
        self.assertEqual(loaded.timing.commit_grace_ms, 999)
        self.assertEqual(loaded.connection.transport, "tcp")
        self.assertFalse(loaded.seeded_from_env)

    def test_force_reseed_overwrites_with_env(self) -> None:
        repo = MachineSettingsRepository()
        repo.seed_from_env(dataclasses.replace(AppConfig(), commit_grace_ms=1500), force=False)
        edited = repo.load_settings()
        edited.timing.commit_grace_ms = 999
        edited.seeded_from_env = False
        repo.save_settings(edited)

        self.assertTrue(repo.seed_from_env(dataclasses.replace(AppConfig(), commit_grace_ms=77), force=True))
        loaded = repo.load_settings()
        self.assertEqual(loaded.timing.commit_grace_ms, 77)
        self.assertTrue(loaded.seeded_from_env)


class AdapterFromConnectionTest(unittest.TestCase):
    def test_dry_run_wins_regardless_of_transport(self) -> None:
        conn = PlcConnectionConfig(dry_run=True, transport="tcp", host="10.0.0.9")
        self.assertIsInstance(build_plc_adapter_from_connection(conn), DryRunPlcAdapter)

    def test_tcp_uses_connection_fields(self) -> None:
        conn = PlcConnectionConfig(dry_run=False, transport="tcp", host="10.0.0.9", port=1502, timeout_ms=750)
        mc = _make_mock_client()
        with mock.patch("backend.app.services.plc_adapter.ModbusTcpClient", return_value=mc) as Cls:
            adapter = build_plc_adapter_from_connection(conn)
        self.assertIsInstance(adapter, ModbusTcpPlcAdapter)
        Cls.assert_called_once_with(host="10.0.0.9", port=1502, timeout=0.75)

    def test_transport_is_case_insensitive(self) -> None:
        conn = PlcConnectionConfig(dry_run=False, transport="TCP", host="10.0.0.9", port=502)
        with mock.patch("backend.app.services.plc_adapter.ModbusTcpClient", return_value=_make_mock_client()):
            adapter = build_plc_adapter_from_connection(conn)
        self.assertIsInstance(adapter, ModbusTcpPlcAdapter)

    def test_rtu_uses_serial_fields(self) -> None:
        conn = PlcConnectionConfig(
            dry_run=False, transport="rtu", serial_port="COM5", serial_baudrate=19200,
            serial_parity="E", serial_bytesize=8, serial_stopbits=1, timeout_ms=500, modbus_unit_id=3,
        )
        with mock.patch("backend.app.services.plc_adapter.ModbusSerialClient", return_value=_make_mock_client()) as Cls:
            adapter = build_plc_adapter_from_connection(conn)
        self.assertIsInstance(adapter, ModbusRtuPlcAdapter)
        Cls.assert_called_once_with(port="COM5", baudrate=19200, parity="E", stopbits=1, bytesize=8, timeout=0.5)

    def test_unknown_transport_degrades_to_dry_run(self) -> None:
        conn = PlcConnectionConfig(dry_run=False, transport="banana")
        self.assertIsInstance(build_plc_adapter_from_connection(conn), DryRunPlcAdapter)


def _settings_for_worker() -> MachineSettings:
    s = MachineSettings()
    s.sticker.relay_clamp_address = 7
    s.sticker.relay_ok_light_buzzer_address = 6
    s.sticker.relay_enji_buzzer_address = 5
    s.sticker.input_release_address = 4
    s.sticker.input_template_address = 3
    s.sticker.input_clamp_engaged_address = 2
    s.sticker.clamp_feedback_enabled = True
    s.sticker.accept_pulse_ms = 321
    s.sticker.min_reclamp_interval_ms = 1234
    s.sticker.release_input_debounce_ms = 56
    return s


class WorkerApplyMachineSettingsTest(unittest.TestCase):
    def _worker(self) -> PlcWorker:
        # Constructor args mimic what env would have given: all different from
        # the settings above so any leak of legacy values is visible.
        w = PlcWorker(
            DryRunPlcAdapter(),
            accept_pulse_ms=1000,
            input_release_address=0,
            input_template_address=1,
            input_clamp_engaged_address=9,
            clamp_feedback_enabled=False,
            relay_clamp_address=0,
            relay_ok_light_buzzer_address=1,
            relay_enji_buzzer_address=2,
        )
        w.configure_guards(min_reclamp_interval_ms=3000, release_input_debounce_ms=500, dry_run=True)
        return w

    def test_apply_syncs_io_mirror_used_by_poll_loop_and_status(self) -> None:
        w = self._worker()
        w.apply_machine_settings(_settings_for_worker())

        status = w.status()
        self.assertEqual(status["strategy"], "sticker-flow")
        self.assertTrue(status["clamp_feedback_enabled"])
        self.assertEqual(status["clamp_feedback_address"], 2)
        self.assertEqual(w.relay_clamp, 7)
        self.assertEqual(w.relay_ok_light_buzzer, 6)
        self.assertEqual(w.relay_enji_buzzer, 5)
        # poll-loop mirrors
        self.assertEqual(w._input_release_address, 4)  # noqa: SLF001
        self.assertEqual(w._input_template_address, 3)  # noqa: SLF001
        self.assertEqual(w._accept_pulse_ms, 321)  # noqa: SLF001

    def test_apply_sets_guards_from_sticker_section(self) -> None:
        w = self._worker()
        w.apply_machine_settings(_settings_for_worker())
        self.assertEqual(w._min_reclamp_interval_ms, 1234)  # noqa: SLF001
        self.assertEqual(w._release_input_debounce_ms, 56)  # noqa: SLF001

    def test_apply_never_flips_dry_run(self) -> None:
        w = self._worker()
        self.assertTrue(w.status()["dry_run"])
        w.apply_machine_settings(_settings_for_worker())
        self.assertTrue(w.status()["dry_run"], "PUT /machine-settings path must keep boot-time dry_run")

    def test_configure_guards_without_dry_run_keeps_current(self) -> None:
        w = self._worker()
        w.configure_guards(min_reclamp_interval_ms=1)
        self.assertTrue(w.status()["dry_run"])
        w.configure_guards(min_reclamp_interval_ms=1, dry_run=False)
        self.assertFalse(w.status()["dry_run"])

    def test_clamp_engaged_reads_feedback_address_from_settings(self) -> None:
        # With feedback enabled at address 2, clamp_engaged() must follow the
        # input snapshot at that address, not the constructor's address 9.
        w = self._worker()
        w.apply_machine_settings(_settings_for_worker())
        w._last_input_snapshot = [False, False, True, False, False, False, False, False]  # noqa: SLF001
        self.assertTrue(w.clamp_engaged())
        w._last_input_snapshot = [False] * 8  # noqa: SLF001
        self.assertFalse(w.clamp_engaged())


class SessionServiceApplyMachineSettingsTest(unittest.TestCase):
    def test_apply_overrides_env_derived_timing_and_clamp_feedback(self) -> None:
        service = InspectionSessionService(
            template_runtime=mock.MagicMock(),
            profiles_repo=mock.MagicMock(),
            results_repo=mock.MagicMock(),
            sticker_inference=mock.MagicMock(),
        )
        # Defaults (what env/AppConfig would have produced)
        self.assertEqual(service._commit_grace_ms, 1500)  # noqa: SLF001
        self.assertFalse(service._plc_clamp_feedback_enabled)  # noqa: SLF001

        s = MachineSettings()
        s.timing.commit_grace_ms = 4321
        s.timing.accept_stable_frames = 7
        s.timing.reject_timeout_ms = 60000
        s.timing.phase_next_part_delay_ms = 6000
        s.sticker.clamp_feedback_enabled = True
        s.sticker.clamp_feedback_timeout_ms = 777
        s.sticker.clamp_feedback_fallback_delay_ms = 55

        service.apply_machine_settings(s)

        self.assertEqual(service._commit_grace_ms, 4321)  # noqa: SLF001
        self.assertEqual(service._accept_stable_frames, 7)  # noqa: SLF001
        self.assertEqual(service._reject_timeout_ms, 60000)  # noqa: SLF001
        self.assertEqual(service._phase_next_part_delay_ms, 6000)  # noqa: SLF001
        self.assertTrue(service._plc_clamp_feedback_enabled)  # noqa: SLF001
        self.assertEqual(service._plc_clamp_feedback_timeout_ms, 777)  # noqa: SLF001
        self.assertEqual(service._plc_clamp_feedback_fallback_delay_ms, 55)  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
