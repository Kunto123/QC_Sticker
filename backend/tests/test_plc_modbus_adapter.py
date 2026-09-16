"""Tests for the CURRENT minimal PLC adapter + worker input-polling.

FASE 0 rewrite: the old clamp/readback/command-mode adapter API was
INTENTIONALLY replaced (orchestrator-adjudicated) with a lean
write_coil / read_inputs / slave_id design based on the proven testall.py flow.
The obsolete old-API tests were retired; this file covers the NEW surface that
actually exists in backend/app/services/plc_adapter.py today so PLC coverage does
not drop to zero.

Covered public surface:
- DryRunPlcAdapter: connect/disconnect/write_coil/write_coils/read_inputs/all_off/status
- ModbusTcpPlcAdapter: constructor wiring, lazy connect on write, write_coil ->
  client.write_coil(addr, value, device_id=slave_id), read_inputs -> FC02, error raise
- ModbusRtuPlcAdapter: constructor wiring + write path
- build_plc_adapter: dry-run / tcp / rtu / unknown selection
- PlcWorker._poll_inputs (new model): IN1 manual-release requires stable debounce
  before all-off; IN2 template-cycle increments once per debounce window.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.plc_adapter import (
    DryRunPlcAdapter,
    ModbusRtuPlcAdapter,
    ModbusTcpPlcAdapter,
    build_plc_adapter,
)
from backend.app.workers.plc_worker import PlcWorker


def _make_mock_client(*, input_bits=None) -> mock.MagicMock:
    """A MagicMock shaped like a pymodbus client instance for the CURRENT adapter."""
    client = mock.MagicMock()
    client.connected = False

    def _connect() -> bool:
        client.connected = True
        return True

    client.connect.side_effect = _connect

    ok_resp = mock.MagicMock()
    ok_resp.isError.return_value = False
    client.write_coil.return_value = ok_resp

    di_resp = mock.MagicMock()
    di_resp.isError.return_value = False
    di_resp.bits = list(input_bits) if input_bits is not None else [False] * 8
    client.read_discrete_inputs.return_value = di_resp
    return client


def _make_config(**overrides) -> SimpleNamespace:
    defaults = dict(
        plc_dry_run=False,
        plc_transport="tcp",
        plc_timeout_ms=750,
        plc_modbus_unit_id=7,
        plc_host="10.0.0.5",
        plc_port=502,
        plc_serial_port="COM3",
        plc_serial_baudrate=9600,
        plc_serial_parity="N",
        plc_serial_bytesize=8,
        plc_serial_stopbits=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class DryRunPlcAdapterTest(unittest.TestCase):
    def test_lifecycle_and_status(self) -> None:
        a = DryRunPlcAdapter()
        a.connect()
        self.assertTrue(a.is_connected())
        # write paths must not raise
        a.write_coil(0, True)
        a.write_coils({1: True, 2: False})
        a.all_off(4)
        st = a.status()
        self.assertEqual(st["adapter"], "DryRunPlcAdapter")
        self.assertTrue(st["connected"])
        a.disconnect()

    def test_read_inputs_returns_all_false(self) -> None:
        a = DryRunPlcAdapter()
        self.assertEqual(a.read_inputs(count=8), [False] * 8)


class ModbusTcpPlcAdapterTest(unittest.TestCase):
    def _make(self, mc, **kw):
        with mock.patch("backend.app.services.plc_adapter.ModbusTcpClient", return_value=mc):
            return ModbusTcpPlcAdapter("10.0.0.5", 502, **kw)

    def test_constructor_wires_client_with_host_port_timeout(self) -> None:
        mc = _make_mock_client()
        with mock.patch("backend.app.services.plc_adapter.ModbusTcpClient", return_value=mc) as Cls:
            ModbusTcpPlcAdapter("10.0.0.5", 502, slave_id=7, timeout=0.75)
        Cls.assert_called_once_with(host="10.0.0.5", port=502, timeout=0.75)

    def test_status_reports_adapter_name(self) -> None:
        mc = _make_mock_client()
        adapter = self._make(mc, slave_id=1)
        self.assertEqual(adapter.status()["adapter"], "ModbusTcpPlcAdapter")

    def test_write_coil_lazy_connects_and_uses_device_id(self) -> None:
        mc = _make_mock_client()
        adapter = self._make(mc, slave_id=7)
        self.assertFalse(adapter.is_connected())
        adapter.write_coil(12, True)
        self.assertTrue(adapter.is_connected())
        mc.connect.assert_called_once()
        mc.write_coil.assert_called_once_with(12, True, device_id=7)

    def test_write_coil_raises_on_modbus_error(self) -> None:
        mc = _make_mock_client()
        err = mock.MagicMock()
        err.isError.return_value = True
        mc.write_coil.return_value = err
        adapter = self._make(mc, slave_id=1)
        with self.assertRaises(RuntimeError):
            adapter.write_coil(0, True)

    def test_read_inputs_uses_fc02_and_slave_id(self) -> None:
        mc = _make_mock_client(input_bits=[True, False, True, False, False, False, False, False])
        adapter = self._make(mc, slave_id=9)
        result = adapter.read_inputs(address=0, count=8)
        mc.read_discrete_inputs.assert_called_once_with(0, count=8, device_id=9)
        self.assertEqual(result, [True, False, True, False, False, False, False, False])

    def test_write_coils_batch(self) -> None:
        mc = _make_mock_client()
        adapter = self._make(mc, slave_id=3)
        adapter.write_coils({0: True, 1: False})
        self.assertEqual(mc.write_coil.call_count, 2)
        mc.write_coil.assert_any_call(0, True, device_id=3)
        mc.write_coil.assert_any_call(1, False, device_id=3)

    def test_disconnect_closes_client(self) -> None:
        mc = _make_mock_client()
        adapter = self._make(mc)
        adapter.disconnect()
        mc.close.assert_called_once()


class ModbusRtuPlcAdapterTest(unittest.TestCase):
    def test_constructor_wires_serial_client(self) -> None:
        mc = _make_mock_client()
        with mock.patch("backend.app.services.plc_adapter.ModbusSerialClient", return_value=mc) as Cls:
            ModbusRtuPlcAdapter("COM3", baudrate=9600, slave_id=255, timeout=0.75)
        Cls.assert_called_once_with(
            port="COM3", baudrate=9600, parity="N", stopbits=1, bytesize=8, timeout=0.75
        )

    def test_write_coil_uses_device_id(self) -> None:
        mc = _make_mock_client()
        with mock.patch("backend.app.services.plc_adapter.ModbusSerialClient", return_value=mc):
            adapter = ModbusRtuPlcAdapter("COM3", slave_id=255)
        adapter.write_coil(0, True)
        mc.write_coil.assert_called_once_with(0, True, device_id=255)


class BuildPlcAdapterTest(unittest.TestCase):
    def test_dry_run_selected_when_flag_set(self) -> None:
        adapter = build_plc_adapter(_make_config(plc_dry_run=True))
        self.assertIsInstance(adapter, DryRunPlcAdapter)

    def test_tcp_selected(self) -> None:
        mc = _make_mock_client()
        with mock.patch("backend.app.services.plc_adapter.ModbusTcpClient", return_value=mc) as Cls:
            adapter = build_plc_adapter(_make_config(plc_transport="tcp"))
        self.assertIsInstance(adapter, ModbusTcpPlcAdapter)
        Cls.assert_called_once_with(host="10.0.0.5", port=502, timeout=0.75)

    def test_rtu_selected(self) -> None:
        mc = _make_mock_client()
        with mock.patch("backend.app.services.plc_adapter.ModbusSerialClient", return_value=mc):
            adapter = build_plc_adapter(_make_config(plc_transport="rtu"))
        self.assertIsInstance(adapter, ModbusRtuPlcAdapter)

    def test_unknown_transport_falls_back_to_dry_run(self) -> None:
        # NEW design: unknown transport degrades to DryRun (does NOT raise). This is
        # the intended behavior of the minimal factory.
        adapter = build_plc_adapter(_make_config(plc_transport="banana"))
        self.assertIsInstance(adapter, DryRunPlcAdapter)


class PlcWorkerInputPollingTest(unittest.TestCase):
    """New accept-pulse + input-polling model (was: clamp hold/release)."""

    class _FakeAdapter:
        """Records write_coil calls; read_inputs returns a fixed snapshot."""

        def __init__(self, inputs) -> None:
            self._inputs = list(inputs)
            self.coil_writes: list[tuple[int, bool]] = []

        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

        def is_connected(self) -> bool:
            return True

        def read_inputs(self, address: int = 0, count: int = 8):
            return list(self._inputs)

        def write_coil(self, address: int, value: bool) -> None:
            self.coil_writes.append((address, bool(value)))

        def all_off(self, num_channels: int = 4) -> None:
            for i in range(num_channels):
                self.coil_writes.append((i, False))

        def status(self) -> dict:
            return {}

    def _worker(self, adapter) -> PlcWorker:
        # IN1=release@0, IN2=template@1 (constructor defaults). Fast debounce.
        w = PlcWorker(adapter, num_channels=4)
        w.configure_guards(release_input_debounce_ms=0, dry_run=True)
        return w

    def test_input1_manual_release_triggers_all_off_after_stable_debounce(self) -> None:
        # IN1 HIGH with debounce_ms=0 -> stable immediately -> _all_off writes each
        # channel OFF via write_coil (new model uses write_coil, not adapter.all_off).
        adapter = self._FakeAdapter([True, False, False, False, False, False, False, False])
        worker = self._worker(adapter)
        worker._poll_inputs()  # noqa: SLF001
        off_writes = [c for c in adapter.coil_writes if c[1] is False]
        self.assertEqual(len(off_writes), worker.num_channels)
        self.assertEqual(worker.status()["state"], "IDLE")

    def test_input1_high_but_not_yet_stable_does_not_release(self) -> None:
        # With a long debounce, a single poll of IN1 HIGH must NOT release yet.
        adapter = self._FakeAdapter([True, False, False, False, False, False, False, False])
        worker = PlcWorker(adapter, num_channels=4)
        worker.configure_guards(release_input_debounce_ms=10_000, dry_run=True)
        worker._poll_inputs()  # noqa: SLF001
        self.assertEqual(adapter.coil_writes, [], "must not release before debounce elapses")

    def test_input2_increments_template_cycle_once_per_debounce(self) -> None:
        adapter = self._FakeAdapter([False, True, False, False, False, False, False, False])
        worker = self._worker(adapter)
        worker._poll_inputs()  # noqa: SLF001
        self.assertEqual(worker.status()["template_cycle_event_id"], 1)
        # Second immediate poll is within debounce -> no increment
        worker._poll_inputs()  # noqa: SLF001
        self.assertEqual(worker.status()["template_cycle_event_id"], 1)

    def test_notify_decision_enqueues_one_command(self) -> None:
        adapter = self._FakeAdapter([False] * 8)
        worker = self._worker(adapter)
        self.assertEqual(worker.status()["cmd_queue_depth"], 0)
        worker.notify_decision("ACCEPT", event_id="evt-1")
        self.assertEqual(worker.status()["cmd_queue_depth"], 1)


if __name__ == "__main__":
    unittest.main()
