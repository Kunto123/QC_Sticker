"""
PLC Modbus adapter — simple, berdasarkan testall.py yang sudah terbukti bekerja.

Referensi: D:/pythonmodbus/testall.py
- Slave ID: 255
- Timeout: 0.5s
- Write delay: 50ms setelah setiap write
- Input read: count=8 (sesuai firmware)
- Simple connect di awal, tidak ada retry loop kompleks
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod

try:
    from pymodbus.client import ModbusSerialClient, ModbusTcpClient
except ModuleNotFoundError:
    ModbusSerialClient = None
    ModbusTcpClient = None

try:
    from fxplc.client.FXPLCClient import FXPLCClient
    from fxplc.transports.TransportSerial import TransportSerial
except ImportError:
    # Catch ImportError (not just ModuleNotFoundError) so a partial/namespace
    # install or a missing fxplc dependency (e.g. pyserial) degrades gracefully
    # instead of crashing app startup. The FX adapter raises a clear error on use.
    FXPLCClient = None  # type: ignore[assignment]
    TransportSerial = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class PlcAdapter(ABC):
    @abstractmethod
    def connect(self) -> None: ...
    @abstractmethod
    def disconnect(self) -> None: ...
    @abstractmethod
    def is_connected(self) -> bool: ...
    @abstractmethod
    def write_coil(self, address: int, value: bool) -> None: ...
    @abstractmethod
    def write_coils(self, coils: dict[int, bool]) -> None: ...
    @abstractmethod
    def read_inputs(self, address: int = 0, count: int = 8) -> list[bool]: ...

    def all_off(self, num_channels: int = 4) -> None:
        for i in range(num_channels):
            self.write_coil(i, False)
            time.sleep(0.05)
    def status(self) -> dict:
        return {"adapter": type(self).__name__, "connected": self.is_connected()}


class DryRunPlcAdapter(PlcAdapter):
    def connect(self) -> None:
        logger.info("[plc-dry-run] connect")

    def disconnect(self) -> None:
        logger.info("[plc-dry-run] disconnect")

    def is_connected(self) -> bool:
        return True

    def write_coil(self, address: int, value: bool) -> None:
        logger.info("[plc-dry-run] write_coil addr=%d value=%s", address, value)

    def write_coils(self, coils: dict[int, bool]) -> None:
        for addr, val in coils.items():
            self.write_coil(addr, val)

    def read_inputs(self, address: int = 0, count: int = 8) -> list[bool]:
        return [False] * count


class ModbusRtuPlcAdapter(PlcAdapter):
    def __init__(
        self,
        port: str = "COM7",
        baudrate: int = 9600,
        slave_id: int = 255,
        timeout: float = 0.5,
        parity: str = "N",
        bytesize: int = 8,
        stopbits: int = 1,
    ):
        if ModbusSerialClient is None:
            raise RuntimeError("pymodbus is not installed")
        self._slave_id = slave_id
        self._client = ModbusSerialClient(
            port=port,
            baudrate=baudrate,
            parity=parity,
            stopbits=stopbits,
            bytesize=bytesize,
            timeout=timeout,
        )
        self._port = port

    def connect(self) -> None:
        if self._client.connected:
            return
        if not self._client.connect():
            raise RuntimeError(f"failed to connect to modbus RTU on {self._port}")
        logger.info("[plc-modbus-rtu] connected to %s (slave=%d)", self._port, self._slave_id)

    def disconnect(self) -> None:
        self._client.close()
        logger.info("[plc-modbus-rtu] disconnected from %s", self._port)

    def is_connected(self) -> bool:
        return self._client.connected

    def _ensure_connected(self) -> None:
        if self._client.connected:
            return
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                if self._client.connect():
                    logger.info("[plc-modbus-rtu] reconnected to %s (attempt %d)", self._port, attempt)
                    return
            except Exception as exc:
                logger.warning("[plc-modbus-rtu] reconnect attempt %d failed: %s", attempt, exc)
            if attempt < max_retries:
                time.sleep(0.5 * attempt)
        raise RuntimeError(f"modbus reconnect failed on {self._port} after {max_retries} attempts")

    def write_coil(self, address: int, value: bool) -> None:
        self._ensure_connected()
        # Drain stale bytes from serial buffer before sending new command.
        # Prevents recv-buffer spam when polling at 100ms and response hasn't
        # fully arrived before the next write.
        try:
            if hasattr(self._client, 'socket') and self._client.socket:
                self._client.socket.reset_input_buffer()
        except Exception:
            pass
        resp = self._client.write_coil(address, bool(value), device_id=self._slave_id)
        if resp.isError():
            raise RuntimeError(f"write_coil error addr={address}: {resp}")
        time.sleep(0.05)  # delay 50ms seperti testall.py

    def write_coils(self, coils: dict[int, bool]) -> None:
        for addr, val in coils.items():
            self.write_coil(addr, val)

    def read_inputs(self, address: int = 0, count: int = 8) -> list[bool]:
        self._ensure_connected()
        try:
            resp = self._client.read_discrete_inputs(address, count=count, device_id=self._slave_id)
            if resp.isError():
                raise RuntimeError(f"[plc-modbus-rtu] read_inputs error: {resp}")
        except Exception:
            # Item 1: close client so worker's connect() actually reconnects
            try:
                self._client.close()
            except Exception:
                pass
            raise
        return [bool(resp.bits[i]) for i in range(count)]


class ModbusTcpPlcAdapter(PlcAdapter):
    """Modbus TCP simple — referensi testall.py"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 502,
        slave_id: int = 255,
        timeout: float = 0.5,
    ):
        if ModbusTcpClient is None:
            raise RuntimeError("pymodbus is not installed")
        self._slave_id = slave_id
        self._client = ModbusTcpClient(host=host, port=port, timeout=timeout)

    def connect(self) -> None:
        if self._client.connected:
            return
        if not self._client.connect():
            raise RuntimeError(f"failed to connect to modbus TCP")
        logger.info("[plc-modbus-tcp] connected (slave=%d)", self._slave_id)

    def disconnect(self) -> None:
        self._client.close()
        logger.info("[plc-modbus-tcp] disconnected")

    def is_connected(self) -> bool:
        return self._client.connected

    def _ensure_connected(self) -> None:
        if self._client.connected:
            return
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                if self._client.connect():
                    logger.info("[plc-modbus-tcp] reconnected (attempt %d)", attempt)
                    return
            except Exception as exc:
                logger.warning("[plc-modbus-tcp] reconnect attempt %d failed: %s", attempt, exc)
            if attempt < max_retries:
                time.sleep(0.5 * attempt)
        raise RuntimeError("modbus TCP reconnect failed after 3 attempts")

    def write_coil(self, address: int, value: bool) -> None:
        self._ensure_connected()
        resp = self._client.write_coil(address, bool(value), device_id=self._slave_id)
        if resp.isError():
            raise RuntimeError(f"write_coil error addr={address}: {resp}")
        time.sleep(0.05)

    def write_coils(self, coils: dict[int, bool]) -> None:
        for addr, val in coils.items():
            self.write_coil(addr, val)

    def read_inputs(self, address: int = 0, count: int = 8) -> list[bool]:
        self._ensure_connected()
        try:
            resp = self._client.read_discrete_inputs(address, count=count, device_id=self._slave_id)
            if resp.isError():
                raise RuntimeError(f"[plc-modbus-tcp] read_inputs error: {resp}")
        except Exception:
            # Item 1: close client so worker's connect() actually reconnects
            try:
                self._client.close()
            except Exception:
                pass
            raise
        return [bool(resp.bits[i]) for i in range(count)]


class FXComputerLinkPlcAdapter(PlcAdapter):
    """FX Computer Link (fxplc) adapter — bridge async fxplc to sync PlcAdapter.

    Uses a dedicated event loop on its own thread + persistent connection.
    All calls are serialized through the single loop thread, matching the
    single-threaded PLC worker: no race conditions.
    """

    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 38400,
        timeout: float = 2.0,
    ):
        if FXPLCClient is None or TransportSerial is None:
            raise RuntimeError(
                "fxplc is not installed. Install via: "
                "pip install git+https://github.com/KrystianD/fxplc.git"
            )
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._client: FXPLCClient | None = None
        self._transport = None  # TransportSerial — kept so we can close the serial port
        self._connected: bool = False
        self._last_known_inputs: list[bool] = []  # hold last-good for sub-threshold blips
        self._lock = threading.Lock()

    def connect(self) -> None:
        """Start the event loop thread and open serial connection.

        Safe to call again after a failed connect (e.g. wrong/unplugged COM port):
        the event loop is reused instead of leaking a new thread each retry.
        """
        with self._lock:
            if self._connected and self._client is not None:
                return
            # Reuse a still-running loop; only start one if needed.
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                self._loop_thread = threading.Thread(
                    target=self._loop.run_forever,
                    name="qc-fxplc-loop",
                    daemon=True,
                )
                self._loop_thread.start()
            try:
                # TransportSerial + FXPLCClient are SYNC constructors (pyserial opens
                # the port in __init__). They must NOT go through _run_sync (which
                # expects a coroutine). The event loop is only used for the async
                # read_bit/write_bit calls later.
                self._transport = TransportSerial(
                    self._port, baudrate=self._baudrate, timeout=self._timeout
                )
                self._client = FXPLCClient(self._transport)
            except Exception:
                # Close any half-opened serial handle so the port is freed and the
                # next connect() doesn't hit "Access is denied" on its own leak.
                self._close_transport_quietly()
                self._client = None
                self._connected = False
                raise
            self._connected = True
            logger.info(
                "[plc-fx] connected (port=%s, baudrate=%d, timeout=%.1fs)",
                self._port,
                self._baudrate,
                self._timeout,
            )

    def disconnect(self) -> None:
        """Close serial connection and stop the event loop thread."""
        with self._lock:
            self._client = None
            self._close_transport_quietly()
            self._connected = False
            self._last_known_inputs = []  # clear on disconnect
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=3.0)
                self._loop_thread = None
            if self._loop is not None:
                self._loop.close()
                self._loop = None
            logger.info("[plc-fx] disconnected from %s", self._port)

    def is_connected(self) -> bool:
        return self._connected and self._client is not None

    def _ensure_connected(self) -> None:
        """Lazily (re)connect on demand. Raises the REAL underlying serial error
        (port busy / access denied / not found) instead of a generic 'not connected',
        so failures surfaced via test-coil / writes are actionable."""
        if self._client is not None:
            return
        self.connect()  # may raise the underlying serial error
        if self._client is None:
            raise RuntimeError(f"[plc-fx] not connected (port={self._port})")

    def write_coil(self, address: int, value: bool) -> None:
        self._ensure_connected()
        fx_label = f"Y{format(address, 'o')}"
        try:
            self._run_sync(self._client.write_bit(fx_label, bool(value)))
        except Exception:
            # Item 1: reset connection state so worker's connect() actually reopens
            self._connected = False
            self._client = None
            self._close_transport_quietly()
            raise
        time.sleep(0.05)

    def write_coils(self, coils: dict[int, bool]) -> None:
        for addr, val in coils.items():
            self.write_coil(addr, val)

    def read_inputs(self, address: int = 0, count: int = 8) -> list[bool]:
        self._ensure_connected()
        result: list[bool] = []
        consecutive_fail = 0
        max_bit_failures = 3  # raise after K consecutive bit read failures
        for i in range(address, address + count):
            fx_label = f"X{format(i, 'o')}"
            try:
                bit = self._run_sync(self._client.read_bit(fx_label))
                result.append(bool(bit))
                consecutive_fail = 0  # reset on success
            except Exception as exc:
                consecutive_fail += 1
                logger.warning(
                    "[plc-fx] read_bit %s failed (%d/%d): %r",
                    fx_label, consecutive_fail, max_bit_failures, exc,
                )
                # If the very first read fails and we reset, don't bother
                # reading the rest — likely a device-level failure.
                if consecutive_fail == 1 and i == address:
                    # Immediately stop trying more bits — the transport is dead
                    consecutive_fail = max_bit_failures
                if consecutive_fail >= max_bit_failures:
                    # Item 1: reset connection state so worker's connect() actually reopens
                    self._connected = False
                    self._client = None
                    self._close_transport_quietly()
                    raise RuntimeError(
                        f"[plc-fx] {consecutive_fail} consecutive read failures — "
                        f"serial link may be degraded"
                    ) from exc
                # Sub-threshold: hold last-known-good instead of False
                if i < len(self._last_known_inputs):
                    result.append(self._last_known_inputs[i])
                else:
                    result.append(False)
        # Update last-known-good on successful read
        self._last_known_inputs = list(result)
        return result

    def all_off(self, num_channels: int = 4) -> None:
        """Turn off Y0..Ynum_channels-1."""
        for i in range(num_channels):
            try:
                self.write_coil(i, False)
            except Exception as exc:
                logger.error("[plc-fx] all_off coil %d failed: %s", i, exc)
        logger.info("[plc-fx] all_off done (%d channels)", num_channels)

    def status(self) -> dict:
        return {
            "adapter": type(self).__name__,
            "connected": self.is_connected(),
            "transport": "fx",
            "port": self._port,
        }

    # ── Internal helpers ──────────────────────────────────────────────

    def _close_transport_quietly(self) -> None:
        """Close the serial transport if open, swallowing errors. Releases COM port."""
        t = self._transport
        self._transport = None
        if t is None:
            return
        for closer in ("close", "disconnect"):
            fn = getattr(t, closer, None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    pass
        # Fallback: close the underlying pyserial object if exposed.
        for attr in ("_serial", "serial", "ser"):
            s = getattr(t, attr, None)
            if s is not None and hasattr(s, "close"):
                try:
                    s.close()
                except Exception:
                    pass

    def _run_sync(self, coro):
        """Submit a coroutine to the dedicated loop and block until done."""
        if self._loop is None or self._loop.is_closed():
            raise RuntimeError("[plc-fx] event loop is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=max(self._timeout, 5.0))


def build_plc_adapter(config) -> PlcAdapter:
    """Factory: pilih adapter berdasarkan config."""
    if config.plc_dry_run:
        return DryRunPlcAdapter()

    if config.plc_transport == "fx":
        return FXComputerLinkPlcAdapter(
            port=config.plc_serial_port or "COM3",
            baudrate=config.plc_serial_baudrate,
            timeout=config.plc_timeout_ms / 1000.0,
        )
    if config.plc_transport == "rtu":
        return ModbusRtuPlcAdapter(
            port=config.plc_serial_port or "COM7",
            baudrate=config.plc_serial_baudrate,
            slave_id=config.plc_modbus_unit_id,
            timeout=config.plc_timeout_ms / 1000.0,
            parity=config.plc_serial_parity,
            bytesize=config.plc_serial_bytesize,
            stopbits=config.plc_serial_stopbits,
        )
    elif config.plc_transport == "tcp":
        return ModbusTcpPlcAdapter(
            host=config.plc_host or "127.0.0.1",
            port=config.plc_port or 502,
            slave_id=config.plc_modbus_unit_id,
            timeout=config.plc_timeout_ms / 1000.0,
        )
    else:
        return DryRunPlcAdapter()
