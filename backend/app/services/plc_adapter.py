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
    # Tangkap ImportError (bukan cuma ModuleNotFoundError) supaya install
    # partial/namespace atau dependency fxplc yang hilang (mis. pyserial)
    # degradasi dengan aman, tidak crash saat app startup. Adapter FX
    # melempar error yang jelas saat dipakai.
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
        # Kuras byte lama dari buffer serial sebelum kirim command baru.
        # Mencegah spam recv-buffer saat polling di 100ms dan respons belum
        # sepenuhnya tiba sebelum write berikutnya.
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
            # Item 1: tutup client supaya connect() worker benar-benar reconnect
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
            # Item 1: tutup client supaya connect() worker benar-benar reconnect
            try:
                self._client.close()
            except Exception:
                pass
            raise
        return [bool(resp.bits[i]) for i in range(count)]


class FXComputerLinkPlcAdapter(PlcAdapter):
    """Adapter FX Computer Link (fxplc) — jembatan fxplc async ke PlcAdapter sync.

    Pakai event loop khusus di thread sendiri + koneksi persisten. Semua
    pemanggilan diserialisasi lewat satu thread loop, sesuai dengan PLC
    worker yang single-threaded: tidak ada race condition.
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
        self._transport = None  # TransportSerial — disimpan supaya bisa tutup serial port
        self._connected: bool = False
        self._last_known_inputs: list[bool] = []  # simpan last-good untuk blip sub-threshold
        self._lock = threading.Lock()

    def connect(self) -> None:
        """Mulai thread event loop dan buka koneksi serial.

        Aman dipanggil lagi setelah connect gagal (mis. COM port salah/tercabut):
        event loop dipakai ulang, bukan bocor thread baru tiap retry.
        """
        with self._lock:
            if self._connected and self._client is not None:
                return
            # Pakai ulang loop yang masih jalan; cuma start satu kalau perlu.
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                self._loop_thread = threading.Thread(
                    target=self._loop.run_forever,
                    name="qc-fxplc-loop",
                    daemon=True,
                )
                self._loop_thread.start()
            try:
                # TransportSerial + FXPLCClient adalah konstruktor SYNC (pyserial
                # buka port di __init__). Keduanya TIDAK BOLEH lewat _run_sync
                # (yang mengharapkan coroutine). Event loop cuma dipakai untuk
                # pemanggilan async read_bit/write_bit belakangan.
                self._transport = TransportSerial(
                    self._port, baudrate=self._baudrate, timeout=self._timeout
                )
                self._client = FXPLCClient(self._transport)
            except Exception:
                # Tutup handle serial yang setengah-terbuka supaya port bebas dan
                # connect() berikutnya tidak kena "Access is denied" akibat bocor sendiri.
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
        """Tutup koneksi serial dan hentikan thread event loop."""
        with self._lock:
            self._client = None
            self._close_transport_quietly()
            self._connected = False
            self._last_known_inputs = []  # kosongkan saat disconnect
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
        """(Re)connect lazy sesuai kebutuhan. Melempar error serial ASLI di bawahnya
        (port sibuk / akses ditolak / tidak ketemu) daripada 'not connected' generik,
        supaya kegagalan yang muncul lewat test-coil / write bisa ditindaklanjuti."""
        if self._client is not None:
            return
        self.connect()  # bisa melempar error serial di bawahnya
        if self._client is None:
            raise RuntimeError(f"[plc-fx] tidak terhubung (port={self._port})")

    def write_coil(self, address: int, value: bool) -> None:
        self._ensure_connected()
        fx_label = f"Y{format(address, 'o')}"
        try:
            self._run_sync(self._client.write_bit(fx_label, bool(value)))
        except Exception:
            # Item 1: reset connection state supaya connect() worker benar-benar buka ulang
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
        max_bit_failures = 3  # lempar error setelah K kegagalan read bit berturutan
        for i in range(address, address + count):
            fx_label = f"X{format(i, 'o')}"
            try:
                bit = self._run_sync(self._client.read_bit(fx_label))
                result.append(bool(bit))
                consecutive_fail = 0  # reset kalau berhasil
            except Exception as exc:
                consecutive_fail += 1
                logger.warning(
                    "[plc-fx] read_bit %s failed (%d/%d): %r",
                    fx_label, consecutive_fail, max_bit_failures, exc,
                )
                # Kalau read pertama langsung gagal, tidak usah lanjut baca
                # yang lain — kemungkinan kegagalan di level device.
                if consecutive_fail == 1 and i == address:
                    # Langsung berhenti coba bit lain — transport-nya mati
                    consecutive_fail = max_bit_failures
                if consecutive_fail >= max_bit_failures:
                    # Item 1: reset connection state supaya connect() worker benar-benar buka ulang
                    self._connected = False
                    self._client = None
                    self._close_transport_quietly()
                    raise RuntimeError(
                        f"[plc-fx] {consecutive_fail} kegagalan read berturutan — "
                        f"koneksi serial mungkin bermasalah"
                    ) from exc
                # Sub-threshold: pakai last-known-good, bukan False
                if i < len(self._last_known_inputs):
                    result.append(self._last_known_inputs[i])
                else:
                    result.append(False)
        # Update last-known-good setelah read berhasil
        self._last_known_inputs = list(result)
        return result

    def all_off(self, num_channels: int = 4) -> None:
        """Matikan Y0..Ynum_channels-1."""
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

    # ── Helper internal ──────────────────────────────────────────────

    def _close_transport_quietly(self) -> None:
        """Tutup transport serial kalau masih terbuka, telan errornya. Membebaskan COM port."""
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
        # Fallback: tutup objek pyserial di baliknya kalau terekspos.
        for attr in ("_serial", "serial", "ser"):
            s = getattr(t, attr, None)
            if s is not None and hasattr(s, "close"):
                try:
                    s.close()
                except Exception:
                    pass

    def _run_sync(self, coro):
        """Kirim satu coroutine ke loop khusus dan blok sampai selesai."""
        if self._loop is None or self._loop.is_closed():
            raise RuntimeError("[plc-fx] event loop tidak berjalan")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=max(self._timeout, 5.0))


def build_plc_adapter(conn) -> PlcAdapter:
    """Factory: pilih adapter dari MachineSettings.connection (PlcConnectionConfig).

    dry_run → DryRun; transport "fx" / "rtu" / "tcp"; selain itu degradasi ke DryRun.
    """
    if conn.dry_run:
        return DryRunPlcAdapter()
    transport = str(conn.transport or "").strip().lower()
    timeout_s = max(0.1, float(conn.timeout_ms) / 1000.0)
    if transport == "fx":
        return FXComputerLinkPlcAdapter(
            port=conn.serial_port or "COM3",
            baudrate=conn.serial_baudrate,
            timeout=timeout_s,
        )
    if transport == "rtu":
        return ModbusRtuPlcAdapter(
            port=conn.serial_port or "COM7",
            baudrate=conn.serial_baudrate,
            slave_id=conn.modbus_unit_id,
            timeout=timeout_s,
            parity=conn.serial_parity,
            bytesize=conn.serial_bytesize,
            stopbits=conn.serial_stopbits,
        )
    if transport == "tcp":
        return ModbusTcpPlcAdapter(
            host=conn.host or "127.0.0.1",
            port=conn.port or 502,
            slave_id=conn.modbus_unit_id,
            timeout=timeout_s,
        )
    return DryRunPlcAdapter()
