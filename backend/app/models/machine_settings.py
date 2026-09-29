"""Model Machine Settings — config PLC, timing, dan inference per mesin.

Disimpan di `data/json_store/machine_settings.json` dan diedit dari tab Admin →
Machine Settings. File ini SATU-SATUNYA sumber kebenaran untuk nilai-nilai
ini: tidak ada lagi yang dibaca dari `.env`. File/key yang tidak ada artinya
"pakai default dataclass" (PC baru).

Bagian
  connection  transport PLC (butuh restart backend supaya berlaku)
  io          alamat relay / input + timing pulse & guard PLC (diterapkan langsung)
  timing      timer inspeksi / commit policy (diterapkan langsung)
  inference   model sticker + setting device / thread (butuh restart)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SETTINGS_VERSION = 2


@dataclass(slots=True)
class PlcConnectionConfig:
    enabled: bool = False
    dry_run: bool = True
    transport: str = "tcp"  # "tcp" | "rtu" | "fx"
    host: str = "127.0.0.1"
    port: int = 5020
    serial_port: str = ""
    serial_baudrate: int = 9600
    serial_parity: str = "N"
    serial_bytesize: int = 8
    serial_stopbits: int = 1
    timeout_ms: int = 1000
    modbus_unit_id: int = 255


@dataclass(slots=True)
class PlcIoConfig:
    """Peta relay / input + timing sisi-PLC untuk alur sticker."""
    # Alamat coil relay (CH1=Enji Buzzer, CH2=OK Light+Buzzer, CH3=Clamp)
    relay_clamp_address: int = 3
    relay_ok_light_buzzer_address: int = 2
    relay_enji_buzzer_address: int = 1
    # Alamat input
    input_release_address: int = 0
    input_template_address: int = 1
    input_clamp_engaged_address: int = 2
    clamp_feedback_enabled: bool = False
    # Timing / guard
    accept_pulse_ms: int = 1000
    min_reclamp_interval_ms: int = 3000
    release_input_debounce_ms: int = 200


@dataclass(slots=True)
class TimingConfig:
    """Timer inspeksi / phase operator / setting commit-policy."""
    # Pacing phase operator (gate non-blocking)
    phase_next_part_delay_ms: int = 2000
    phase_sticker_install_delay_ms: int = 0
    # Threshold stabilitas sebelum commit ACCEPT
    accept_stable_frames: int = 1
    accept_stable_ms: int = 200
    # Guard commit
    commit_grace_ms: int = 1500
    reject_timeout_ms: int = 15000
    # Part ready
    part_ready_release_ms: int = 300
    part_ready_settle_ms_default: int = 0
    # Cache / holdover
    inference_cache_grace_ms: int = 300
    accept_holdover_ms: int = 2000
    inference_cache_ttl_ms: int = 10000
    # Keamanan / sesi
    session_idle_timeout_s: int = 300
    max_consecutive_rejects: int = 0


@dataclass(slots=True)
class IdentitySettings:
    line: str = ""  # reserved — nanti jadi basis tabel terpisah


@dataclass(slots=True)
class InferenceConfig:
    """Runtime model sticker — setting hardware per-PC."""
    mode: str = "auto"      # auto | ultralytics | onnx | openvino | tflite | classic
    device: str = "auto"    # auto | cpu | cuda
    cuda_device_id: int = 0
    num_threads: int = 4
    interval_ms: int = 0    # ms minimum antar-run inference; 0 = tiap frame
    timeout_s: float = 5.0
    # Model fallback kalau template aktif tidak punya vision.model_path
    default_model_path: str = ""
    default_model_meta_path: str = ""


def _section(cls, data: dict[str, Any] | None):
    """Bangun dataclass bagian dari dict, buang key yang tidak dikenal."""
    raw = data or {}
    return cls(**{k: v for k, v in raw.items() if k in cls.__slots__})


@dataclass(slots=True)
class MachineSettings:
    """Machine settings tingkat atas — satu record per mesin."""
    version: int = SETTINGS_VERSION
    connection: PlcConnectionConfig = field(default_factory=PlcConnectionConfig)
    io: PlcIoConfig = field(default_factory=PlcIoConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    identity: IdentitySettings = field(default_factory=IdentitySettings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SETTINGS_VERSION,
            "connection": asdict(self.connection),
            "io": asdict(self.io),
            "timing": asdict(self.timing),
            "inference": asdict(self.inference),
            "identity": asdict(self.identity),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MachineSettings:
        # File v1 menyimpan peta I/O di bawah "sticker" (dan "counter" yang tidak pernah dipakai).
        io_raw = data.get("io")
        if io_raw is None:
            io_raw = data.get("sticker")
        return cls(
            version=SETTINGS_VERSION,
            connection=_section(PlcConnectionConfig, data.get("connection")),
            io=_section(PlcIoConfig, io_raw),
            timing=_section(TimingConfig, data.get("timing")),
            inference=_section(InferenceConfig, data.get("inference")),
            identity=_section(IdentitySettings, data.get("identity")),
        )
