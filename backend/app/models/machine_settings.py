"""Machine Settings model — per-machine PLC, timing and inference config.

Stored as `data/json_store/machine_settings.json` and edited from the Admin →
Machine Settings tab. This file is the ONLY source of truth for these values:
nothing here is read from `.env` any more. A missing file / missing key means
"use the dataclass default" (fresh PC).

Sections
  connection  PLC transport (needs a backend restart to take effect)
  io          relay / input addresses + PLC pulse & guard timing (live-applied)
  timing      inspection timers / commit policy (live-applied)
  inference   sticker model + device / thread settings (needs a restart)
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
    """Relay / input map + PLC-side timing for the sticker flow."""
    # Relay coil addresses (CH1=Enji Buzzer, CH2=OK Light+Buzzer, CH3=Clamp)
    relay_clamp_address: int = 3
    relay_ok_light_buzzer_address: int = 2
    relay_enji_buzzer_address: int = 1
    # Input addresses
    input_release_address: int = 0
    input_template_address: int = 1
    input_clamp_engaged_address: int = 2
    clamp_feedback_enabled: bool = False
    # Timing / guards
    accept_pulse_ms: int = 1000
    min_reclamp_interval_ms: int = 3000
    release_input_debounce_ms: int = 200


@dataclass(slots=True)
class TimingConfig:
    """Inspection timer / operator-phase / commit-policy settings."""
    # Operator phase pacing (non-blocking gates)
    phase_next_part_delay_ms: int = 2000
    phase_sticker_install_delay_ms: int = 0
    # Stability thresholds before an ACCEPT commit
    accept_stable_frames: int = 1
    accept_stable_ms: int = 200
    # Commit guard
    commit_grace_ms: int = 1500
    reject_timeout_ms: int = 15000
    # Part ready
    part_ready_release_ms: int = 300
    part_ready_settle_ms_default: int = 0
    # Cache / holdover
    inference_cache_grace_ms: int = 300
    accept_holdover_ms: int = 2000
    inference_cache_ttl_ms: int = 10000
    # Safety / session
    session_idle_timeout_s: int = 300
    max_consecutive_rejects: int = 0


@dataclass(slots=True)
class IdentitySettings:
    line: str = ""  # reserved — will back a separate table later


@dataclass(slots=True)
class InferenceConfig:
    """Sticker model runtime — per-PC hardware settings."""
    mode: str = "auto"      # auto | ultralytics | onnx | openvino | tflite | classic
    device: str = "auto"    # auto | cpu | cuda
    cuda_device_id: int = 0
    num_threads: int = 4
    interval_ms: int = 0    # min ms between inference runs; 0 = every frame
    timeout_s: float = 5.0
    # Fallback model when the active template has no vision.model_path
    default_model_path: str = ""
    default_model_meta_path: str = ""


def _section(cls, data: dict[str, Any] | None):
    """Build a section dataclass from a dict, dropping unknown keys."""
    raw = data or {}
    return cls(**{k: v for k, v in raw.items() if k in cls.__slots__})


@dataclass(slots=True)
class MachineSettings:
    """Top-level machine settings — one record per machine."""
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
        # v1 files stored the I/O map under "sticker" (and a never-used "counter").
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
