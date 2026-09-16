"""
PLC Worker — strategy-based, reads config from MachineSettings DB.

Flow:
  IDLE → CLAMPING → ACCEPT (clamp OFF, OK pulse → IDLE)
                   → REJECT (enji buzzer ON, clamp stays → wait release → IDLE)
  Any state → Input release → IDLE (all off)
  Any state → Input template → cycle template

Strategy pattern:
  PlcWorker delegates mode-specific behavior to PlcFlowStrategy.
  Strategy is selected by `validator_mode` from the active template.
  Strategy reads addresses/timing from MachineSettings (DB, not env).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from backend.app.services.counter_flow import CounterFlow
from backend.app.services.plc_flow_strategy import PlcFlowStrategy
from backend.app.services.sticker_flow import StickerFlow

if TYPE_CHECKING:
    from backend.app.services.plc_adapter import PlcAdapter

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 0.2
_INPUT_READ_COUNT = 8
_CMD_QUEUE_MAX = 64


def _build_strategy(
    validator_mode: str,
    adapter: "PlcAdapter",
    settings,  # MachineSettings
    num_channels: int = 4,
    dry_run: bool = False,
) -> PlcFlowStrategy:
    """Factory: select strategy based on validator_mode."""
    mode = (validator_mode or "sticker").strip().lower()
    if mode == "component_count":
        return CounterFlow(adapter, settings.counter, num_channels)
    if mode == "defect":
        from backend.app.services.defect_flow import DefectFlow
        return DefectFlow(adapter, settings.sticker, num_channels)
    # Default: sticker mode
    return StickerFlow(adapter, settings.sticker, num_channels)


class PlcWorker:
    def __init__(
        self,
        adapter: "PlcAdapter",
        accept_pulse_ms: int = 1000,
        num_channels: int = 4,
        input_release_address: int = 0,
        input_template_address: int = 1,
        input_clamp_engaged_address: int = 2,
        clamp_feedback_enabled: bool = False,
        relay_clamp_address: int = 3,
        relay_ok_light_buzzer_address: int = 2,
        relay_enji_buzzer_address: int = 1,
    ) -> None:
        self._adapter = adapter
        self._num_channels = num_channels
        self._accept_pulse_ms = int(accept_pulse_ms)

        # Legacy constructor args (kept for backward compat / container.py wiring)
        # These are OVERRIDDEN once strategy is set from MachineSettings.
        self._input_release_address = max(0, int(input_release_address))
        self._input_template_address = max(0, int(input_template_address))
        self._input_clamp_engaged_address = max(0, int(input_clamp_engaged_address))
        self._clamp_feedback_enabled = bool(clamp_feedback_enabled)
        self._relay_clamp = max(0, int(relay_clamp_address))
        self._relay_ok_light_buzzer = max(0, int(relay_ok_light_buzzer_address))
        self._relay_enji_buzzer = max(0, int(relay_enji_buzzer_address))

        # Strategy (set via set_strategy or set_validator_mode)
        self._strategy: PlcFlowStrategy | None = None

        # ── Cycle Lock State ──
        self._cycle_locked: bool = False
        self._cycle_lock_reason: str = ""
        self._last_clamp_off_at: float = 0.0
        self._last_part_ready_event_id: str | None = None

        # Config (set from container)
        self._min_reclamp_interval_ms: int = 3000
        self._release_input_debounce_ms: int = 500
        self._dry_run: bool = False

        # Release input edge tracking
        self._release_input_started_at: float | None = None
        self._release_input_triggered: bool = False

        # State
        self._state: str = "IDLE"
        self._accept_pulse_end: float | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Command queue
        self._cmd_queue: list[dict] = []
        self._cmd_event = threading.Event()

        # Input debounce
        self._last_input_press: dict[int, float] = {}
        self._input_debounce_s: float = 0.5
        self._template_cycle_event_id: int = 0
        self._last_template_cycle_at: float | None = None
        self._last_input_snapshot: list[bool] = []

        # Item 1: current decision event_id for actuation callback
        self._current_decision_event_id: str | None = None

        # Health tracking for commit interlock (Item 1)
        self._last_poll_ok_at: float = 0.0
        self._last_write_ok_at: float = 0.0

        # Reconnect backoff — prevents tight reconnect loop when device is dead
        self._reconnect_failures: int = 0
        self._reconnect_backoff_until: float = 0.0
        self._max_reconnect_backoff_s: float = 30.0  # cap at 30s

        # Callbacks
        self._template_cycle_callback = None
        self._on_state_change_callback = None
        self._on_actuation_result_callback = None  # Item 1: actuation ACK/NACK

    # ── Strategy setup ───────────────────────────────────────────────

    def set_strategy(self, strategy: PlcFlowStrategy) -> None:
        """Set the active flow strategy. Called from container after seed."""
        self._strategy = strategy
        logger.info("[plc-worker] strategy set: %s", strategy.flow_name)

    def set_validator_mode(self, mode: str, settings=None) -> None:
        """Select strategy by validator_mode string.

        If settings (MachineSettings) is provided, build strategy from DB.
        Otherwise, fall back to legacy constructor args (backward compat).
        """
        if settings is not None:
            self._strategy = _build_strategy(
                mode, self._adapter, settings, self._num_channels, self._dry_run
            )
            logger.info(
                "[plc-worker] strategy built from MachineSettings: %s",
                self._strategy.flow_name,
            )
        else:
            # Legacy fallback — build from constructor args
            from backend.app.models.machine_settings import StickerModeConfig, CounterModeConfig
            if mode == "component_count":
                cfg = CounterModeConfig(
                    relay_clamp_address=self._relay_clamp,
                    relay_ok_light_buzzer_address=self._relay_ok_light_buzzer,
                    relay_enji_buzzer_address=self._relay_enji_buzzer,
                    input_release_address=self._input_release_address,
                    input_template_address=self._input_template_address,
                    input_clamp_engaged_address=self._input_clamp_engaged_address,
                    clamp_feedback_enabled=self._clamp_feedback_enabled,
                )
                self._strategy = CounterFlow(self._adapter, cfg, self._num_channels)
            else:
                cfg = StickerModeConfig(
                    relay_clamp_address=self._relay_clamp,
                    relay_ok_light_buzzer_address=self._relay_ok_light_buzzer,
                    relay_enji_buzzer_address=self._relay_enji_buzzer,
                    input_release_address=self._input_release_address,
                    input_template_address=self._input_template_address,
                    input_clamp_engaged_address=self._input_clamp_engaged_address,
                    clamp_feedback_enabled=self._clamp_feedback_enabled,
                    accept_pulse_ms=self._accept_pulse_ms,
                )
                if mode == "defect":
                    from backend.app.services.defect_flow import DefectFlow
                    self._strategy = DefectFlow(self._adapter, cfg, self._num_channels)
                else:
                    self._strategy = StickerFlow(self._adapter, cfg, self._num_channels)

    # ── Public API (unchanged signatures) ───────────────────────────

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        try:
            self._adapter.connect()
            self._adapter.all_off(self._num_channels)
        except Exception as exc:
            # Don't abort: start the worker anyway so it self-heals once the port
            # is available (poll loop reconnects + writes lazy-connect on demand).
            logger.warning(
                "[plc-worker] initial connect failed (%s) — starting anyway, will retry", exc
            )
        self._state = "IDLE"
        self._thread = threading.Thread(target=self._loop, name="qc-plc-worker", daemon=True)
        self._thread.start()
        logger.info("[plc-worker] started (state=%s, strategy=%s)", self._state, self._strategy.flow_name if self._strategy else "none")

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        self._cmd_event.set()
        try:
            self._adapter.all_off(self._num_channels)
        except Exception as exc:
            logger.error("[plc-worker] stop all_off failed: %s", exc)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        try:
            self._adapter.disconnect()
        except Exception:
            pass
        self._state = "IDLE"
        logger.info("[plc-worker] stopped")

    def enqueue_part_ready(self, *, event_id: str | None) -> None:
        self._enqueue_cmd({"type": "part_ready", "event_id": event_id})

    def notify_decision(self, decision: str, *, event_id: str | None = None) -> None:
        self._enqueue_cmd({"type": "decision", "decision": decision, "event_id": event_id})

    def force_release(self, *, reason: str = "manual") -> None:
        self._enqueue_cmd({"type": "force_release", "reason": reason})

    def set_template_cycle_callback(self, callback) -> None:
        self._template_cycle_callback = callback

    def set_on_state_change_callback(self, callback) -> None:
        self._on_state_change_callback = callback

    def set_on_actuation_result_callback(self, callback) -> None:
        """Item 1: set callback for actuation ACK/NACK.

        Callback signature: cb(event_id: str, decision: str, ok: bool, reason: str = "")
        Called after _on_accept/_on_reject completes (ok=True) or fails (ok=False).
        """
        self._on_actuation_result_callback = callback

    def configure_guards(
        self,
        *,
        min_reclamp_interval_ms: int = 3000,
        release_input_debounce_ms: int = 500,
        dry_run: bool = False,
    ) -> None:
        self._min_reclamp_interval_ms = max(0, int(min_reclamp_interval_ms))
        self._release_input_debounce_ms = max(0, int(release_input_debounce_ms))
        self._dry_run = bool(dry_run)

    def unlock_cycle(self, *, reason: str = "manual") -> None:
        logger.info("[plc-worker] cycle unlocked — %s", reason)
        self._cycle_locked = False
        self._cycle_lock_reason = ""
        self._last_part_ready_event_id = None

    def status(self) -> dict:
        with self._lock:
            clamp_engaged = self._clamp_engaged_from_snapshot_locked()
            return {
                "running": self._thread is not None and self._thread.is_alive(),
                "state": self._state,
                "connected": self._adapter.is_connected(),
                "strategy": self._strategy.flow_name if self._strategy else "none",
                "clamp_feedback_enabled": self._clamp_feedback_enabled,
                "clamp_feedback_address": self._input_clamp_engaged_address,
                "clamp_engaged": clamp_engaged,
                "template_cycle_event_id": self._template_cycle_event_id,
                "last_template_cycle_at": self._last_template_cycle_at,
                "last_input_snapshot": list(self._last_input_snapshot),
                "cmd_queue_depth": len(self._cmd_queue),
                "cycle_locked": self._cycle_locked,
                "cycle_lock_reason": self._cycle_lock_reason,
                "dry_run": self._dry_run,
                "last_poll_ok_at": self._last_poll_ok_at,
                "last_write_ok_at": self._last_write_ok_at,
                **self._adapter.status(),
            }

    def is_healthy(self, max_stale_ms: int = 5000) -> bool:
        """Return True if PLC has been polled successfully within max_stale_ms.

        Used by inspection_session as a commit interlock: if the PLC link is
        stale/disconnected, commits are blocked to prevent persisting results
        while no relay can fire.
        """
        import time
        if self._thread is None or not self._thread.is_alive():
            return False
        if self._dry_run:
            return True  # dry-run doesn't need live PLC
        if not self._adapter.is_connected():
            return False
        last_ok = self._last_poll_ok_at
        if last_ok <= 0:
            return False  # never successfully polled
        age_ms = (time.monotonic() - last_ok) * 1000.0
        return age_ms <= float(max_stale_ms)

    def clamp_engaged(self) -> bool:
        with self._lock:
            return self._clamp_engaged_from_snapshot_locked()

    def _clamp_engaged_from_snapshot_locked(self) -> bool:
        if self._clamp_feedback_enabled:
            if self._input_clamp_engaged_address < len(self._last_input_snapshot):
                return bool(self._last_input_snapshot[self._input_clamp_engaged_address])
            return False
        return self._state in {"CLAMPING", "CLAMPED", "REJECT_BUZZER"}

    # ── Backward-compatible coil access for diagnostics ──────────────

    @property
    def relay_clamp(self) -> int:
        return self._relay_clamp

    @property
    def relay_ok_light_buzzer(self) -> int:
        return self._relay_ok_light_buzzer

    @property
    def relay_enji_buzzer(self) -> int:
        return self._relay_enji_buzzer

    @property
    def num_channels(self) -> int:
        return self._num_channels

    # ── Command Queue ────────────────────────────────────────────────

    def _enqueue_cmd(self, cmd: dict) -> None:
        with self._lock:
            if len(self._cmd_queue) >= _CMD_QUEUE_MAX:
                self._cmd_queue.pop(0)
                logger.warning("[plc-worker] cmd queue full, dropped oldest")
            self._cmd_queue.append(cmd)
        self._cmd_event.set()

    def _dequeue_cmd(self) -> dict | None:
        with self._lock:
            if self._cmd_queue:
                return self._cmd_queue.pop(0)
        return None

    # ── State setter with callback ───────────────────────────────────

    def _set_state(self, new_state: str) -> None:
        with self._lock:
            old = self._state
            self._state = new_state
        logger.info("[plc-worker] %s → %s", old, new_state)
        if self._on_state_change_callback and old != new_state:
            try:
                self._on_state_change_callback(old, new_state)
            except Exception as exc:
                logger.error("[plc-worker] state change callback error: %s", exc)

    def _write_coil(self, addr: int, value: bool) -> None:
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                self._adapter.write_coil(addr, value)
                self._last_write_ok_at = time.monotonic()  # Item 1: track successful write
                return
            except Exception as exc:
                logger.error("[plc-worker] write_coil addr=%d attempt %d failed: %s", addr, attempt, exc)
                if attempt < max_retries:
                    time.sleep(0.1 * attempt)
        raise RuntimeError(f"write_coil addr={addr} failed after {max_retries} attempts")

    def _all_off(self, reason: str) -> None:
        logger.info("[plc-worker] ALL OFF — %s", reason)
        for i in range(self._num_channels):
            try:
                self._write_coil(i, False)
                logger.info("[plc-worker] coil[%d] OFF (ok)", i)
            except Exception as exc:
                logger.error("[plc-worker] coil[%d] OFF failed: %s", i, exc)
        self._last_clamp_off_at = time.time()
        self._set_state("IDLE")

    # ── ACCEPT / REJECT (delegated to strategy when available) ───────

    def _on_accept(self) -> None:
        _event_id = self._current_decision_event_id
        _decision = "ACCEPT"
        try:
            if self._strategy is not None:
                self._strategy.on_accept(self)
                self._set_state("ACCEPT_PULSE")
            else:
                # Legacy hardcoded fallback
                self._set_state("ACCEPT_PULSE")
                self._write_coil(self._relay_clamp, False)
                self._write_coil(self._relay_ok_light_buzzer, True)
                self._accept_pulse_end = time.time() + (self._accept_pulse_ms / 1000.0)
                logger.info("[plc-worker] ACCEPT — CH1 pulse %dms (legacy)", self._accept_pulse_ms)
            # Item 1: actuation ACK
            self._fire_actuation_result(_event_id, _decision, True)
        except Exception as exc:
            logger.error("[plc-worker] ACCEPT actuation failed: %s", exc)
            self._enter_plc_fault(f"accept_failed: {exc}")
            self._fire_actuation_result(_event_id, _decision, False, str(exc))

    def _on_reject(self) -> None:
        _event_id = self._current_decision_event_id
        _decision = "REJECT"
        try:
            if self._strategy is not None:
                self._strategy.on_reject(self)
                self._set_state("REJECT_BUZZER")
            else:
                self._set_state("REJECT_BUZZER")
                self._write_coil(self._relay_enji_buzzer, True)
                self._write_coil(self._relay_clamp, True)
                logger.info("[plc-worker] REJECT — CH2 ON, waiting Input 1 (legacy)")
            # Item 1: actuation ACK
            self._fire_actuation_result(_event_id, _decision, True)
        except Exception as exc:
            logger.error("[plc-worker] REJECT actuation failed: %s", exc)
            self._enter_plc_fault(f"reject_failed: {exc}")
            self._fire_actuation_result(_event_id, _decision, False, str(exc))

    def _finish_accept_pulse(self) -> None:
        if self._strategy is not None and isinstance(self._strategy, StickerFlow):
            self._strategy.finish_accept_pulse(self)
        else:
            self._write_coil(self._relay_ok_light_buzzer, False)
            self._accept_pulse_end = None
        self._last_clamp_off_at = time.time()
        self._set_state("IDLE")
        logger.info("[plc-worker] ACCEPT done → IDLE")

    # ── Command handlers ─────────────────────────────────────────────

    def _handle_cmd(self, cmd: dict) -> None:
        cmd_type = cmd.get("type")
        try:
            if cmd_type == "part_ready":
                self._cmd_part_ready(cmd)
            elif cmd_type == "decision":
                self._cmd_decision(cmd)
            elif cmd_type == "force_release":
                self._cmd_force_release(cmd)
            else:
                logger.warning("[plc-worker] unknown cmd type: %s", cmd_type)
        except Exception as exc:
            logger.error("[plc-worker] cmd %s error: %s", cmd_type, exc)

    def _cmd_part_ready(self, cmd: dict) -> None:
        event_id = cmd.get("event_id")
        if event_id and event_id == self._last_part_ready_event_id:
            logger.info("[plc-worker] part_ready dedup — event=%s", event_id)
            return
        if self._cycle_locked:
            logger.info("[plc-worker] part_ready blocked (cycle_locked=%s) — event=%s",
                        self._cycle_lock_reason, event_id)
            return
        now = time.time()
        if self._last_clamp_off_at > 0:
            elapsed_since_release_ms = (now - self._last_clamp_off_at) * 1000.0
            if elapsed_since_release_ms < self._min_reclamp_interval_ms:
                logger.info(
                    "[plc-worker] part_ready blocked (reclamp interval %.0fms < %dms) — event=%s",
                    elapsed_since_release_ms, self._min_reclamp_interval_ms, event_id,
                )
                return
        with self._lock:
            if self._state != "IDLE":
                logger.info("[plc-worker] part_ready ignored (state=%s)", self._state)
                return
        self._last_part_ready_event_id = event_id

        if self._strategy is not None:
            self._strategy.on_part_ready(self)
            self._set_state("CLAMPING")
        else:
            # Legacy fallback
            self._write_coil(self._relay_clamp, True)
            self._write_coil(self._relay_ok_light_buzzer, False)
            self._write_coil(self._relay_enji_buzzer, False)
            self._set_state("CLAMPING")
        logger.info("[plc-worker] CLAMPING — event=%s", event_id)

    def _cmd_decision(self, cmd: dict) -> None:
        decision = cmd.get("decision")
        event_id = cmd.get("event_id")
        with self._lock:
            if self._state not in {"CLAMPING", "CLAMPED"}:
                # Item 1: don't silently drop — log and fire NACK callback
                logger.warning(
                    "[plc-worker] decision '%s' dropped — state=%s (not CLAMPING/CLAMPED)",
                    decision, self._state,
                )
                self._fire_actuation_result(event_id, decision or "?", False, f"wrong_state:{self._state}")
                return
        # Store event_id for the actuation callback
        self._current_decision_event_id = event_id
        if decision == "ACCEPT":
            self._on_accept()
        elif decision == "REJECT":
            self._on_reject()

    def _cmd_force_release(self, cmd: dict) -> None:
        self._all_off(cmd.get("reason", "manual"))

    # ── Item 1: FAULT state + actuation callback ─────────────────────

    def _enter_plc_fault(self, reason: str) -> None:
        """Transition to FAULT state: lock cycle, drive fault output."""
        with self._lock:
            self._cycle_locked = True
            self._cycle_lock_reason = f"plc_fault:{reason}"
        self._set_state("FAULT")
        # Drive fault output (enji buzzer relay) if reachable
        try:
            self._write_coil(self._relay_enji_buzzer, True)
        except Exception:
            pass  # best-effort; already in fault
        logger.error("[plc-worker] FAULT state entered: %s", reason)

    def _fire_actuation_result(self, event_id: str | None, decision: str, ok: bool, reason: str = "") -> None:
        """Item 1: notify caller of actuation result (ACK/NACK)."""
        if self._on_actuation_result_callback is not None:
            try:
                self._on_actuation_result_callback(event_id, decision, ok, reason)
            except Exception as exc:
                logger.error("[plc-worker] actuation result callback error: %s", exc)

    # ── Main loop ────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                # Process all pending commands first
                while True:
                    cmd = self._dequeue_cmd()
                    if cmd is None:
                        break
                    self._handle_cmd(cmd)

                # Check accept pulse timeout
                if self._accept_pulse_end is not None and time.time() >= self._accept_pulse_end:
                    self._finish_accept_pulse()

                # Also check strategy-level accept pulse (StickerFlow)
                if self._strategy is not None and isinstance(self._strategy, StickerFlow):
                    if self._strategy.is_accept_pulse_complete() and self._state == "ACCEPT_PULSE":
                        self._finish_accept_pulse()

                # Poll inputs
                self._poll_inputs()
            except Exception as exc:
                logger.error("[plc-worker] _loop unhandled exception: %s", exc, exc_info=True)
            self._cmd_event.wait(timeout=_POLL_INTERVAL_S)
            self._cmd_event.clear()

    def _poll_inputs(self) -> None:
        # Reconnect backoff: skip polling if we're in backoff window
        if self._reconnect_backoff_until > time.time():
            return
        try:
            inputs = self._adapter.read_inputs(address=0, count=_INPUT_READ_COUNT)
        except Exception as exc:
            self._reconnect_failures += 1
            # Exponential backoff: 2^failures seconds, capped at _max_reconnect_backoff_s
            _delay = min(self._max_reconnect_backoff_s, 2 ** self._reconnect_failures)
            # Minimum 2s floor to avoid rapid retries on sustained failures
            _delay = max(2.0, _delay)
            self._reconnect_backoff_until = time.time() + _delay
            logger.warning(
                "[plc-worker] read_inputs error (%d consecutive): %s — "
                "reconnect backoff %.0fs",
                self._reconnect_failures, exc, _delay,
            )
            try:
                # Force full disconnect first so the adapter actually reopens
                # the serial port (connect() early-returns if _connected=True).
                self._adapter.disconnect()
            except Exception as disconnect_exc:
                logger.warning("[plc-worker] disconnect before reconnect failed: %s", disconnect_exc)
            try:
                self._adapter.connect()
                logger.info("[plc-worker] reconnect successful after read error")
            except Exception as reconnect_exc:
                logger.error("[plc-worker] reconnect failed: %s", reconnect_exc)
            return
        # Successful poll — reset backoff counter
        self._reconnect_failures = 0
        self._reconnect_backoff_until = 0.0
        if not inputs or len(inputs) < 2:
            return

        now = time.time()
        with self._lock:
            self._last_input_snapshot = list(inputs[:_INPUT_READ_COUNT])
        # Track successful poll for commit interlock health check
        self._last_poll_ok_at = time.monotonic()

        # Input release (IN1) — edge triggered + stable debounce
        _release_addr = self._input_release_address
        if _release_addr < len(inputs):
            _release_active = bool(inputs[_release_addr])
            _now = now
            if _release_active:
                if self._release_input_started_at is None:
                    self._release_input_started_at = _now
                _stable_ms = (_now - self._release_input_started_at) * 1000.0
                logger.debug(
                    "[plc-worker] IN1 HIGH — stable %.0fms / debounce %dms",
                    _stable_ms, self._release_input_debounce_ms,
                )
                if _stable_ms >= self._release_input_debounce_ms:
                    if not self._release_input_triggered:
                        self._release_input_triggered = True
                        logger.info("[plc-worker] INPUT 1 — Manual Release (stable %.0fms)", _stable_ms)
                        self._all_off("input_1")
            else:
                if self._release_input_started_at is not None and self._release_input_triggered:
                    logger.debug("[plc-worker] IN1 LOW — release cycle complete")
                self._release_input_started_at = None
                self._release_input_triggered = False

        # Input template cycle (IN2) — edge triggered + debounce
        if self._input_template_address < len(inputs) and inputs[self._input_template_address]:
            last = self._last_input_press.get(self._input_template_address, 0.0)
            if now - last > self._input_debounce_s:
                self._last_input_press[self._input_template_address] = now
                logger.info("[plc-worker] INPUT 2 — Ganti Template")
                with self._lock:
                    self._template_cycle_event_id += 1
                    self._last_template_cycle_at = now
                    self._last_input_snapshot = list(inputs[:8])
                if self._template_cycle_callback:
                    try:
                        self._template_cycle_callback()
                    except Exception as exc:
                        logger.error("[plc-worker] template cycle error: %s", exc)

        # Clamp feedback
        if (
            self._clamp_feedback_enabled
            and self._input_clamp_engaged_address < len(inputs)
            and inputs[self._input_clamp_engaged_address]
        ):
            with self._lock:
                state = self._state
            if state == "CLAMPING":
                self._set_state("CLAMPED")
