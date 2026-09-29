from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from shared.contracts.enums import SessionStatus
from shared.contracts.templates import InspectionTemplate


@dataclass(slots=True)
class SessionState:
    session_id: str
    client_id: str
    camera_index: int
    template: InspectionTemplate
    status: SessionStatus = SessionStatus.IDLE
    line_id: str | None = None
    station_id: str | None = None
    part_ready_roi_override: dict[str, Any] = field(default_factory=dict)
    sticker_roi_override: dict[str, Any] = field(default_factory=dict)
    latest_result: dict[str, Any] | None = None
    last_persisted_at: datetime | None = None
    last_persisted_key: str | None = None
    frame_index: int = 0
    current_presence: bool = False
    current_event_id: str | None = None
    current_event_key: str | None = None
    current_event_started_at: datetime | None = None
    current_event_stable_frames: int = 0
    current_event_committed: bool = False
    cooldown_until: datetime | None = None
    event_sequence: int = 0
    session_total: int = 0
    session_accept: int = 0
    session_reject: int = 0
    session_reject_breakdown: dict[str, int] = field(default_factory=dict)
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    last_committed_result: dict[str, Any] | None = None
    part_ready_ratio_history: list[float] = field(default_factory=list)
    part_ready_ema_ratio: float = -1.0
    last_overlay_b64: str | None = None
    # Debounce settle-time: timestamp frame pertama saat part_ready True
    # di ready-run saat ini. Direset ke None begitu part_ready jadi False atau
    # presence hilang.
    part_ready_settle_started_at: datetime | None = None
    # Mode constant-output PLC: True begitu enqueue_part_ready() sudah dipanggil
    # untuk ready-run saat ini, mencegah trigger duplikat. Direset saat part pergi.
    plc_part_ready_triggered: bool = False
    plc_clamp_requested_at: float = 0.0
    plc_clamp_ready_at: float = 0.0
    plc_clamp_timeout: bool = False
    plc_clamp_event_id: str | None = None
    operator_sticker_delay_started_at: float = 0.0
    operator_sticker_ready_at: float = 0.0
    operator_state: str = "IDLE"
    inspection_has_run_for_current_part: bool = False
    inspection_result_cache: dict[str, Any] | None = None
    # State inference async
    inference_result_cache: dict[str, Any] | None = None
    inference_result_ts: float = 0.0
    inference_frame_counter: int = 0
    inference_thread_busy: bool = False
    inference_submit_at: float = 0.0  # timestamp monotonic saat inference terakhir disubmit
    _inference_executor: concurrent.futures.ThreadPoolExecutor | None = None  # lazy-init per sesi
    # Counter generasi inference — bertambah tiap kali hasil inference baru
    # di-commit ke cache. Dipakai gate accept untuk menghitung run inference
    # yang benar-benar berbeda (bukan cuma baca ulang hasil cache yang sama).
    inference_result_generation: int = 0
    inference_accept_count: int = 0
    inference_accept_first_ts: float = 0.0
    inference_last_counted_generation: int = -1
    # TTL cache inference (ms) — configurable per sesi dari app config.
    inference_cache_ttl_ms: int = 10000
    gap_ref_cache: dict[str, Any] = field(default_factory=dict)  # cache untuk gap reference patch yang sudah dimuat
    part_removed_seen_at: datetime | None = None
    # Counter hysteresis: jumlah frame settled berturut-turut.
    # Direset ke 0 kalau part_ready/presence hilang.
    settle_frame_count: int = 0
    # Timestamp saat part pertama kali settled (consecutive_part_ready_frames >= threshold).
    # Dipakai untuk reject timeout: kalau tidak ada accept-commit dalam reject_timeout_ms, reject sebagai COMMIT_TIMEOUT.
    # Direset ke None saat PLC kembali IDLE atau part pergi.
    part_ready_settled_at: datetime | None = None
    # Cooldown inference: timestamp (ms) run inference terakhir.
    # Mencegah inference jalan lebih dari sekali per detik.
    consecutive_part_ready_frames: int = 0
    last_inference_ms: int = 0
    # Interval inference (ms): waktu minimum antar-run inference.
    # 0 = tidak terbatas (tiap frame), 200 = maks ~5 fps inference.
    inference_interval_ms: int = 0
    # COOLDOWN manual release: timestamp (detik sejak epoch) sampai kapan
    # re-clamp diblokir setelah IN1 (manual release). Mencegah re-clamp instan.
    manual_release_cooldown_until: float = 0.0
    # Timestamp aktivitas terakhir (detik sejak epoch) — diupdate tiap proses frame.
    # Dipakai untuk auto-end idle timeout.
    last_activity_at: float = 0.0
    # Counter reject berturut-turut — bertambah tiap keputusan reject di-commit.
    # Direset ke 0 saat accept. Dipakai untuk mewajibkan N reject berturut-turut sebelum reject final.
    consecutive_reject_count: int = 0
    # Maks reject berturut-turut yang diizinkan sebelum auto-commit reject (0 = langsung, tanpa jeda).
    max_consecutive_rejects: int = 0
    # ── State Latch Part-Ready ──
    # Begitu raw_part_ready settled dan clamp diminta, kita latch supaya
    # drop sesaat pada raw_part_ready (bayangan, getaran) tidak membatalkan siklus.
    part_ready_latched: bool = False
    part_ready_latched_at: datetime | None = None
    # Timestamp saat raw_part_ready terakhir drop ketika latched.
    # Dipakai untuk debounce pelepasan latch lewat release_ms.
    part_ready_unsettled_at: datetime | None = None
    # ── Pelacakan Stabilitas Inspection Policy ──
    # Policy key = hash dari (decision, reject_reason_code, detected_class, expected_class)
    # Dipakai untuk melacak frame stabil berturut-turut sebelum commit.
    # Timestamp level-siklus: saat ACCEPT pertama terlihat di siklus clamping ini.
    # Bertahan melewati celah deteksi (menjembatani expiry holdover) — direset
    # cuma saat commit atau saat siklus reset (part diangkat). Dipakai untuk cek commit_grace_ms.
    accept_cycle_started_at: datetime | None = None
    last_policy_key: str = ""
    policy_stable_started_at: datetime | None = None
    policy_stable_frames: int = 0
    # Setelah commit ACCEPT, tunggu part benar-benar pergi sebelum mengizinkan siklus berikutnya.
    awaiting_part_removal_after_commit: bool = False
    policy_holdover_expires_at: datetime | None = None
    part_absent_started_at: datetime | None = None
    # ── Cache Hasil Inference untuk Menangani Tangan yang Menghalangi ──
    # Saat YOLO mendeteksi class valid tapi part_ready sempat drop (mis. tangan
    # menghalangi saat menunggu commit), kita cache hasil inference valid
    # terakhir beserta timestamp-nya. Kalau part_ready kembali dalam jendela
    # grace, kita pakai hasil cache itu daripada menjalankan ulang inference
    # (yang mungkin gagal karena terhalang).
    last_valid_inference: dict[str, Any] | None = None
    last_valid_inference_ts: float = 0.0  # timestamp monotonic
