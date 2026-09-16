"""DefectFlow — PLC flow strategy for Defect Scan mode.

The clamping/release behavior is identical to StickerFlow
(part_ready → clamp → inspect → accept/reject → release).
Only the evaluation logic differs (anomaly detection per-ROI).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from backend.app.services.sticker_flow import StickerFlow

if TYPE_CHECKING:
    from backend.app.services.plc_adapter import PlcAdapter
    from backend.app.models.machine_settings import StickerModeConfig

logger = __import__("logging").getLogger(__name__)


class DefectFlow(StickerFlow):
    """Defect Scan mode — uses same clamping cycle as sticker mode.

    Evaluation is handled by DefectEvaluator (not StickerEvaluator),
    but the PLC relay sequence (part_ready → clamp → ACC/NG → release)
    is identical.
    """

    @property
    def flow_name(self) -> str:
        return "defect"
