"""Módulo A — localizador do braço por segmentação (T2.4; design §7.1; RF-20).

Cadeia: YOLOE prompt-free acha o "guitar" (caixa + máscara) -> SAM 2.1 refina a máscara com a
caixa como prompt -> ``mask_geometry`` extrai eixo, bordas e lado da cabeça.
Se o SAM 3 estiver disponível (``neck.sam.version: 3`` e ``models/sam3.pt``), usa texto direto.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ghirl.neck.base import NeckDetection, NeckHint
from ghirl.neck.mask_geometry import analyze_mask
from ghirl.neck.segmenters import Sam2BoxSegmenter, Sam3TextSegmenter, YoloeGuitarSegmenter

log = logging.getLogger(__name__)


class SamLocator:
    name = "sam"
    is_heavy = True

    def __init__(self, sam_version: int = 2, sam2_model: str = "models/sam2.1_t.pt",
                 yoloe_model: str = "models/yoloe-26s-seg-pf.pt", sam3_model: str = "models/sam3.pt",
                 sam3_prompts=("guitar neck", "guitar"), device: int | str = 0):
        self.sam3 = None
        if sam_version == 3:
            try:
                self.sam3 = Sam3TextSegmenter(sam3_model, sam3_prompts, device=device)
            except FileNotFoundError as exc:
                log.warning("SAM 3 indisponível, usando YOLOE + SAM 2.1: %s", exc)
        self.yoloe = YoloeGuitarSegmenter(yoloe_model, device=device) if self.sam3 is None else None
        self.sam2 = Sam2BoxSegmenter(sam2_model, device=device) if self.sam3 is None else None
        self.last_stage = ""

    def detect(self, image: np.ndarray, hint: NeckHint | None = None, frame_index: int = 0) -> NeckDetection | None:
        t0 = time.perf_counter()
        det = None
        if self.sam3 is not None:
            for r in self.sam3.segment(image):
                det = analyze_mask(r.mask, frame_index, source=f"sam3:{r.label}")
                if det is not None:
                    break
        else:
            cands = self.yoloe.segment(image)
            if hint is not None and hint.box is not None and cands:
                cands.sort(key=lambda r: -_iou_box(r.box, hint.box))
            for r in cands[:2]:
                refined = self.sam2.segment(image, box=r.box)
                for mask, src in ((refined.mask if refined else None, "yoloe+sam2"), (r.mask, "yoloe")):
                    if mask is None:
                        continue
                    det = analyze_mask(mask, frame_index, source=src)
                    if det is not None:
                        det.confidence *= r.confidence ** 0.5
                        break
                if det is not None:
                    break
        if det is not None:
            det.latency_ms = (time.perf_counter() - t0) * 1000.0
        return det


def _iou_box(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0
