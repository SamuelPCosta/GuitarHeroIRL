"""Mãos com MediaPipe Hand Landmarker (T1.7; RF-10, RF-11 provisório, RF-13).

Lateralidade: a documentação do MediaPipe diz que o rótulo supõe imagem ESPELHADA, mas o experimento
da T1.7 (MediaPipe 1.2.2, imagem original não espelhada) mostrou o rótulo CORRETO: mão esquerda real
→ "Left" em 54/54 quadros, direita → "Right" em 59/60; na imagem espelhada os rótulos se invertem.
Por isso ``hands.handedness_swap = false``. Reconfirmar com ``python -m ghirl.hands.handedness_check``
se trocar de versão do MediaPipe.
O papel definitivo de cada mão (notas/ritmo) virá da geometria em relação ao braço (T2.8).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from ghirl.app.config import HandsConfig
from ghirl.capture.base import Frame

Side = Literal["left", "right"]
Role = Literal["notas", "ritmo", "desconhecido"]

# Índices dos pontos do MediaPipe Hands
WRIST = 0
THUMB_TIP, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 4, 5, 6, 7, 8
MIDDLE_TIP, RING_TIP, PINKY_TIP = 12, 16, 20
FINGERTIPS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class HandTrackerError(RuntimeError):
    pass


@dataclass
class Hand:
    landmarks_px: np.ndarray  # (21, 2) float32, pixels da imagem NÃO espelhada
    landmarks_z: np.ndarray  # (21,) profundidade relativa do MediaPipe
    handedness_raw: str  # rótulo do MediaPipe, sem correção ("Left"/"Right")
    score: float
    side: Side  # mão real do usuário, após a correção configurada
    role: Role = "desconhecido"


def real_side(raw_label: str, swap: bool) -> Side:
    side: Side = "left" if raw_label.lower().startswith("l") else "right"
    if swap:
        side = "right" if side == "left" else "left"
    return side


def assign_roles_by_label(hands: list[Hand], fretting: Side) -> None:
    """Papel provisório pelo rótulo corrigido. Se as duas mãos tiverem o mesmo lado, a de maior
    confiança fica com ele e a outra recebe o lado oposto."""
    if len(hands) == 2 and hands[0].side == hands[1].side:
        lo = min(hands, key=lambda h: h.score)
        lo.side = "right" if lo.side == "left" else "left"
    for h in hands:
        h.role = "notas" if h.side == fretting else "ritmo"


def hand_mask(hands: list[Hand], shape: tuple[int, int], dilate_px: int = 15, scale: float = 0.25) -> np.ndarray:
    """Máscara uint8 (255 = mão) do casco convexo dilatado de cada mão, no tamanho ``shape`` (h, w).

    Calculada em resolução reduzida (``scale``) e ampliada, para custar < 1 ms.
    """
    h, w = shape
    sh, sw = max(1, int(h * scale)), max(1, int(w * scale))
    small = np.zeros((sh, sw), np.uint8)
    for hand in hands:
        pts = (hand.landmarks_px * scale).astype(np.int32)
        cv2.fillConvexPoly(small, cv2.convexHull(pts), 255)
    k = max(1, int(round(dilate_px * scale)))
    if hands and k > 0:
        small = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


class HandTracker:
    def __init__(self, cfg: HandsConfig):
        from mediapipe.tasks.python import BaseOptions, vision

        model = Path(cfg.model_path)
        if not model.exists():
            raise HandTrackerError(
                f"modelo não encontrado: {model}. Baixe com:\n  curl -L -o {model} "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
                "float16/latest/hand_landmarker.task"
            )
        self.cfg = cfg
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.max_hands,
            min_hand_detection_confidence=cfg.min_conf,
            min_hand_presence_confidence=cfg.min_conf,
            min_tracking_confidence=cfg.min_conf,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opts)
        self._last_ts_ms = -1

    def _timestamp_ms(self, ts: float) -> int:
        # O MediaPipe exige carimbos estritamente crescentes (replay em loop recomeça o tempo).
        t = int(round(ts * 1000.0))
        if t <= self._last_ts_ms:
            t = self._last_ts_ms + 1
        self._last_ts_ms = t
        return t

    def detect(self, frame: Frame) -> list[Hand]:
        import mediapipe as mp

        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        res = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self._timestamp_ms(frame.ts)
        )
        h, w = frame.image.shape[:2]
        hands: list[Hand] = []
        for lms, cats in zip(res.hand_landmarks, res.handedness, strict=False):
            xy = np.array([[p.x * w, p.y * h] for p in lms], np.float32)
            z = np.array([p.z for p in lms], np.float32)
            cat = cats[0]
            raw = cat.category_name or cat.display_name
            hands.append(Hand(landmarks_px=xy, landmarks_z=z, handedness_raw=raw, score=float(cat.score),
                              side=real_side(raw, self.cfg.handedness_swap)))
        assign_roles_by_label(hands, self.cfg.fretting)
        return hands

    def close(self) -> None:
        self._landmarker.close()
