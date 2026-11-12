"""Mãos com o MediaPipe Hand Landmarker (modo VIDEO).

Não há papel "mão das notas / mão do ritmo": quem decide é a posição. Uma nota conta como tocada
quando QUALQUER ponta de dedo está na casa certa; a pestana é medida no indicador que atravessa o
braço. A mão do ritmo fica no corpo do violão, longe das casas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ghirl.capture import Frame
from ghirl.config import HandsConfig

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
    score: float = 1.0


def fingertips(hands: list[Hand]) -> np.ndarray:
    """Pontas dos dedos (exceto o polegar) de todas as mãos, (N, 2)."""
    if not hands:
        return np.zeros((0, 2), np.float32)
    return np.concatenate([h.landmarks_px[list(FINGERTIPS)] for h in hands])


def hand_mask(hands: list[Hand], shape: tuple[int, int], dilate_px: int = 15, scale: float = 0.25) -> np.ndarray:
    """Máscara uint8 (255 = mão) do casco convexo dilatado de cada mão (calculada reduzida, < 1 ms)."""
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

    def detect(self, frame: Frame) -> list[Hand]:
        import mediapipe as mp

        # o MediaPipe exige carimbos estritamente crescentes (o replay em loop recomeça o tempo)
        t = max(int(round(frame.ts * 1000.0)), self._last_ts_ms + 1)
        self._last_ts_ms = t
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        res = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t)
        h, w = frame.image.shape[:2]
        return [Hand(np.array([[p.x * w, p.y * h] for p in lms], np.float32), float(cats[0].score))
                for lms, cats in zip(res.hand_landmarks, res.handedness, strict=False)]

    def close(self) -> None:
        self._landmarker.close()


# ---------------------------------------------------------------------------------------- gestos

FINGER_MCP = (5, 9, 13, 17)


def finger_extension(hand: Hand) -> np.ndarray:
    """Para indicador..mínimo: |ponta - pulso| / |MCP - pulso|. ~2 esticado, ~1 dobrado."""
    p = hand.landmarks_px
    tips = np.linalg.norm(p[list(FINGERTIPS)] - p[WRIST], axis=1)
    mcps = np.linalg.norm(p[list(FINGER_MCP)] - p[WRIST], axis=1)
    return tips / np.maximum(mcps, 1e-6)


def is_fist(hand: Hand) -> bool:
    return bool(finger_extension(hand).max() < 1.3)


def is_open(hand: Hand) -> bool:
    return bool(finger_extension(hand).min() > 1.55)


class GestureDetector:
    """Gestos para controlar de longe, com qualquer mão:

    - fechar e abrir a mão (punho por até ``advance_max_s``)  -> "avancar"
    - mão fechada por ``reset_hold_s``                        -> "reiniciar"

    O punho precisa aparecer em 2 quadros seguidos (e sumir em 2 seguidos), para um quadro ruim do
    MediaPipe não disparar nem cortar o gesto.
    """

    def __init__(self, advance_max_s: float = 1.5, reset_hold_s: float = 3.0, open_within_s: float = 0.7):
        self.advance_max_s = advance_max_s
        self.reset_hold_s = reset_hold_s
        self.open_within_s = open_within_s
        self.fist_since: float | None = None
        self._fist_frames = 0
        self._free_frames = 0
        self._fired = False
        self._advance_until: float | None = None
        self.ts = 0.0

    @property
    def hold_progress(self) -> float:
        """0..1 do punho em direção ao "reiniciar" (para a interface)."""
        if self.fist_since is None or self._fired:
            return 0.0
        return float(np.clip((self.ts - self.fist_since) / self.reset_hold_s, 0, 1))

    def update(self, ts: float, hands: list[Hand]) -> str | None:
        self.ts = ts
        fist = any(is_fist(h) for h in hands)
        self._fist_frames = self._fist_frames + 1 if fist else 0
        self._free_frames = 0 if fist else self._free_frames + 1
        if self.fist_since is None and self._fist_frames >= 2:
            self.fist_since = ts
            self._advance_until = None
        if self.fist_since is not None:
            if not self._fired and ts - self.fist_since >= self.reset_hold_s:
                self._fired = True
                return "reiniciar"
            if self._free_frames >= 2:  # o punho acabou
                if not self._fired and ts - self.fist_since <= self.advance_max_s:
                    self._advance_until = ts + self.open_within_s
                self.fist_since, self._fired = None, False
        if self._advance_until is not None:
            if ts > self._advance_until:
                self._advance_until = None
            elif any(is_open(h) for h in hands):
                self._advance_until = None
                return "avancar"
        return None
