"""Desenho de depuração: mãos (pontos e esqueleto) e rótulos de texto.

Geometria vai na imagem NÃO espelhada; texto é desenhado depois do espelhamento, na posição
espelhada, para continuar legível (``draw_labels``).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ghirl.hands.mp_tracker import FINGERTIPS, HAND_CONNECTIONS, WRIST, Hand

ROLE_COLORS = {"notas": (60, 220, 60), "ritmo": (255, 160, 40), "desconhecido": (180, 180, 180)}


@dataclass
class Label:
    point: tuple[float, float]  # coordenadas na imagem NÃO espelhada
    text: str
    color: tuple[int, int, int] = (255, 255, 255)


def draw_hands(img: np.ndarray, hands: list[Hand]) -> list[Label]:
    labels: list[Label] = []
    for hand in hands:
        color = ROLE_COLORS.get(hand.role, (180, 180, 180))
        pts = hand.landmarks_px.astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            r = 6 if i in FINGERTIPS else 3
            cv2.circle(img, tuple(p), r, (255, 255, 255) if i in FINGERTIPS else color, -1, cv2.LINE_AA)
        wx, wy = hand.landmarks_px[WRIST]
        side = "esq" if hand.side == "left" else "dir"
        labels.append(Label((float(wx), float(wy) + 22), f"{hand.role} ({side}, mp={hand.handedness_raw})", color))
    return labels


def draw_labels(view: np.ndarray, labels: list[Label], mirrored: bool) -> np.ndarray:
    w = view.shape[1]
    for lb in labels:
        x, y = lb.point
        if mirrored:
            x = w - 1 - x
        (tw, th), _ = cv2.getTextSize(lb.text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x0 = int(np.clip(x - tw / 2, 0, max(0, w - tw)))
        y0 = int(np.clip(y, th + 2, view.shape[0] - 2))
        cv2.putText(view, lb.text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, lb.text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lb.color, 1, cv2.LINE_AA)
    return view
