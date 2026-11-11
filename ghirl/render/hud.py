"""HUD (RF-43): texto desenhado DEPOIS do espelhamento, para nunca sair invertido."""

from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_panel(img: np.ndarray, lines: list[str], origin: tuple[int, int] = (10, 10),
               scale: float = 0.55, color=(255, 255, 255), bg_alpha: float = 0.55) -> np.ndarray:
    if not lines:
        return img
    sizes = [cv2.getTextSize(t, FONT, scale, 1)[0] for t in lines]
    line_h = max(s[1] for s in sizes) + 8
    w = max(s[0] for s in sizes) + 16
    h = line_h * len(lines) + 8
    x, y = origin
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    roi = img[y:y1, x:x1]
    roi[:] = (roi * (1 - bg_alpha)).astype(img.dtype)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x + 8, y + 4 + line_h * (i + 1) - 6), FONT, scale, color, 1, cv2.LINE_AA)
    return img


def hud_lines(state: str, capture_fps: float, proc_fps: float, stage_ms: dict[str, float] | None,
              show_stages: bool) -> list[str]:
    lines = [f"{state}", f"captura {capture_fps:5.1f} FPS | processamento {proc_fps:5.1f} FPS"]
    if show_stages and stage_ms:
        lines += [f"  {k:<10} {v:6.1f} ms" for k, v in stage_ms.items()]
    return lines
