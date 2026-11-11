"""Desenho do overlay ancorado no mundo (desenhado na imagem NÃO espelhada).

No Marco 1 só existe a grade fixa de bolinhas (RF-40); depois a geometria vem do braço.
"""

from __future__ import annotations

import cv2
import numpy as np

# Uma cor por corda (BGR), corda 1 (Mi aguda) ... corda 6 (Mi grave).
STRING_COLORS = {
    1: (80, 80, 255),
    2: (60, 180, 255),
    3: (60, 230, 230),
    4: (90, 220, 90),
    5: (255, 170, 60),
    6: (230, 90, 200),
}


def fixed_grid_centers(width: int, height: int, n_strings: int = 6, n_frets: int = 12,
                       box: tuple[float, float, float, float] = (0.15, 0.35, 0.85, 0.65)) -> np.ndarray:
    """Centros (n_strings, n_frets, 2) de uma grade fixa ocupando ``box`` (x0, y0, x1, y1 normalizados).

    Linha 0 = corda 6 (em cima, como os bordões ficam para um destro de frente para a câmera).
    """
    x0, y0, x1, y1 = box
    xs = np.linspace(x0 * width, x1 * width, n_frets)
    ys = np.linspace(y0 * height, y1 * height, n_strings)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx, gy], axis=-1)


def draw_fixed_grid(img: np.ndarray, radius: int | None = None) -> np.ndarray:
    h, w = img.shape[:2]
    centers = fixed_grid_centers(w, h)
    n_strings, n_frets = centers.shape[:2]
    r = radius or max(4, int(min(w, h) * 0.015))
    for row in range(n_strings):
        string = n_strings - row  # linha 0 = corda 6
        color = STRING_COLORS[string]
        p0 = tuple(int(v) for v in centers[row, 0])
        p1 = tuple(int(v) for v in centers[row, -1])
        cv2.line(img, p0, p1, (200, 200, 200), 1, cv2.LINE_AA)
        for col in range(n_frets):
            c = tuple(int(v) for v in centers[row, col])
            cv2.circle(img, c, r, color, -1, cv2.LINE_AA)
            cv2.circle(img, c, r, (20, 20, 20), 1, cv2.LINE_AA)
    return img
