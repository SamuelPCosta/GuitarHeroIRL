"""Geometria 2D e tipos do braço.

Convenções: pontos (N, 2) float64 em px (x, y); reta = (ponto, direção unitária); ``AxisFrame``: s ao
longo do eixo (``u``), t através (``n`` = ``u`` girado +90°, y para baixo). Toda a geometria do braço
(eixo, bordas, calibração) vive no quadro-chave K; ``H`` leva K ao quadro atual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# ==================================================================================================
# retas, homografias e quadro do eixo
# ==================================================================================================

def as_points(pts) -> np.ndarray:
    a = np.asarray(pts, dtype=np.float64)
    return a.reshape(-1, 2)


def apply_homography(H: np.ndarray, pts) -> np.ndarray:
    """Aplica a homografia 3x3 ``H`` a pontos (N, 2)."""
    p = as_points(pts)
    ph = np.hstack([p, np.ones((len(p), 1))]) @ np.asarray(H, np.float64).T
    w = ph[:, 2:3]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("ponto levado ao infinito pela homografia")
    return ph[:, :2] / w


def normalize(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("vetor nulo não pode ser normalizado")
    return v / n


def perp(u) -> np.ndarray:
    """``u`` girado +90°: (x, y) -> (-y, x)."""
    u = np.asarray(u, np.float64)
    return np.array([-u[1], u[0]])


@dataclass(frozen=True)
class Line:
    point: np.ndarray
    direction: np.ndarray  # unitária

    @staticmethod
    def through(p0, p1) -> Line:
        p0 = np.asarray(p0, np.float64)
        return Line(p0, normalize(np.asarray(p1, np.float64) - p0))

    @staticmethod
    def make(point, direction) -> Line:
        return Line(np.asarray(point, np.float64), normalize(direction))

    def signed_distance(self, pts) -> np.ndarray:
        """Distância com sinal (positivo no lado de ``perp(direction)``)."""
        return (as_points(pts) - self.point) @ perp(self.direction)

    def project(self, pts) -> np.ndarray:
        """Pé da perpendicular de cada ponto sobre a reta."""
        p = as_points(pts)
        s = (p - self.point) @ self.direction
        return self.point + s[:, None] * self.direction

    def transformed(self, H: np.ndarray) -> Line:
        """A mesma reta levada por uma homografia (retas continuam retas)."""
        a, b = apply_homography(H, [self.point, self.point + 100.0 * self.direction])
        return Line.through(a, b)


def intersect(l1: Line, l2: Line) -> np.ndarray | None:
    """Interseção de duas retas (None se paralelas)."""
    d1, d2 = l1.direction, l2.direction
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return None
    diff = l2.point - l1.point
    t = (diff[0] * d2[1] - diff[1] * d2[0]) / den
    return l1.point + t * d1


def undirected_angle_deg(d1, d2) -> float:
    """Ângulo entre direções sem sentido (0..90°)."""
    c = abs(float(np.dot(normalize(d1), normalize(d2))))
    return float(np.degrees(np.arccos(min(1.0, c))))


@dataclass(frozen=True)
class AxisFrame:
    """Quadro local do braço: origem no eixo, ``u`` ao longo do braço, ``n`` através."""

    origin: np.ndarray
    u: np.ndarray

    @staticmethod
    def make(origin, direction) -> AxisFrame:
        return AxisFrame(np.asarray(origin, np.float64), normalize(direction))

    @property
    def n(self) -> np.ndarray:
        return perp(self.u)

    @property
    def line(self) -> Line:
        return Line(self.origin, self.u)

    def to_local(self, pts) -> np.ndarray:
        """Imagem -> (s, t)."""
        d = as_points(pts) - self.origin
        return np.stack([d @ self.u, d @ self.n], axis=1)

    def to_image(self, st) -> np.ndarray:
        """(s, t) -> imagem."""
        st = as_points(st)
        return self.origin + st[:, :1] * self.u + st[:, 1:2] * self.n

    def flipped(self) -> AxisFrame:
        """Mesmo eixo com o sentido de s invertido (t também inverte, mantendo a orientação)."""
        return AxisFrame(self.origin, -self.u)

    def shifted(self, ds: float) -> AxisFrame:
        """Move a origem ao longo do eixo."""
        return AxisFrame(self.origin + ds * self.u, self.u)


# ==================================================================================================
# tipos do braço
# ==================================================================================================

PoseStatus = Literal["ok", "degradado", "perdido"]


@dataclass
class NeckDetection:
    """Braço encontrado num quadro (coordenadas desse quadro)."""

    frame_index: int
    mask: np.ndarray | None  # uint8 255 = braço (e cabeça), mesmo tamanho do quadro
    axis: AxisFrame  # origem no eixo; u ao longo do braço
    edge_a: Line  # borda do lado t < 0 (em cima, para o braço horizontal)
    edge_b: Line  # borda do lado t > 0
    extent: tuple[float, float]  # [s_min, s_max] do trecho estreito (braço) ao longo de ``axis``
    nut_side_hint: int | None = None  # +1 se a cabeça está no sentido +u, -1 se no -u, None se incerto
    confidence: float = 0.0
    latency_ms: float = 0.0
    full_mask: np.ndarray | None = None  # máscara bruta do segmentador (violão inteiro, se for o caso)
    source: str = ""

    def width_at(self, s: float) -> float:
        """Largura do braço (distância entre as bordas) na posição s do eixo."""
        p = self.axis.to_image([[s, 0.0]])[0]
        return float(abs(self.edge_a.signed_distance(p)[0]) + abs(self.edge_b.signed_distance(p)[0]))

    def quad(self) -> np.ndarray:
        """Quadrilátero (4, 2) do braço: bordas cortadas nas extremidades de ``extent``."""

        pts = []
        for s, edge in ((self.extent[0], self.edge_a), (self.extent[1], self.edge_a),
                        (self.extent[1], self.edge_b), (self.extent[0], self.edge_b)):
            cut = Line(self.axis.to_image([[s, 0.0]])[0], self.axis.n)
            p = intersect(cut, edge)
            pts.append(p if p is not None else cut.point)
        return np.array(pts)


@dataclass
class Keyframe:
    """Quadro de referência K do braço."""

    frame_index: int
    gray: np.ndarray  # imagem em tons de cinza de K
    detection: NeckDetection  # geometria em coordenadas de K
    track_mask: np.ndarray  # uint8: região rastreável em K (braço + cabeça + junção)
    orb_keypoints: tuple = ()  # para relocalização (T2.7)
    orb_descriptors: np.ndarray | None = None


@dataclass
class NeckPose:
    """Pose do braço no quadro atual."""

    frame_index: int
    H: np.ndarray | None  # 3x3, K -> quadro atual
    quality: float = 0.0
    inliers: int = 0
    reproj_rmse: float = float("inf")
    status: PoseStatus = "perdido"
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.H is not None and self.status != "perdido"
