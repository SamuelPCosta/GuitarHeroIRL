"""Geometria 2D: homografias, retas e o quadro local do eixo do braço (T2.1; design §5).

Convenções:
- pontos são arrays (N, 2) float64 em pixels (x, y);
- uma reta é (ponto, direção unitária);
- ``AxisFrame``: s ao longo do eixo (direção ``u``), t perpendicular (direção ``n`` = ``u`` girado
  +90° no sistema da imagem, onde y cresce para baixo).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
