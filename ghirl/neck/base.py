"""Tipos e interfaces da localização/rastreamento do braço (T2.1; design §4, §5).

Fluxo: um ``NeckLocator`` (SAM, YOLO, clássico) encontra o braço num quadro e devolve uma
``NeckDetection``; dela se constrói o quadro-chave ``Keyframe`` (K). Um ``NeckTracker`` estima, a
cada quadro, a homografia ``H`` de K para o quadro atual (``NeckPose``). Toda a geometria do braço
(eixo, bordas, calibração) mora em coordenadas de K.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from ghirl.capture.base import Frame
from ghirl.utils.geometry import AxisFrame, Line

PoseStatus = Literal["ok", "degradado", "perdido"]


@dataclass
class NeckHint:
    """Dica opcional para o localizador (caixa ou pontos de prompt), em pixels."""

    box: tuple[float, float, float, float] | None = None  # x0, y0, x1, y1
    points: np.ndarray | None = None  # (N, 2) pontos positivos


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
        from ghirl.utils.geometry import intersect

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


class NeckLocator(Protocol):
    name: str
    is_heavy: bool  # True -> roda no SegWorker (assíncrono)

    def detect(self, image: np.ndarray, hint: NeckHint | None = None) -> NeckDetection | None: ...


class NeckTracker(Protocol):
    def init(self, frame: Frame, keyframe: Keyframe) -> None: ...

    def update(self, frame: Frame, exclude_mask: np.ndarray | None) -> NeckPose: ...
