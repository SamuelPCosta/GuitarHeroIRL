"""Tipos comuns das fontes de vídeo (câmera ao vivo e replay)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Frame:
    """Um quadro de vídeo. ``image`` está em BGR e **nunca** espelhado."""

    image: np.ndarray
    ts: float  # time.perf_counter() no momento em que o quadro foi lido da câmera
    index: int  # contador de quadros capturados pela fonte (0, 1, 2, ...)


class FrameSource(Protocol):
    def start(self) -> None: ...

    def read_latest(self, timeout: float | None = 1.0) -> Frame | None:
        """Retorna o quadro mais recente ainda não entregue (espera até ``timeout``)."""
        ...

    def stop(self) -> None: ...

    @property
    def fps_measured(self) -> float: ...
