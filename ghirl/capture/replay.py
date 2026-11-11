"""Reprodução de sessões gravadas como se fossem a câmera (T1.4, RF-03, RNF-08).

Modos:
    "sequential"  entrega TODOS os quadros, em ordem, sem esperar (determinístico; testes e avaliação)
    "realtime"    imita a câmera: respeita os tempos gravados e pula quadros se o consumidor
                  atrasar (para demos; não é determinístico)

``Frame.ts`` é sempre o carimbo **gravado** (relativo ao 1º quadro, em segundos). Todo o pipeline
deve medir tempo pelas diferenças de ``Frame.ts``, nunca pelo relógio de parede, para que o
replay sequencial seja reprodutível.
Aceita também um arquivo de vídeo comum (sem timestamps.json): ts = índice / FPS do arquivo.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

import cv2

from ghirl.capture.base import Frame
from ghirl.capture.camera import FpsMeter


class ReplayError(RuntimeError):
    pass


def _find_video(path: Path) -> Path:
    if path.is_file():
        return path
    for name in ("video.avi", "video.mkv", "video.mp4"):
        if (path / name).exists():
            return path / name
    raise ReplayError(f"nenhum vídeo encontrado em {path}")


class ReplaySource:
    def __init__(self, path: str | Path, mode: Literal["sequential", "realtime"] = "sequential",
                 loop: bool = False):
        self.path = Path(path)
        self.mode = mode
        self.loop = loop
        self.video_path = _find_video(self.path)
        self.timestamps: list[float] | None = None
        ts_file = (self.path if self.path.is_dir() else self.path.parent) / "timestamps.json"
        if self.path.is_dir() and ts_file.exists():
            data = json.loads(ts_file.read_text(encoding="utf-8"))
            self.timestamps = [f["ts"] for f in data["frames"]]
        self._cap: cv2.VideoCapture | None = None
        self._next = 0  # índice do próximo quadro a decodificar
        self._fps_file = 30.0
        self._t_start = 0.0
        self._loop_offset = 0.0
        self._meter = FpsMeter()
        self._finished = False

    # ------------------------------------------------------------------ ciclo de vida

    def start(self) -> None:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ReplayError(f"não foi possível abrir {self.video_path}")
        self._cap = cap
        self._fps_file = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._next = 0
        self._loop_offset = 0.0
        self._finished = False
        self._t_start = time.perf_counter()

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> ReplaySource:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------ leitura

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def frame_count(self) -> int:
        if self.timestamps is not None:
            return len(self.timestamps)
        if self._cap is not None:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0

    def _ts_of(self, i: int) -> float:
        if self.timestamps is not None and i < len(self.timestamps):
            return self.timestamps[i]
        return i / self._fps_file

    def _decode_next(self) -> Frame | None:
        assert self._cap is not None
        ok, image = self._cap.read()
        if not ok or (self.timestamps is not None and self._next >= len(self.timestamps)):
            if not self.loop or self._next == 0:
                self._finished = True
                return None
            # recomeça: ts continua crescendo para não confundir quem mede intervalos
            self._loop_offset += self._ts_of(self._next - 1) + 1.0 / self._fps_file
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._next = 0
            ok, image = self._cap.read()
            if not ok:
                self._finished = True
                return None
        frame = Frame(image=image, ts=self._loop_offset + self._ts_of(self._next), index=self._next)
        self._next += 1
        return frame

    def read_latest(self, timeout: float | None = 1.0) -> Frame | None:
        if self._cap is None:
            raise ReplayError("ReplaySource não foi iniciado (chame start())")
        if self._finished:
            return None
        if self.mode == "sequential":
            frame = self._decode_next()
        else:
            frame = self._read_realtime(timeout)
        if frame is not None:
            self._meter.tick(frame.ts)
        return frame

    def _read_realtime(self, timeout: float | None) -> Frame | None:
        elapsed = time.perf_counter() - self._t_start
        nxt_ts = self._loop_offset + self._ts_of(self._next)
        wait = nxt_ts - elapsed
        if wait > 0:
            if timeout is not None and wait > timeout:
                time.sleep(timeout)
                return None
            time.sleep(wait)
            elapsed = time.perf_counter() - self._t_start
        # pula quadros já "vencidos", como a câmera faz com um consumidor lento
        frame = self._decode_next()
        while frame is not None:
            following = self._loop_offset + self._ts_of(self._next)
            if following > elapsed or (self.timestamps is not None and self._next >= len(self.timestamps)):
                break
            nxt = self._decode_next()
            if nxt is None:
                break
            frame = nxt
        return frame

    @property
    def fps_measured(self) -> float:
        return self._meter.fps
