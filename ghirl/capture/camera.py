"""Captura da webcam em thread dedicada (T1.3, RF-01, RF-02).

A thread lê quadros continuamente e guarda só o mais recente; o laço principal pega sempre o
último quadro disponível (quadros antigos são descartados, nunca enfileirados).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2

from ghirl.app.config import CameraConfig
from ghirl.capture.base import Frame

log = logging.getLogger(__name__)

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}

# Convenção do OpenCV para CAP_PROP_AUTO_EXPOSURE (V4L/DSHOW): 0.25 = manual, 0.75 = automático.
_AUTO_EXPOSURE_MANUAL = 0.25
_AUTO_EXPOSURE_AUTO = 0.75


class CameraError(RuntimeError):
    pass


def fourcc_to_str(value: float) -> str:
    v = int(value)
    if v <= 0:
        return "?"
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


@dataclass
class AppliedSettings:
    """Valores pedidos e lidos de volta do driver (para log e diagnóstico)."""

    requested: dict[str, Any] = field(default_factory=dict)
    actual: dict[str, Any] = field(default_factory=dict)
    accepted: dict[str, bool] = field(default_factory=dict)

    def warnings(self) -> list[str]:
        return [f"driver não aceitou {k}={self.requested[k]!r} (lido: {self.actual.get(k)!r})"
                for k, ok in self.accepted.items() if not ok]


def apply_settings(cap: cv2.VideoCapture, cfg: CameraConfig) -> AppliedSettings:
    """Aplica resolução/FPS/formato/exposição e registra o que o driver realmente usou.

    Chamar só uma vez, logo após abrir a câmera. Trocar a exposição várias vezes com a câmera
    já transmitindo travou a webcam de teste (quadros pretos até reconectar) — ver T1.3.
    """
    s = AppliedSettings()
    set_ok: dict[str, bool] = {}

    def req(name: str, prop: int, value: float) -> None:
        s.requested[name] = value
        set_ok[name] = bool(cap.set(prop, value))

    # Ordem importa no DSHOW: largura, altura e FPS ANTES do FOURCC. Com o FOURCC primeiro, a troca
    # de resolução volta o formato para YUY2 e a webcam cai para ~7 FPS a 720p (medido na T1.3).
    req("width", cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    req("height", cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    req("fps", cv2.CAP_PROP_FPS, cfg.fps)
    s.requested["fourcc"] = cfg.fourcc
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*cfg.fourcc))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # menos latência quando o backend respeita

    if cfg.auto_exposure:
        req("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE, _AUTO_EXPOSURE_AUTO)
    else:
        req("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE, _AUTO_EXPOSURE_MANUAL)
        if cfg.exposure is not None:
            req("exposure", cv2.CAP_PROP_EXPOSURE, cfg.exposure)
    if cfg.gain is not None:
        req("gain", cv2.CAP_PROP_GAIN, cfg.gain)

    # Leitura de volta só depois de tudo aplicado (uma propriedade pode alterar outra).
    checks = {
        "width": (cv2.CAP_PROP_FRAME_WIDTH, lambda a: int(a) == cfg.width),
        "height": (cv2.CAP_PROP_FRAME_HEIGHT, lambda a: int(a) == cfg.height),
        "fps": (cv2.CAP_PROP_FPS, lambda a: abs(a - cfg.fps) < 1),
        "auto_exposure": (cv2.CAP_PROP_AUTO_EXPOSURE, None),
        "exposure": (cv2.CAP_PROP_EXPOSURE,
                     lambda a: cfg.exposure is not None and abs(a - cfg.exposure) < 0.5),
        "gain": (cv2.CAP_PROP_GAIN, lambda a: cfg.gain is not None and abs(a - cfg.gain) < 1),
    }
    for name, (prop, check) in checks.items():
        if name not in s.requested:
            continue
        actual = cap.get(prop)
        s.actual[name] = actual
        s.accepted[name] = set_ok[name] and (check(actual) if check else True)
    s.actual["fourcc"] = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
    s.accepted["fourcc"] = s.actual["fourcc"] == cfg.fourcc
    s.actual["exposure_now"] = cap.get(cv2.CAP_PROP_EXPOSURE)
    s.actual["gain_now"] = cap.get(cv2.CAP_PROP_GAIN)
    return s


class FpsMeter:
    """FPS a partir dos carimbos de tempo dos últimos ``window_s`` segundos."""

    def __init__(self, window_s: float = 2.0):
        self.window_s = window_s
        self._ts: deque[float] = deque()

    def tick(self, ts: float) -> None:
        self._ts.append(ts)
        while self._ts and ts - self._ts[0] > self.window_s:
            self._ts.popleft()

    @property
    def fps(self) -> float:
        if len(self._ts) < 2:
            return 0.0
        span = self._ts[-1] - self._ts[0]
        return (len(self._ts) - 1) / span if span > 0 else 0.0


class CameraSource:
    """Webcam em thread; ``read_latest`` entrega sempre o quadro mais novo ainda não entregue."""

    def __init__(self, cfg: CameraConfig, cap_factory=None):
        """``cap_factory(index, api) -> objeto tipo VideoCapture`` (injeção para testes)."""
        self.cfg = cfg
        self._cap_factory = cap_factory or cv2.VideoCapture
        self.settings: AppliedSettings | None = None
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._cond = threading.Condition()
        self._running = False
        self._latest: Frame | None = None
        self._delivered_index = -1
        self._captured = 0
        self._delivered = 0
        self._fps = FpsMeter()
        self._error: Exception | None = None

    # ------------------------------------------------------------------ ciclo de vida

    def start(self) -> None:
        backend = BACKENDS[self.cfg.backend]
        cap = self._cap_factory(self.cfg.index, backend)
        if not cap.isOpened():
            raise CameraError(
                f"não foi possível abrir a câmera {self.cfg.index} (backend {self.cfg.backend})"
            )
        self.settings = apply_settings(cap, self.cfg)
        for w in self.settings.warnings():
            log.warning("câmera: %s", w)
        log.info("câmera: aplicado %s", self.settings.actual)
        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="CameraSource", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._cond:
            self._cond.notify_all()

    def __enter__(self) -> CameraSource:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------ thread de captura

    def _loop(self) -> None:
        assert self._cap is not None
        failures = 0
        while self._running:
            ok, image = self._cap.read()
            ts = time.perf_counter()
            if not ok or image is None:
                failures += 1
                if failures >= 50:
                    self._error = CameraError("a câmera parou de entregar quadros")
                    log.error("%s", self._error)
                    self._running = False
                    break
                time.sleep(0.01)
                continue
            failures = 0
            frame = Frame(image=image, ts=ts, index=self._captured)
            with self._cond:
                self._latest = frame
                self._captured += 1
                self._fps.tick(ts)
                self._cond.notify_all()
        with self._cond:
            self._cond.notify_all()

    # ------------------------------------------------------------------ consumo

    def read_latest(self, timeout: float | None = 1.0) -> Frame | None:
        deadline = None if timeout is None else time.perf_counter() + timeout
        with self._cond:
            while self._latest is None or self._latest.index == self._delivered_index:
                if self._error is not None:
                    raise self._error
                if not self._running:
                    return None
                remaining = None if deadline is None else deadline - time.perf_counter()
                if remaining is not None and remaining <= 0:
                    return None
                self._cond.wait(remaining)
            frame = self._latest
            self._delivered_index = frame.index
            self._delivered += 1
            return frame

    # ------------------------------------------------------------------ métricas

    @property
    def fps_measured(self) -> float:
        with self._cond:
            return self._fps.fps

    @property
    def stats(self) -> dict[str, int]:
        with self._cond:
            return {"captured": self._captured, "delivered": self._delivered,
                    "dropped": self._captured - self._delivered}
