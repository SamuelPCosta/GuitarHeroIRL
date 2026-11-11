"""Worker assíncrono para localizadores pesados (T2.4; RNF-06).

O laço principal chama ``submit(frame)`` (não bloqueia; o pedido mais recente substitui o anterior
ainda não atendido) e ``poll()`` (devolve o resultado mais recente, uma vez). O modelo é criado
DENTRO da thread, para que o carregamento (segundos) não trave a janela.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from ghirl.capture.base import Frame
from ghirl.neck.base import NeckDetection, NeckHint

log = logging.getLogger(__name__)


@dataclass
class SegResponse:
    frame: Frame  # o quadro que foi segmentado (a detecção está nas coordenadas dele)
    detection: NeckDetection | None
    latency_ms: float
    error: str | None = None


class SegWorker:
    def __init__(self, locator_factory: Callable[[], object]):
        self._factory = locator_factory
        self._cond = threading.Condition()
        self._pending: tuple[Frame, NeckHint | None] | None = None
        self._result: SegResponse | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self.ready = False  # modelo carregado
        self.busy = False
        self.load_error: str | None = None
        self.calls = 0

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="SegWorker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def submit(self, frame: Frame, hint: NeckHint | None = None) -> None:
        with self._cond:
            self._pending = (frame, hint)
            self._cond.notify_all()

    def poll(self) -> SegResponse | None:
        with self._cond:
            r, self._result = self._result, None
            return r

    @property
    def idle(self) -> bool:
        """Sem pedido pendente nem em processamento (bom momento para pedir de novo)."""
        with self._cond:
            return self.ready and not self.busy and self._pending is None

    def _loop(self) -> None:
        try:
            locator = self._factory()
            self.ready = True
            log.info("SegWorker: modelo carregado")
        except Exception as exc:  # noqa: BLE001 - reportado na interface
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.error("SegWorker: falha ao carregar o modelo: %s", self.load_error)
            return
        while True:
            with self._cond:
                while self._running and self._pending is None:
                    self._cond.wait()
                if not self._running:
                    return
                frame, hint = self._pending
                self._pending = None
                self.busy = True
            t0 = time.perf_counter()
            err = None
            det = None
            try:
                det = locator.detect(frame.image, hint, frame_index=frame.index)
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                log.exception("SegWorker: erro na detecção")
            resp = SegResponse(frame, det, (time.perf_counter() - t0) * 1000.0, err)
            with self._cond:
                self._result = resp
                self.busy = False
                self.calls += 1
