"""Fontes de vídeo: webcam em thread, replay de sessões gravadas e gravador.

``Frame.image`` está em BGR e nunca espelhado. O pipeline mede tempo por ``Frame.ts``.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import cv2
import numpy as np

from ghirl.config import CameraConfig

# ==================================================================================================
# tipos
# ==================================================================================================

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


# ==================================================================================================
# webcam
# ==================================================================================================

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
        self.warmup_frames = 2 if cap_factory is None else 0
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
        warmup = self.warmup_frames
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
            if warmup > 0:  # os primeiros quadros vêm velhos do buffer do driver
                warmup -= 1
                continue
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


# ==================================================================================================
# replay
# ==================================================================================================

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


# ==================================================================================================
# gravação
# ==================================================================================================

log = logging.getLogger(__name__)

CODECS = {
    # nome: (fourcc, extensão)
    "mjpg": ("MJPG", ".avi"),
    "ffv1": ("FFV1", ".mkv"),
}


class RecorderError(RuntimeError):
    pass


class Recorder:
    def __init__(self, out_dir: str | Path, fps: float, size: tuple[int, int], codec: str = "mjpg",
                 jpeg_quality: int = 95, camera_info: dict | None = None):
        if codec not in CODECS:
            raise RecorderError(f"codec desconhecido {codec!r}; opções: {list(CODECS)}")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        fourcc, ext = CODECS[codec]
        self.codec = codec
        self.video_path = self.out_dir / f"video{ext}"
        self.size = size  # (largura, altura)
        self.fps = fps
        self.camera_info = camera_info or {}
        self._writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter.fourcc(*fourcc), fps, size)
        if not self._writer.isOpened():
            raise RecorderError(f"não foi possível abrir o gravador {codec} em {self.video_path}")
        if codec == "mjpg":
            self._writer.set(cv2.VIDEOWRITER_PROP_QUALITY, jpeg_quality)
        self._queue: queue.Queue[Frame | None] = queue.Queue(maxsize=120)
        self._ts: list[dict] = []
        self._t0: float | None = None
        self._dropped = 0
        self._started_at = dt.datetime.now().isoformat(timespec="seconds")
        self._thread = threading.Thread(target=self._loop, name="Recorder", daemon=True)
        self._thread.start()
        self._closed = False

    def write(self, frame: Frame) -> None:
        h, w = frame.image.shape[:2]
        if (w, h) != self.size:
            raise RecorderError(f"quadro {w}x{h} difere do tamanho do gravador {self.size}")
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self._dropped += 1  # disco lento: perde o quadro em vez de travar o laço

    def _loop(self) -> None:
        while True:
            frame = self._queue.get()
            if frame is None:
                break
            if self._t0 is None:
                self._t0 = frame.ts
            self._writer.write(frame.image)
            self._ts.append({"index": len(self._ts), "source_index": frame.index,
                             "ts": round(frame.ts - self._t0, 6)})

    def close(self) -> Path:
        if self._closed:
            return self.out_dir
        self._closed = True
        self._queue.put(None)
        self._thread.join()
        self._writer.release()
        n = len(self._ts)
        duration = self._ts[-1]["ts"] if n else 0.0
        (self.out_dir / "timestamps.json").write_text(json.dumps({"frames": self._ts}), encoding="utf-8")
        (self.out_dir / "camera.json").write_text(
            json.dumps(self.camera_info, indent=2, default=str), encoding="utf-8")
        meta = {"started_at": self._started_at, "frames": n, "duration_s": duration,
                "fps_nominal": self.fps, "fps_effective": (n - 1) / duration if duration > 0 else 0.0,
                "codec": self.codec, "video": self.video_path.name, "size": list(self.size),
                "dropped_by_recorder": self._dropped}
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if self._dropped:
            log.warning("gravador descartou %d quadros (disco lento?)", self._dropped)
        return self.out_dir

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
