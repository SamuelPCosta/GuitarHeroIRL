"""Gravação de sessões (T1.4, RF-03).

Uma sessão gravada é uma pasta com:
    video.avi | video.mkv   vídeo (MJPG de alta qualidade, ou FFV1 sem perdas)
    timestamps.json         {"frames": [{"index": i, "ts": t}, ...]}  (ts relativo ao 1º quadro, s)
    camera.json             propriedades pedidas/aplicadas e a configuração da câmera
    meta.json               data, duração, número de quadros, codec

A escrita acontece numa thread própria para não atrasar o laço principal.

Uso (CLI):
    python -m ghirl.capture.recorder --out data/rec/teste --seconds 20
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import queue
import threading
import time
from pathlib import Path

import cv2

from ghirl.capture.base import Frame

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


def _main() -> int:
    from ghirl.app.config import add_config_args, config_from_args
    from ghirl.capture.camera import CameraSource

    parser = argparse.ArgumentParser(description="Grava uma sessão da webcam para replay.")
    add_config_args(parser)
    parser.add_argument("--out", required=True, help="pasta de saída (ex.: data/rec/teste)")
    parser.add_argument("--seconds", type=float, default=0, help="duração (0 = até apertar ESC/Q)")
    parser.add_argument("--codec", choices=list(CODECS), default="mjpg")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = config_from_args(args)

    with CameraSource(cfg.camera) as cam:
        first = cam.read_latest(timeout=5.0)
        if first is None:
            raise RecorderError("a câmera não entregou quadros")
        h, w = first.image.shape[:2]
        info = {"config": dataclasses.asdict(cfg.camera),
                "requested": cam.settings.requested, "actual": cam.settings.actual}
        rec = Recorder(args.out, cfg.camera.fps, (w, h), args.codec, camera_info=info)
        t_end = time.perf_counter() + args.seconds if args.seconds > 0 else float("inf")
        print(f"Gravando em {Path(args.out).resolve()} ... (ESC ou Q para parar)")
        frame = first
        try:
            while time.perf_counter() < t_end:
                rec.write(frame)
                if not args.no_preview:
                    view = cv2.flip(frame.image, 1) if cfg.display.mirror else frame.image.copy()
                    cv2.circle(view, (30, 30), 12, (0, 0, 255), -1)
                    cv2.putText(view, f"REC {frame.ts - first.ts:5.1f}s  {cam.fps_measured:4.1f} FPS",
                                (52, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.imshow("Gravando", view)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                nxt = cam.read_latest(timeout=1.0)
                if nxt is not None:
                    frame = nxt
        finally:
            out = rec.close()
            cv2.destroyAllWindows()
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    print(f"Gravado: {meta['frames']} quadros, {meta['duration_s']:.1f} s, {meta['fps_effective']:.1f} FPS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
