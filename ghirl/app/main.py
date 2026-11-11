"""Laço principal do Guitar Hero IRL (T1.5; design §6).

Uso:
    python -m ghirl.app.main                         # webcam
    python -m ghirl.app.main --replay data/rec/x     # sessão gravada (tempo real)
    python -m ghirl.app.main --replay data/rec/x --replay-mode sequential --no-window --max-frames 300
    python -m ghirl.app.main --record data/rec/y     # usa a webcam e grava a sessão

Teclas: ESC/Q sai · D depuração (tempos por estágio) · M espelho · H HUD
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from ghirl.app.config import Config, add_config_args, config_from_args
from ghirl.capture.base import Frame, FrameSource
from ghirl.capture.camera import CameraSource, FpsMeter
from ghirl.capture.recorder import Recorder
from ghirl.capture.replay import ReplaySource
from ghirl.render.hud import draw_panel, hud_lines
from ghirl.render.overlay import draw_fixed_grid
from ghirl.utils.jsonl import JsonlLogger
from ghirl.utils.profiler import Profiler

log = logging.getLogger("ghirl")

KEY_ESC = 27


class App:
    def __init__(self, cfg: Config, source: FrameSource, logger: JsonlLogger | None = None,
                 recorder: Recorder | None = None, show: bool = True):
        self.cfg = cfg
        self.source = source
        self.logger = logger
        self.recorder = recorder
        self.show = show
        self.mirror = cfg.display.mirror
        self.debug = cfg.display.debug
        self.show_hud = True
        self.state = "SEM_CALIBRACAO"  # a máquina de estados real chega no Marco 3
        self.profiler = Profiler()
        self.proc_fps = FpsMeter()
        self.frames = 0
        self._running = True

    # ------------------------------------------------------------------ um quadro

    def process(self, frame: Frame) -> np.ndarray:
        """Processa um quadro e devolve a imagem a exibir (já espelhada, se for o caso)."""
        prof = self.profiler
        prof.begin_frame()

        with prof.stage("overlay"):
            # Overlay ancorado no mundo: sempre sobre a imagem NÃO espelhada.
            annotated = frame.image.copy()
            draw_fixed_grid(annotated)

        with prof.stage("render"):
            view = cv2.flip(annotated, 1) if self.mirror else annotated
            if self.show_hud:
                lines = hud_lines(self.state, self.source.fps_measured, self.proc_fps.fps,
                                  prof.last, self.debug)
                draw_panel(view, lines)

        if self.recorder is not None:
            with prof.stage("gravacao"):
                self.recorder.write(frame)

        times = prof.end_frame()
        self.proc_fps.tick(time.perf_counter())  # vazão real do laço (relógio de parede)
        if self.logger is not None:
            self.logger.write({"frame": frame.index, "ts": round(frame.ts, 6), "state": self.state,
                               "t_ms": {k: round(v, 3) for k, v in times.items()}})
        self.frames += 1
        return view

    # ------------------------------------------------------------------ teclado

    def handle_key(self, key: int) -> None:
        if key < 0:
            return
        k = key & 0xFF
        if k in (KEY_ESC, ord("q"), ord("Q")):
            self._running = False
        elif k in (ord("d"), ord("D")):
            self.debug = not self.debug
        elif k in (ord("m"), ord("M")):
            self.mirror = not self.mirror
        elif k in (ord("h"), ord("H")):
            self.show_hud = not self.show_hud

    # ------------------------------------------------------------------ laço

    def run(self, max_frames: int | None = None) -> int:
        win = self.cfg.display.window_name
        if self.show:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        try:
            while self._running and (max_frames is None or self.frames < max_frames):
                frame = self.source.read_latest(timeout=1.0)
                if frame is None:
                    if getattr(self.source, "finished", False):
                        break
                    continue
                view = self.process(frame)
                if self.show:
                    cv2.imshow(win, view)
                    self.handle_key(cv2.waitKey(1))
                    if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                        break
        finally:
            if self.show:
                cv2.destroyAllWindows()
        return self.frames


def build_source(cfg: Config, replay: str | None, replay_mode: str, loop: bool) -> FrameSource:
    if replay:
        return ReplaySource(replay, mode=replay_mode, loop=loop)
    return CameraSource(cfg.camera)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument("--replay", help="pasta de sessão gravada ou arquivo de vídeo")
    parser.add_argument("--replay-mode", choices=["realtime", "sequential"], default="realtime")
    parser.add_argument("--loop", action="store_true", help="repete o replay")
    parser.add_argument("--record", help="grava a sessão nesta pasta (só com webcam)")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = config_from_args(args)

    source = build_source(cfg, args.replay, args.replay_mode, args.loop)
    source.start()
    logger = JsonlLogger.in_dir(cfg.log.dir) if cfg.log.jsonl else None
    recorder = None
    try:
        if args.record:
            if args.replay:
                parser.error("--record só funciona com a webcam")
            first = source.read_latest(timeout=5.0)
            if first is None:
                raise RuntimeError("a câmera não entregou quadros")
            h, w = first.image.shape[:2]
            info = {"config": dataclasses.asdict(cfg.camera)}
            if isinstance(source, CameraSource) and source.settings is not None:
                info |= {"requested": source.settings.requested, "actual": source.settings.actual}
            recorder = Recorder(args.record, cfg.camera.fps, (w, h), camera_info=info)
        app = App(cfg, source, logger=logger, recorder=recorder, show=not args.no_window)
        n = app.run(max_frames=args.max_frames)
    finally:
        source.stop()
        if recorder is not None:
            recorder.close()
        if logger is not None:
            logger.close()

    print(f"\n{n} quadros processados")
    print(app.profiler.report())
    if isinstance(source, CameraSource):
        print(f"câmera: {source.stats}")
    if logger is not None:
        print(f"log: {Path(logger.path).resolve()}")
    if recorder is not None:
        print(f"gravação: {Path(args.record).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
