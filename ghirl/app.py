"""Pipeline do Guitar Hero IRL (sem interface) e linha de comando.

    python -m ghirl                                  # janela Qt com a webcam  (ou: GuitarHeroIRL.pyw)
    python -m ghirl --replay data/rec/violao_01      # sessão gravada
    python -m ghirl --record data/rec/nova           # usa a webcam e grava a sessão

Fluxo: acha o violão sozinho -> calibração (pestana na casa 1 e na 12, cada uma confirmada por
"Capturar", espaço ou o gesto fechar-e-abrir a mão) -> o exercício mostra UMA nota por vez; ela vale
quando um dedo está nela e um dedo da outra mão toca a corda (para baixo). Mão fechada por 3 s
recomeça a calibração.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from ghirl.calib import Calibrator, NeckGeometry, NeckModel, transfer_calibration
from ghirl.capture import CameraSource, FpsMeter, Frame, FrameSource, Recorder, ReplaySource
from ghirl.config import Config, add_config_args, config_from_args
from ghirl.game import Exercise, Game, PluckDetector, list_exercises
from ghirl.geometry import apply_homography
from ghirl.hands import (
    FINGERTIPS,
    HAND_CONNECTIONS,
    INDEX_DIP,
    INDEX_PIP,
    INDEX_TIP,
    THUMB_TIP,
    WRIST,
    GestureDetector,
    HandTracker,
    hand_mask,
    is_fist,
)
from ghirl.tracking import PoseSmoother

log = logging.getLogger("ghirl")

# BGR por corda (1 = Mi agudo ... 6 = Mi grave)
STRING_COLORS = {1: (80, 80, 255), 2: (60, 180, 255), 3: (60, 230, 230), 4: (90, 220, 90),
                 5: (255, 170, 60), 6: (230, 90, 200)}
FLASH_S = 0.35
PLUCK_TIPS = (THUMB_TIP, INDEX_TIP, 12, 16, 20)  # polegar, indicador, médio, anelar, mínimo


def _pt(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


class Pipeline:
    """Um quadro por vez: mãos -> braço -> gestos -> calibração/jogo -> desenho. A interface só lê o estado."""

    def __init__(self, cfg: Config, hand_tracker=None, neck=None, log_dir: str | None = None):
        self.cfg = cfg
        self.hand_tracker = hand_tracker
        self.neck = neck  # NeckEstimator ou None
        self.hands: list = []
        self.H: np.ndarray | None = None
        self.calib: Calibrator | None = None
        self.model: NeckModel | None = None
        self.game: Game | None = None
        self.exercises = list_exercises(cfg.game.exercises_dir)
        g = cfg.gestures
        self.gestures = GestureDetector(g.advance_max_s, g.reset_hold_s) if g.enabled else None
        self.pluck = PluckDetector(cfg.game.pluck_direction)
        d = cfg.display
        self.smoother = PoseSmoother(d.smooth_min_cutoff, d.smooth_beta) if d.smooth else None
        self.debug = cfg.display.debug
        self.show_frets = cfg.display.show_frets
        self.message = ""  # retorno da última ação
        self.last_gesture = ""
        self.ts = 0.0
        self.frames = 0
        self.times: dict[str, float] = {}
        self.fps = FpsMeter()
        self._hit_at: tuple[float, np.ndarray] | None = None
        self._pluck_at: tuple[float, np.ndarray] | None = None  # (ts, ponta do dedo que tocou, na imagem)
        self._fist_px: np.ndarray | None = None  # pulso da mão fechada (anel de progresso)
        self._log = None
        if log_dir:
            path = Path(log_dir) / f"run_{dt.datetime.now():%Y%m%d_%H%M%S}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log = path.open("w", encoding="utf-8")
        if cfg.game.exercise in self.exercises:
            self.load_exercise(cfg.game.exercise)

    # ------------------------------------------------------------------ estado

    @property
    def state(self) -> str:
        if self.neck is None:
            return "sem_braco"
        if self.neck.status == "procurando":
            return "procurando"
        if self.neck.status == "perdido":
            return "perdido"
        if self.model is None:
            return "calibrando"
        if self.game is None:
            return "sem_exercicio"
        return "fim" if self.game.finished else "jogando"

    # ------------------------------------------------------------------ ações (botões, teclas, gestos)

    def advance(self) -> str:
        """Avançar: captura a pestana na calibração; recomeça o exercício quando ele termina."""
        if self.state == "calibrando":
            return self.capture_barre()
        if self.state == "fim":
            self.restart_exercise()
            self.message = "exercício recomeçado"
        return self.message

    def capture_barre(self) -> str:
        if self.calib is None:
            self.message = "o violão ainda não foi encontrado"
            return self.message
        err = self.calib.capture()
        if err:
            self.message = err
        elif self.calib.done:
            self.model = self.calib.model
            n = len(self.calib.frets_found)
            self.message = f"Calibrado! (ajustado por {n} trastes)" if n else "Calibrado!"
            log.info("calibrado: %s, pestanas %s, trastes %s", self.model.calibration, self.calib.measures,
                     [f for f, _ in self.calib.frets_found])
            if self.game is not None:
                self.game.restart()
        else:
            self.message = f"casa {self.calib.measures[-1][0]} capturada"
        return self.message

    def undo_capture(self) -> None:
        if self.calib is not None:
            self.calib.undo()
            self.model = None
            self.message = ""

    def recalibrate(self) -> None:
        """Recomeça a calibração do zero, com um quadro-chave novo do violão."""
        self.model = None
        self.calib = None
        if self.neck is not None and hasattr(self.neck, "reset"):
            self.neck.reset()
        self.message = "calibração recomeçada"

    def load_exercise(self, key: str) -> None:
        self.game = Game(Exercise.load(self.exercises[key]), self.cfg.game)
        self.message = ""

    def restart_exercise(self) -> None:
        if self.game is not None:
            self.game.restart()

    # ------------------------------------------------------------------ quadro

    def _sync_calibration(self) -> None:
        kf = self.neck.keyframe
        # um quadro-chave novo invalida a calibração (ela vive em coordenadas de K)
        if self.model is not None and self.model.keyframe_id != self.neck.keyframe_id:
            self.model = self._transfer(self.model)
        if self.calib is not None and (kf is None or self.calib.keyframe_id != self.neck.keyframe_id):
            self.calib = None
        if self.calib is None and kf is not None and self.neck.status == "rastreando":
            geom = NeckGeometry.from_detection(kf.detection, self.cfg.calib.bass_edge)
            self.calib = Calibrator(self.cfg.calib, geom, self.neck.keyframe_id, gray_K=kf.gray)

    def _transfer(self, model: NeckModel) -> NeckModel | None:
        """O quadro-chave mudou sozinho: leva a calibração para o novo e reajusta pelos trastes."""
        from ghirl.tracking import register_keyframes

        old, new = getattr(self.neck, "prev_keyframe", None), self.neck.keyframe
        H = register_keyframes(old, new) if old is not None and new is not None else None
        geom = NeckGeometry.from_detection(new.detection, self.cfg.calib.bass_edge) if new is not None else None
        out = transfer_calibration(model, H, geom, new.gray, self.neck.keyframe_id,
                                   self.cfg.calib.string_margin) if H is not None else None
        if out is None:
            log.info("quadro-chave mudou e a calibração não pôde ser transferida")
            self.message = "violão em posição nova: refaça a calibração"
            return None
        log.info("calibração transferida para o quadro-chave novo (%d trastes)", len(out[1]))
        self.message = f"calibração reajustada pelos trastes ({len(out[1])})"
        return out[0]

    def _over_guitar(self, hand) -> bool:
        """Mão sobre o violão. No jogo os gestos dela são ignorados (dedilhando, a mão fecha às vezes)."""
        kf = self.neck.keyframe if self.neck is not None else None
        if self.H is None or kf is None or kf.detection.full_mask is None:
            return False
        x, y = apply_homography(np.linalg.inv(self.H), [hand.landmarks_px[WRIST]])[0].astype(int)
        m = kf.detection.full_mask
        return bool(0 <= y < m.shape[0] and 0 <= x < m.shape[1] and m[y, x] > 0)

    def _gestures(self, ts: float) -> None:
        if self.gestures is None:
            return
        hands = self.hands if self.state == "calibrando" else [h for h in self.hands if not self._over_guitar(h)]
        ev = self.gestures.update(ts, hands)
        fist = next((h for h in hands if is_fist(h)), None)
        self._fist_px = fist.landmarks_px[WRIST] if fist is not None else None
        if ev == "avancar":
            self.last_gesture = "mão fechou e abriu: avançar"
            self.advance()
        elif ev == "reiniciar" and self.neck is not None and self.neck.keyframe is not None:
            self.last_gesture = "mão fechada 3 s: recalibrar"
            self.recalibrate()

    def _play(self, ts: float) -> None:
        """Dedos da mão das notas + dedilhado da outra mão -> jogo."""
        Hinv = np.linalg.inv(self.H)
        mus = [self.model.to_musical(apply_homography(Hinv, h.landmarks_px)) for h in self.hands]
        # mão que dedilha: a que está sobre o corpo (bem além da casa 12)
        med = [float(np.median(m[:, 0])) for m in mus]
        pluck_i = int(np.argmax(med)) if med and max(med) > self.cfg.game.pluck_min_fret else None
        note = self.game.current
        tips_k = mus[pluck_i][list(PLUCK_TIPS), 1] if pluck_i is not None else None
        pluck = self.pluck.update(ts, tips_k, note.string)
        if pluck:
            self._pluck_at = (ts, self.hands[pluck_i].landmarks_px[PLUCK_TIPS[self.pluck.finger]])
        tips = [m[list(FINGERTIPS)] for i, m in enumerate(mus) if i != pluck_i]
        pos = np.concatenate(tips) if tips else np.zeros((0, 2))
        if self.game.update(ts, pos, pluck):
            self._hit_at = (ts, self.model.cell_center_K(note.string, note.fret))

    def process(self, frame: Frame) -> np.ndarray:
        """Processa um quadro e devolve a imagem anotada (NÃO espelhada)."""
        t0 = time.perf_counter()
        self.ts = frame.ts
        mask = None
        if self.hand_tracker is not None:
            self.hands = self.hand_tracker.detect(frame)
            mask = hand_mask(self.hands, frame.image.shape[:2])
        t1 = time.perf_counter()
        self.H = None
        if self.neck is not None:
            pose = self.neck.update(frame, mask)
            self.H = pose.H if pose is not None and pose.ok else None
            if self.smoother is not None:  # a nota (e o acerto) usam a pose suavizada
                self.H = self.smoother(frame.ts, self.H, self.neck.keyframe, self.neck.keyframe_id)
            self._sync_calibration()
        t2 = time.perf_counter()

        self._gestures(frame.ts)
        if self.H is not None:
            if self.model is None and self.calib is not None:
                self.calib.observe(frame.ts, self.hands, self.H)
            elif self.model is not None and self.game is not None and not self.game.finished:
                self._play(frame.ts)

        img = self.draw(frame.image.copy())
        t3 = time.perf_counter()
        self.times = {"maos": (t1 - t0) * 1e3, "braco": (t2 - t1) * 1e3, "resto": (t3 - t2) * 1e3,
                      "total": (t3 - t0) * 1e3}
        self.fps.tick(t3)
        self.frames += 1
        if self._log is not None:
            rec = {"frame": frame.index, "ts": round(frame.ts, 4), "state": self.state,
                   "H": None if self.H is None else np.round(self.H, 5).tolist(),
                   "note": self.game.index if self.game else None,
                   "t_ms": {k: round(v, 2) for k, v in self.times.items()}}
            self._log.write(json.dumps(rec) + "\n")
        return img

    # ------------------------------------------------------------------ desenho (imagem não espelhada)

    def draw(self, img: np.ndarray) -> np.ndarray:
        H, kf = self.H, self.neck.keyframe if self.neck is not None else None
        if self.debug:
            self._draw_debug(img)
        if self._fist_px is not None and self.gestures is not None and self.gestures.hold_progress > 0:
            ang = int(360 * self.gestures.hold_progress)  # anel enchendo até recalibrar
            cv2.ellipse(img, _pt(self._fist_px), (45, 45), -90, 0, ang, (80, 200, 255), 6, cv2.LINE_AA)
        if H is None or kf is None:
            return img
        if self.model is None:
            quad = apply_homography(H, kf.detection.quad())
            cv2.polylines(img, [quad.astype(np.int32)], True, (0, 220, 0), 2, cv2.LINE_AA)
            self._draw_index_fingers(img)
            return img
        if self.show_frets:
            self._draw_frets_and_notes(img)
        if self.game is not None and (note := self.game.current) is not None:
            self._draw_note(img, note.string, note.fret)
        if self._hit_at is not None and self.ts - self._hit_at[0] < FLASH_S:
            c = apply_homography(H, [self._hit_at[1]])[0]
            r = int(14 + 30 * (self.ts - self._hit_at[0]) / FLASH_S)
            cv2.circle(img, _pt(c), r, (80, 255, 80), 3, cv2.LINE_AA)
        if self._pluck_at is not None and self.ts - self._pluck_at[0] < FLASH_S:
            cv2.circle(img, _pt(self._pluck_at[1]), 14, (255, 255, 255), 3, cv2.LINE_AA)
        return img

    def _note_radius(self, k: int, n: int, c: np.ndarray) -> int:
        side = apply_homography(self.H, self.model.point_K([n - 0.4], self.model.strings.w_of_string(k) + 0.1))[0]
        return int(np.clip(1.6 * np.linalg.norm(side - c), 7, 22))

    def _draw_note(self, img: np.ndarray, k: int, n: int) -> None:
        H, m = self.H, self.model
        c = apply_homography(H, [m.cell_center_K(k, n)])[0]
        seg = apply_homography(H, m.string_polyline_K(k, n - 1, n, steps=2))  # a corda dentro da casa
        cv2.line(img, _pt(seg[0]), _pt(seg[1]), STRING_COLORS[k], 2, cv2.LINE_AA)
        r = int(self._note_radius(k, n, c) * (1.0 + 0.15 * np.sin(self.ts * 8.0)))
        cv2.circle(img, _pt(c), r, STRING_COLORS[k], -1, cv2.LINE_AA)
        cv2.circle(img, _pt(c), r, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_frets_and_notes(self, img: np.ndarray, alpha: float = 0.35) -> None:
        """Trastes e todas as notas do exercício, em baixa opacidade (para conferir a calibração)."""
        H, m = self.H, self.model
        layer = img.copy()
        for n in range(13):
            s = apply_homography(H, m.fret_segment_K(n))
            cv2.line(layer, _pt(s[0]), _pt(s[1]), (255, 255, 255), 3 if n == 0 else 2, cv2.LINE_AA)
        if self.game is not None:
            for note in set(self.game.exercise.notes):
                c = apply_homography(H, [m.cell_center_K(note.string, note.fret)])[0]
                cv2.circle(layer, _pt(c), self._note_radius(note.string, note.fret, c), STRING_COLORS[note.string],
                           -1, cv2.LINE_AA)
        cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, dst=img)

    def _draw_index_fingers(self, img: np.ndarray) -> None:
        """Na calibração: indicador verde se a pestana é válida, laranja se não."""
        ok = self.calib is not None and self.calib.barre_ok
        for h in self.hands:
            pts = h.landmarks_px[[INDEX_PIP, INDEX_DIP, INDEX_TIP]]
            cv2.polylines(img, [pts.astype(np.int32)], False, (80, 255, 80) if ok else (0, 160, 255), 4,
                          cv2.LINE_AA)

    def _draw_debug(self, img: np.ndarray) -> None:
        for h in self.hands:
            p = h.landmarks_px
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, _pt(p[a]), _pt(p[b]), (200, 200, 200), 1, cv2.LINE_AA)
        if self.neck is not None and self.H is not None:
            for x, y in self.neck.tracker.points[1].astype(int):
                cv2.circle(img, (int(x), int(y)), 2, (255, 255, 0), -1)
            if self.calib is not None and self.calib.frets_found:  # trastes achados na imagem (amarelo)
                for _n, s in self.calib.frets_found:
                    b, t = self.calib.geom.edge_points_at(s)
                    seg = apply_homography(self.H, [b[0], t[0]])
                    cv2.line(img, _pt(seg[0]), _pt(seg[1]), (0, 255, 255), 1, cv2.LINE_AA)

    def close(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None


# ---------------------------------------------------------------------------------------- montagem


def build(cfg: Config, args) -> tuple[FrameSource, Pipeline, list]:
    """Cria fonte, rastreador de mãos, estimador do braço e pipeline. Devolve também o que fechar."""
    source = ReplaySource(args.replay, mode=args.replay_mode, loop=args.loop) if args.replay else CameraSource(
        cfg.camera)
    source.start()
    closers = [source.stop]
    hand_tracker = None if args.no_hands else HandTracker(cfg.hands)
    if hand_tracker is not None:
        closers.append(hand_tracker.close)
    neck = None
    if not args.no_neck:
        from ghirl.neck import NeckLocator, SegWorker
        from ghirl.tracking import NeckEstimator

        worker = SegWorker(NeckLocator)
        worker.start()
        closers.append(worker.stop)
        neck = NeckEstimator(cfg.neck, worker)
    pipe = Pipeline(cfg, hand_tracker, neck, cfg.log.dir if cfg.log.jsonl else None)
    closers.append(pipe.close)
    return source, pipe, closers


def run_headless(source: FrameSource, pipe: Pipeline, max_frames: int | None, recorder: Recorder | None = None) -> int:
    totals: list[float] = []
    while max_frames is None or pipe.frames < max_frames:
        frame = source.read_latest(timeout=1.0)
        if frame is None:
            if getattr(source, "finished", False):
                break
            continue
        pipe.process(frame)
        if recorder is not None:
            recorder.write(frame)
        totals.append(pipe.times["total"])
    if totals:
        print(f"{pipe.frames} quadros, {np.mean(totals):.1f} ms/quadro em média (P95 {np.percentile(totals, 95):.1f})"
              f", estado final: {pipe.state}")
    return pipe.frames


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ghirl", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--replay", help="pasta de sessão gravada ou arquivo de vídeo")
    p.add_argument("--replay-mode", choices=["realtime", "sequential"], default="realtime")
    p.add_argument("--loop", action="store_true", help="repete o replay")
    p.add_argument("--record", help="grava a sessão nesta pasta (só com webcam)")
    p.add_argument("--headless", action="store_true", help="sem janela (testes, medições)")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--no-hands", action="store_true", help="desliga o MediaPipe")
    p.add_argument("--no-neck", action="store_true", help="desliga a detecção do braço")
    return p


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = config_from_args(args)
    if args.record and args.replay:
        raise SystemExit("--record só funciona com a webcam")
    source, pipe, closers = build(cfg, args)
    recorder = None
    try:
        if args.record:
            first = source.read_latest(timeout=5.0)
            if first is None:
                raise RuntimeError("a câmera não entregou quadros")
            h, w = first.image.shape[:2]
            recorder = Recorder(args.record, cfg.camera.fps, (w, h))
            closers.append(recorder.close)
        if args.headless:
            run_headless(source, pipe, args.max_frames, recorder)
        else:
            from ghirl.ui import run_window

            run_window(cfg, source, pipe, recorder, args.max_frames)
    finally:
        for close in reversed(closers):
            close()
    return 0
