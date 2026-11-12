"""Janela Qt: vídeo à esquerda, painel de controle à direita.

O painel mostra o estado e tem os botões da calibração (Capturar / Desfazer / Recalibrar) e do
exercício. De longe, use os gestos: fechar e abrir a mão = avançar; mão fechada 3 s = recalibrar.
Teclas: espaço = avançar, T = trastes e notas, D = depuração, Esc = sair.
O vídeo é processado no laço da interface (um QTimer).
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ghirl.app import Pipeline
from ghirl.config import Config

STATE_TEXT = {
    "sem_braco": ("Sem detecção do braço", "#bbbbbb"),
    "procurando": ("Procurando o violão…", "#ffd166"),
    "perdido": ("Violão perdido — volte à posição", "#ef476f"),
    "calibrando": ("Calibração", "#4cc9f0"),
    "sem_exercicio": ("Escolha um exercício", "#bbbbbb"),
    "jogando": ("Tocando", "#06d6a0"),
    "fim": ("Exercício concluído!", "#06d6a0"),
}
STRING_NAMES = {1: "Mi agudo", 2: "Si", 3: "Sol", 4: "Ré", 5: "Lá", 6: "Mi grave"}


class VideoLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(480, 270)
        self.setStyleSheet("background:#111;")

    def show_bgr(self, img: np.ndarray) -> None:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg).scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                      Qt.TransformationMode.SmoothTransformation))


class MainWindow(QMainWindow):
    def __init__(self, cfg: Config, source, pipe: Pipeline, recorder=None, max_frames: int | None = None):
        super().__init__()
        self.cfg, self.source, self.pipe, self.recorder, self.max_frames = cfg, source, pipe, recorder, max_frames
        self.setWindowTitle("Guitar Hero IRL")
        self.video = VideoLabel()

        big = QFont()
        big.setPointSize(20)
        big.setBold(True)
        self.state_label = QLabel()
        self.state_label.setFont(big)

        # calibração
        self.calib_text = QLabel()
        self.calib_text.setWordWrap(True)
        self.calib_hint = QLabel()
        self.calib_hint.setWordWrap(True)
        self.btn_capture = QPushButton("Capturar pestana")
        self.btn_capture.setMinimumHeight(48)
        self.btn_capture.clicked.connect(self.on_capture)
        self.btn_undo = QPushButton("Desfazer")
        self.btn_undo.clicked.connect(self.pipe.undo_capture)
        self.btn_recal = QPushButton("Recalibrar")
        self.btn_recal.clicked.connect(self.pipe.recalibrate)
        calib_box = QGroupBox("Calibração")
        cl = QVBoxLayout(calib_box)
        cl.addWidget(self.calib_text)
        cl.addWidget(self.calib_hint)
        cl.addWidget(self.btn_capture)
        row = QHBoxLayout()
        row.addWidget(self.btn_undo)
        row.addWidget(self.btn_recal)
        cl.addLayout(row)

        # exercício
        self.combo = QComboBox()
        for key in pipe.exercises:
            self.combo.addItem(key, key)
        if pipe.game is not None:
            self.combo.setCurrentIndex(max(0, self.combo.findData(pipe.game.exercise.key)))
        self.combo.currentIndexChanged.connect(lambda _i: self.pipe.load_exercise(self.combo.currentData()))
        self.note_label = QLabel()
        self.note_label.setFont(big)
        self.progress = QProgressBar()
        self.time_label = QLabel()
        btn_restart = QPushButton("Reiniciar exercício")
        btn_restart.clicked.connect(self.pipe.restart_exercise)
        ex_box = QGroupBox("Exercício")
        el = QVBoxLayout(ex_box)
        el.addWidget(self.combo)
        el.addWidget(self.note_label)
        el.addWidget(self.progress)
        el.addWidget(self.time_label)
        el.addWidget(btn_restart)

        # opções
        self.chk_mirror = QCheckBox("Espelhar vídeo")
        self.chk_mirror.setChecked(cfg.display.mirror)
        self.chk_frets = QCheckBox("Trastes e notas do exercício  [T]")
        self.chk_frets.setChecked(pipe.show_frets)
        self.chk_frets.toggled.connect(lambda v: setattr(self.pipe, "show_frets", v))
        self.chk_debug = QCheckBox("Depuração (pontos, mãos, trastes achados)  [D]")
        self.chk_debug.setChecked(pipe.debug)
        self.chk_debug.toggled.connect(lambda v: setattr(self.pipe, "debug", v))
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.gesture_label = QLabel()
        self.gesture_label.setWordWrap(True)
        self.gesture_label.setStyleSheet("color:#4cc9f0;")
        help_label = QLabel("Gestos: <b>fechar e abrir a mão</b> = avançar · <b>mão fechada 3 s</b> = recalibrar")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color:#888;")
        self.stats = QLabel()
        self.stats.setStyleSheet("color:#888; font-family:Consolas, monospace;")

        panel = QWidget()
        panel.setFixedWidth(400)
        pl = QVBoxLayout(panel)
        pl.addWidget(self.state_label)
        pl.addWidget(self.message)
        pl.addWidget(self.gesture_label)
        pl.addWidget(calib_box)
        pl.addWidget(ex_box)
        pl.addWidget(self.chk_mirror)
        pl.addWidget(self.chk_frets)
        pl.addWidget(self.chk_debug)
        pl.addWidget(help_label)
        pl.addStretch(1)
        pl.addWidget(self.stats)

        central = QWidget()
        lay = QHBoxLayout(central)
        lay.addWidget(self.video, 1)
        lay.addWidget(panel)
        self.setCentralWidget(central)
        self.resize(1400, 760)

        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.on_capture)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self.close)
        QShortcut(QKeySequence(Qt.Key.Key_T), self, activated=self.chk_frets.toggle)
        QShortcut(QKeySequence(Qt.Key.Key_D), self, activated=self.chk_debug.toggle)
        self.btn_capture.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # espaço não "clica" o botão focado duas vezes

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(0)

    # ------------------------------------------------------------------ ações

    def on_capture(self) -> None:
        self.pipe.advance()

    # ------------------------------------------------------------------ laço

    def tick(self) -> None:
        frame = self.source.read_latest(timeout=0.05)
        if frame is None:
            if getattr(self.source, "finished", False):
                self.timer.stop()
            return
        img = self.pipe.process(frame)
        if self.recorder is not None:
            self.recorder.write(frame)
        self.video.show_bgr(cv2.flip(img, 1) if self.chk_mirror.isChecked() else img)
        self.refresh_panel()
        if self.max_frames is not None and self.pipe.frames >= self.max_frames:
            self.close()

    def refresh_panel(self) -> None:
        p = self.pipe
        state = p.state
        text, color = STATE_TEXT.get(state, (state, "#fff"))
        self.state_label.setText(text)
        self.state_label.setStyleSheet(f"color:{color};")
        self.message.setText(p.message)
        gs = p.gestures
        if gs is not None and gs.hold_progress > 0:
            held = gs.hold_progress * gs.reset_hold_s
            self.gesture_label.setText(f"✊ mão fechada {held:.1f} / {gs.reset_hold_s:.0f} s")
        else:
            self.gesture_label.setText(p.last_gesture)

        cal = p.calib
        calibrating = state == "calibrando" and cal is not None
        self.btn_capture.setEnabled(calibrating)
        self.btn_undo.setEnabled(calibrating and bool(cal.measures))
        if calibrating:
            step = len(cal.measures) + 1
            self.calib_text.setText(f"Etapa {step}/{len(cal.frets)}: faça PESTANA na <b>casa {cal.current_fret}</b> "
                                    "e feche e abra a outra mão (ou clique em Capturar / espaço).")
            self.calib_hint.setText(cal.hint)
            ok = cal.barre_ok
            self.calib_hint.setStyleSheet(f"color:{'#06d6a0' if ok else '#ffa94d'};")
            self.btn_capture.setStyleSheet("background:#2b9348; color:white; font-weight:bold;" if ok else "")
        elif p.model is not None:
            self.calib_text.setText("Calibrado ✔")
            self.calib_hint.setText("")
            self.btn_capture.setStyleSheet("")
        else:
            self.calib_text.setText("Aguardando o violão…")
            self.calib_hint.setText("")

        g = p.game
        if g is not None:
            self.progress.setMaximum(g.total)
            self.progress.setValue(g.index)
            self.progress.setFormat(f"{g.index}/{g.total}")
            if g.current is not None:
                n = g.current
                self.note_label.setText(f"Corda {n.string} ({STRING_NAMES[n.string]}) · Casa {n.fret}")
            else:
                self.note_label.setText("Fim! 🎸")
            self.time_label.setText(f"tempo: {g.elapsed(p.ts):.1f} s")
        t = p.times
        seg = p.neck.last_detection_ms if p.neck is not None else 0
        self.stats.setText(f"{p.fps.fps:4.1f} FPS  câmera {self.source.fps_measured:4.1f}\n"
                           f"mãos {t.get('maos', 0):4.0f} ms  braço {t.get('braco', 0):4.0f} ms\n"
                           f"segmentação {seg:4.0f} ms")


def run_window(cfg: Config, source, pipe: Pipeline, recorder=None, max_frames: int | None = None) -> None:
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    font = app.font()
    font.setPointSize(12)  # para ler de longe
    app.setFont(font)
    win = MainWindow(cfg, source, pipe, recorder, max_frames)
    win.show()
    app.exec()
