"""Exercícios (JSON), detecção do dedilhado e o jogo sequencial.

Uma nota vale quando um dedo está na casa/corda certa E um dedo da outra mão toca essa corda (atravessa
a corda para baixo, sobre o corpo do violão) quase ao mesmo tempo. Aí aparece a próxima nota.

Formato de ``exercises/<nome>.json``::

    {"nome": "...", "descricao": "...", "notas": [{"corda": 6, "casa": 5}, ...]}

corda: 1 (Mi agudo) .. 6 (Mi grave); casa: 1 .. 12.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ghirl.calib import N_FRETS
from ghirl.config import GameConfig


class ExerciseError(ValueError):
    pass


@dataclass(frozen=True)
class Note:
    string: int  # 1..6
    fret: int  # 1..12


@dataclass
class Exercise:
    name: str
    notes: list[Note]
    description: str = ""
    key: str = ""  # nome do arquivo sem .json

    @staticmethod
    def load(path: str | Path) -> Exercise:
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExerciseError(f"{path}: {exc}") from exc
        notes = []
        for i, n in enumerate(data.get("notas", [])):
            try:
                note = Note(int(n["corda"]), int(n["casa"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ExerciseError(f"{path}: nota {i} precisa de 'corda' e 'casa' inteiros") from exc
            if not (1 <= note.string <= 6 and 1 <= note.fret <= N_FRETS):
                raise ExerciseError(f"{path}: nota {i} fora do braço (corda 1–6, casa 1–{N_FRETS}): {n}")
            notes.append(note)
        if not notes:
            raise ExerciseError(f"{path}: nenhuma nota")
        return Exercise(str(data.get("nome", path.stem)), notes, str(data.get("descricao", "")), path.stem)


def list_exercises(directory: str | Path) -> dict[str, Path]:
    """{nome do arquivo sem .json: caminho}, em ordem alfabética."""
    return {p.stem: p for p in sorted(Path(directory).glob("*.json"))}


@dataclass
class Game:
    exercise: Exercise
    cfg: GameConfig = field(default_factory=GameConfig)
    index: int = 0
    hit_times: list[float] = field(default_factory=list)
    started_ts: float | None = None
    _streak: int = 0
    _on_ts: float | None = None  # último instante com o dedo firme na nota
    _pluck_ts: float | None = None  # última vez que a corda foi tocada

    @property
    def current(self) -> Note | None:
        return self.exercise.notes[self.index] if self.index < len(self.exercise.notes) else None

    @property
    def finished(self) -> bool:
        return self.current is None

    @property
    def total(self) -> int:
        return len(self.exercise.notes)

    def restart(self) -> None:
        self.index = 0
        self.hit_times.clear()
        self.started_ts = None
        self._streak = 0
        self._on_ts = self._pluck_ts = None

    def on_note(self, positions: np.ndarray) -> bool:
        """Algum ponto (n̂, k̂) está dentro da célula da nota atual (com folga)?"""
        note = self.current
        if note is None or len(positions) == 0:
            return False
        n_hat, k_hat = np.asarray(positions, np.float64).reshape(-1, 2).T
        e = self.cfg.eps_fret
        inside = (n_hat > note.fret - 1 - e) & (n_hat <= note.fret + e) & (np.abs(k_hat - note.string)
                                                                        <= self.cfg.eps_string)
        return bool(inside.any())

    def update(self, ts: float, positions: np.ndarray, pluck: bool = False) -> bool:
        """Um quadro: posições (n̂, k̂) das pontas dos dedos e se a corda foi tocada. True no acerto."""
        if self.finished:
            return False
        if self.started_ts is None:
            self.started_ts = ts
        self._streak = self._streak + 1 if self.on_note(positions) else 0
        if self._streak >= self.cfg.hold_frames:
            self._on_ts = ts
        if pluck:
            self._pluck_ts = ts
        win = self.cfg.pluck_window_s
        fingered = self._on_ts is not None and ts - self._on_ts <= win
        plucked = not self.cfg.require_pluck or (self._pluck_ts is not None and ts - self._pluck_ts <= win)
        if fingered and plucked and self._streak >= 1:
            self._streak = 0
            self._on_ts = self._pluck_ts = None
            self.hit_times.append(ts)
            self.index += 1
            return True
        return False

    def elapsed(self, ts: float) -> float:
        if self.started_ts is None:
            return 0.0
        end = self.hit_times[-1] if self.finished else ts
        return end - self.started_ts


class PluckDetector:
    """Dedilhado: alguma ponta de dedo da mão sobre o corpo atravessa a corda alvo.

    Em k̂ (corda contínua; 6 em cima, 1 embaixo), "para baixo" (em direção ao chão) é k̂ diminuindo: o dedo
    estava na corda ou acima dela (k̂ >= alvo - 0.1) há pouco e agora está abaixo (k̂ <= alvo - 0.4), com uma
    descida de pelo menos ``min_drop`` corda. Com ``direction="any"`` vale nos dois sentidos.
    """

    def __init__(self, direction: str = "down", window_s: float = 0.3, min_drop: float = 0.5,
                 cooldown_s: float = 0.1):
        self.direction, self.window_s, self.min_drop, self.cooldown_s = direction, window_s, min_drop, cooldown_s
        self._hist: list[tuple[float, np.ndarray]] = []
        self._last = -1e9
        self.finger: int | None = None  # índice do dedo que tocou por último

    def _crossed(self, past: np.ndarray, now: np.ndarray, target: float) -> np.ndarray:
        down = (past >= target - 0.1) & (now <= target - 0.4) & (past - now >= self.min_drop)
        if self.direction == "down":
            return down
        up = (past <= target + 0.1) & (now >= target + 0.4) & (now - past >= self.min_drop)
        return down | up

    def update(self, ts: float, tips_k: np.ndarray | None, target_k: int) -> bool:
        """``tips_k``: k̂ das pontas dos dedos (sempre na mesma ordem) ou None sem a mão."""
        if tips_k is None:
            self._hist.clear()
            return False
        now = np.asarray(tips_k, np.float64)
        self._hist = [(t, k) for t, k in self._hist if ts - t <= self.window_s and k.shape == now.shape]
        hit = False
        if self._hist and ts - self._last >= self.cooldown_s:
            crossed = np.zeros(now.shape, bool)
            for _, past in self._hist:
                crossed |= self._crossed(past, now, float(target_k))
            if crossed.any():
                hit = True
                self.finger = int(np.argmax(crossed))
                self._last = ts
                self._hist = []
        self._hist.append((ts, now))
        return hit
