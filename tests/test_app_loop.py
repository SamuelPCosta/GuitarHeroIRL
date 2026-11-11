"""Testes do laço principal, overlay, HUD, profiler e log (T1.5, T1.6)."""

import time

import cv2
import numpy as np
import pytest

from ghirl.app.config import load_config
from ghirl.app.main import App, main
from ghirl.capture.base import Frame
from ghirl.capture.recorder import Recorder
from ghirl.render.overlay import draw_fixed_grid, fixed_grid_centers
from ghirl.utils.jsonl import JsonlLogger, read_jsonl
from ghirl.utils.profiler import Profiler

W, H = 320, 180


class ListSource:
    """Fonte de quadros em memória (sequencial)."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.finished = False

    def start(self):
        pass

    def stop(self):
        pass

    def read_latest(self, timeout=1.0):
        if not self._frames:
            self.finished = True
            return None
        return self._frames.pop(0)

    @property
    def fps_measured(self):
        return 30.0


def make_frames(n):
    rng = np.random.default_rng(1)
    return [Frame(image=rng.integers(0, 255, (H, W, 3), dtype=np.uint8), ts=i / 30, index=i)
            for i in range(n)]


def cfg_with(*overrides):
    return load_config(None, list(overrides))


# ---------------------------------------------------------------- overlay


def test_grid_centers_shape_and_bounds():
    c = fixed_grid_centers(1280, 720)
    assert c.shape == (6, 12, 2)
    assert c[..., 0].min() >= 0 and c[..., 0].max() <= 1280
    assert np.all(np.diff(c[0, :, 0]) > 0)  # casas da esquerda para a direita
    assert np.all(np.diff(c[:, 0, 1]) > 0)  # cordas de cima para baixo


def test_draw_fixed_grid_changes_pixels_at_centers():
    img = np.zeros((H, W, 3), np.uint8)
    draw_fixed_grid(img)
    for x, y in fixed_grid_centers(W, H).reshape(-1, 2):
        assert img[int(y), int(x)].any()


# ---------------------------------------------------------------- app


def test_process_does_not_modify_source_image():
    frames = make_frames(1)
    original = frames[0].image.copy()
    app = App(cfg_with(), ListSource(frames), show=False)
    app.process(frames[0])
    assert np.array_equal(frames[0].image, original)


def test_mirror_flips_only_display():
    f = make_frames(1)[0]
    app_plain = App(cfg_with("display.mirror=false"), ListSource([]), show=False)
    app_mirror = App(cfg_with("display.mirror=true"), ListSource([]), show=False)
    app_plain.show_hud = app_mirror.show_hud = False
    plain = app_plain.process(f)
    mirrored = app_mirror.process(f)
    # o overlay é desenhado no mundo e depois espelhado: exibição = flip(exibição sem espelho)
    assert np.array_equal(mirrored, cv2.flip(plain, 1))


def test_run_processes_all_frames_and_logs(tmp_path):
    with JsonlLogger(tmp_path / "log.jsonl") as logger:
        app = App(cfg_with(), ListSource(make_frames(25)), logger=logger, show=False)
        n = app.run()
    assert n == 25
    rows = read_jsonl(tmp_path / "log.jsonl")
    assert len(rows) == 25
    assert rows[0]["frame"] == 0 and rows[-1]["frame"] == 24
    assert {"overlay", "render", "total"} <= set(rows[0]["t_ms"])
    assert rows[3]["state"] == "SEM_CALIBRACAO"


def test_max_frames_limits_run():
    app = App(cfg_with(), ListSource(make_frames(30)), show=False)
    assert app.run(max_frames=10) == 10


def test_keys_toggle_and_quit():
    app = App(cfg_with("display.mirror=true"), ListSource([]), show=False)
    app.handle_key(ord("m"))
    app.handle_key(ord("d"))
    assert app.mirror is False and app.debug is True
    app.handle_key(-1)  # sem tecla
    app.handle_key(27)
    assert app._running is False


def test_cli_replay_headless(tmp_path, capsys):
    rec_dir = tmp_path / "rec"
    with Recorder(rec_dir, fps=30, size=(W, H)) as rec:
        for f in make_frames(15):
            rec.write(f)
    rc = main(["--replay", str(rec_dir), "--replay-mode", "sequential", "--no-window",
               "--set", f"log.dir={(tmp_path / 'logs').as_posix()}"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "15 quadros processados" in out and "total" in out
    logs = list((tmp_path / "logs").glob("*.jsonl"))
    assert len(logs) == 1 and len(read_jsonl(logs[0])) == 15


# ---------------------------------------------------------------- profiler


def test_profiler_stages_and_report():
    p = Profiler()
    for _ in range(5):
        p.begin_frame()
        with p.stage("a"):
            time.sleep(0.002)
        with p.stage("b"):
            pass
        times = p.end_frame()
    assert times["a"] >= 1.5 and times["total"] >= times["a"]
    s = p.summary()
    assert list(s) == ["a", "b", "total"] and s["a"]["n"] == 5
    assert "P95" in p.report()


def test_profiler_accumulates_repeated_stage():
    p = Profiler()
    p.begin_frame()
    for _ in range(3):
        with p.stage("x"):
            time.sleep(0.001)
    assert p.end_frame()["x"] >= 2.5


def test_profiler_end_without_begin():
    with pytest.raises(RuntimeError):
        Profiler().end_frame()


def test_jsonl_serializes_numpy(tmp_path):
    with JsonlLogger(tmp_path / "x.jsonl") as lg:
        lg.write({"a": np.float32(1.5), "b": np.arange(3)})
    assert read_jsonl(tmp_path / "x.jsonl") == [{"a": 1.5, "b": [0, 1, 2]}]
