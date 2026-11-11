"""Testes da captura em thread (T1.3) com uma câmera falsa."""

import threading
import time

import cv2
import numpy as np
import pytest

from ghirl.app.config import CameraConfig
from ghirl.capture.camera import CameraError, CameraSource, FpsMeter, apply_settings, fourcc_to_str


class FakeCap:
    """Imita cv2.VideoCapture: entrega quadros a ``fps`` com o índice gravado no pixel (0,0)."""

    def __init__(self, fps=200.0, fail_after=None, opened=True, accept=True):
        self.fps = fps
        self.n = 0
        self.fail_after = fail_after
        self.opened = opened
        self.accept = accept
        self.props = {}
        self.released = False

    def isOpened(self):  # noqa: N802 - API do OpenCV
        return self.opened

    def set(self, prop, value):
        if self.accept:
            self.props[prop] = value
        return self.accept

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def read(self):
        if self.fail_after is not None and self.n >= self.fail_after:
            return False, None
        time.sleep(1.0 / self.fps)
        img = np.zeros((4, 4, 3), np.uint8)
        img[0, 0, 0] = self.n % 256
        self.n += 1
        return True, img

    def release(self):
        self.released = True


def make_source(cap):
    return CameraSource(CameraConfig(), cap_factory=lambda index, api: cap)


def test_read_latest_returns_increasing_unique_frames():
    cap = FakeCap(fps=200)
    with make_source(cap) as src:
        frames = [src.read_latest(timeout=1.0) for _ in range(20)]
    idx = [f.index for f in frames]
    assert all(b > a for a, b in zip(idx, idx[1:], strict=False))
    ts = [f.ts for f in frames]
    assert all(b > a for a, b in zip(ts, ts[1:], strict=False))
    assert cap.released


def test_slow_consumer_skips_old_frames():
    cap = FakeCap(fps=200)
    with make_source(cap) as src:
        src.read_latest()
        time.sleep(0.1)  # ~20 quadros chegam enquanto o consumidor "processa"
        f = src.read_latest()
        stats = src.stats
    assert f.index >= 10  # pegou o mais recente, não o seguinte
    assert stats["dropped"] >= 10


def test_fps_measured_close_to_source_rate():
    cap = FakeCap(fps=100)
    with make_source(cap) as src:
        time.sleep(1.0)
        fps = src.fps_measured
    assert 60 < fps <= 110  # sleep no Windows é impreciso; só a ordem de grandeza


def test_timeout_returns_none_when_no_new_frame():
    cap = FakeCap(fps=2)
    with make_source(cap) as src:
        src.read_latest(timeout=2.0)
        t0 = time.perf_counter()
        assert src.read_latest(timeout=0.05) is None
        assert time.perf_counter() - t0 < 0.4


def test_camera_failure_raises():
    cap = FakeCap(fps=1000, fail_after=3)
    src = make_source(cap)
    src.start()
    try:
        with pytest.raises(CameraError):
            for _ in range(200):
                src.read_latest(timeout=0.2)
    finally:
        src.stop()


def test_open_failure_raises():
    with pytest.raises(CameraError, match="não foi possível abrir"):
        make_source(FakeCap(opened=False)).start()


def test_stop_unblocks_waiting_reader():
    cap = FakeCap(fps=1)
    src = make_source(cap)
    src.start()
    src.read_latest(timeout=2.0)
    result = {}

    def reader():
        result["frame"] = src.read_latest(timeout=None)

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.05)
    src.stop()
    t.join(timeout=3.0)
    assert not t.is_alive()


def test_apply_settings_reports_rejections():
    cfg = CameraConfig()
    s = apply_settings(FakeCap(accept=False), cfg)
    assert not s.accepted["width"]
    assert any("width" in w for w in s.warnings())
    s_ok = apply_settings(FakeCap(accept=True), cfg)
    assert s_ok.accepted["width"] and s_ok.accepted["fourcc"]
    assert s_ok.requested["exposure"] == cfg.exposure


def test_fourcc_roundtrip():
    assert fourcc_to_str(cv2.VideoWriter.fourcc(*"MJPG")) == "MJPG"
    assert fourcc_to_str(0) == "?"


def test_fps_meter_window():
    m = FpsMeter(window_s=1.0)
    for i in range(31):
        m.tick(i / 30)
    assert m.fps == pytest.approx(30, rel=1e-6)
