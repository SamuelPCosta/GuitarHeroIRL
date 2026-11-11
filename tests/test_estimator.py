"""Testes do NeckEstimator com localizador falso (sem GPU)."""

import time

import cv2
import numpy as np
from test_feature_tracker import REGION, h_at, make_keyframe, texture

from ghirl.app.config import NeckConfig
from ghirl.capture.base import Frame
from ghirl.neck.estimator import NeckEstimator
from ghirl.neck.seg_worker import SegWorker

W, H = 1280, 720


class FakeLocator:
    name = "fake"
    is_heavy = True

    def __init__(self):
        self.det = make_keyframe(texture()).detection
        self.det.confidence = 0.9

    def detect(self, image, hint=None, frame_index=0):
        return self.det if image.std() > 3 else None  # quadro "vazio" (uniforme) -> nada


def wait_until(cond, timeout=5.0):
    t0 = time.perf_counter()
    while not cond():
        if time.perf_counter() - t0 > timeout:
            raise TimeoutError
        time.sleep(0.01)


def make_estimator(**cfg):
    worker = SegWorker(FakeLocator)
    worker.start()
    wait_until(lambda: worker.ready)
    return NeckEstimator(NeckConfig(**cfg), worker), worker


def feed_until(est, frames_iter, cond, timeout=5.0):
    t0 = time.perf_counter()
    for f in frames_iter:
        est.update(f)
        if cond():
            return True
        if time.perf_counter() - t0 > timeout:
            return False
        time.sleep(0.005)
    return False


def test_find_track_lose_relocalize_and_timeout():
    est, worker = make_estimator(lost_frames=3, relock_timeout_s=0.5)
    try:
        base = texture()
        blank = np.full_like(base, 90)
        k = iter(range(1, 10_000))

        def frames(img_fn):
            while True:
                i = next(k)
                yield Frame(img_fn(i), i / 30, i)

        # 1) procura e encontra
        assert est.status == "procurando"
        assert feed_until(est, frames(lambda i: base), lambda: est.status == "rastreando")
        kid = est.keyframe_id

        # 2) movimento: pose acompanha
        moved = cv2.warpPerspective(base, h_at(5, 40), (W, H), borderValue=(90, 90, 90))
        pose = est.update(Frame(moved, 1.0, next(k)))
        assert pose is not None and pose.status in ("ok", "degradado")
        err = np.abs(cv2.perspectiveTransform(REGION.reshape(-1, 1, 2).astype(np.float32), pose.H)
                     - cv2.perspectiveTransform(REGION.reshape(-1, 1, 2).astype(np.float32), h_at(5, 40))).mean()
        assert err < 1.5

        # 3) violão some -> perdido
        assert feed_until(est, frames(lambda i: blank), lambda: est.status == "perdido")

        # 4) volta -> relocaliza no MESMO quadro-chave (calibração preservada)
        assert feed_until(est, frames(lambda i: base), lambda: est.status == "rastreando")
        assert est.keyframe_id == kid

        # 5) some por mais que o timeout -> procura de novo (novo K)
        assert feed_until(est, frames(lambda i: blank), lambda: est.status == "perdido")
        time.sleep(0.01)
        assert feed_until(est, (Frame(blank, 100.0 + j, next(k)) for j in range(50)),
                          lambda: est.status == "procurando")
    finally:
        worker.stop()


def test_worker_is_non_blocking_and_latest_wins():
    class Slow(FakeLocator):
        def detect(self, image, hint=None, frame_index=0):
            time.sleep(0.2)
            return super().detect(image, hint, frame_index)

    worker = SegWorker(Slow)
    worker.start()
    wait_until(lambda: worker.ready)
    try:
        img = texture()
        t0 = time.perf_counter()
        for i in range(5):
            worker.submit(Frame(img, i / 30, i))
        assert time.perf_counter() - t0 < 0.05  # submit não bloqueia
        wait_until(lambda: worker.calls >= 2 or (worker.idle and worker.calls >= 1), timeout=3)
        r = None
        wait_until(lambda: worker.idle, timeout=3)
        r = worker.poll()
        assert r is not None and r.frame.index == 4  # o mais recente foi atendido por último
        assert worker.calls <= 2
    finally:
        worker.stop()


def test_worker_reports_load_error():
    def broken():
        raise RuntimeError("sem GPU")

    worker = SegWorker(broken)
    worker.start()
    wait_until(lambda: worker.load_error is not None)
    assert "sem GPU" in worker.load_error and not worker.ready
    worker.stop()
