"""Testes de gravação e replay (T1.4, RF-03, RNF-08)."""

import json
import time

import cv2
import numpy as np
import pytest

from ghirl.capture.base import Frame
from ghirl.capture.recorder import Recorder, RecorderError
from ghirl.capture.replay import ReplayError, ReplaySource

W, H = 160, 120


def synthetic_frames(n, dt=1 / 30, jitter=0.0, t0=1000.0):
    """Quadros com um retângulo que anda (conteúdo distinto por quadro) e ts irregulares."""
    rng = np.random.default_rng(0)
    t = t0
    out = []
    for i in range(n):
        img = np.full((H, W, 3), 40, np.uint8)
        x = (i * 5) % (W - 30)
        cv2.rectangle(img, (x, 40), (x + 30, 80), (0, 200, 255), -1)
        cv2.putText(img, str(i), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        out.append(Frame(image=img, ts=t, index=i + 7))  # índice da fonte != índice gravado
        t += dt + jitter * rng.uniform(-1, 1)
    return out


def record(tmp_path, frames, codec="mjpg"):
    out = tmp_path / f"rec_{codec}"
    with Recorder(out, fps=30, size=(W, H), codec=codec, camera_info={"teste": True}) as rec:
        for f in frames:
            rec.write(f)
    return out


def read_all(src):
    frames = []
    while (f := src.read_latest()) is not None:
        frames.append(f)
    return frames


def test_record_writes_metadata(tmp_path):
    frames = synthetic_frames(20)
    out = record(tmp_path, frames)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["frames"] == 20 and meta["codec"] == "mjpg"
    assert json.loads((out / "camera.json").read_text(encoding="utf-8")) == {"teste": True}
    ts = json.loads((out / "timestamps.json").read_text(encoding="utf-8"))["frames"]
    assert ts[0]["ts"] == 0.0 and ts[0]["source_index"] == 7


def test_replay_sequential_same_count_and_timestamps(tmp_path):
    frames = synthetic_frames(45, jitter=0.01)
    out = record(tmp_path, frames)
    with ReplaySource(out, mode="sequential") as src:
        got = read_all(src)
        assert src.finished
    assert len(got) == len(frames)
    expected_ts = [round(f.ts - frames[0].ts, 6) for f in frames]
    assert [f.ts for f in got] == expected_ts
    assert [f.index for f in got] == list(range(len(frames)))
    # MJPG tem perdas, mas pequenas
    for a, b in zip(frames, got, strict=True):
        assert np.abs(a.image.astype(int) - b.image.astype(int)).mean() < 3.0


def test_replay_is_deterministic(tmp_path):
    out = record(tmp_path, synthetic_frames(30))
    runs = []
    for _ in range(2):
        with ReplaySource(out) as src:
            runs.append([(f.ts, f.image.tobytes()) for f in read_all(src)])
    assert runs[0] == runs[1]


def test_ffv1_is_lossless_when_available(tmp_path):
    frames = synthetic_frames(10)
    try:
        out = record(tmp_path, frames, codec="ffv1")
    except RecorderError:
        pytest.skip("FFV1 indisponível neste build do OpenCV")
    with ReplaySource(out) as src:
        got = read_all(src)
    assert len(got) == 10
    for a, b in zip(frames, got, strict=True):
        assert np.array_equal(a.image, b.image)


def test_realtime_mode_follows_recorded_timing_and_skips(tmp_path):
    out = record(tmp_path, synthetic_frames(30, dt=0.02))  # 0,6 s de vídeo a 50 FPS
    with ReplaySource(out, mode="realtime") as src:
        t0 = time.perf_counter()
        first = src.read_latest()
        time.sleep(0.25)  # consumidor lento: ~12 quadros "passam"
        second = src.read_latest()
        rest = read_all(src)
        elapsed = time.perf_counter() - t0
    assert first.index == 0
    assert second.index >= 8  # pulou quadros vencidos
    idx = [first.index, second.index] + [f.index for f in rest]
    assert idx == sorted(set(idx))
    assert 0.5 < elapsed < 1.5  # respeitou aproximadamente a duração gravada


def test_loop_keeps_timestamps_increasing(tmp_path):
    out = record(tmp_path, synthetic_frames(5))
    with ReplaySource(out, loop=True) as src:
        got = [src.read_latest() for _ in range(12)]
    ts = [f.ts for f in got]
    assert all(b > a for a, b in zip(ts, ts[1:], strict=False))
    assert [f.index for f in got][:7] == [0, 1, 2, 3, 4, 0, 1]


def test_plain_video_file_without_timestamps(tmp_path):
    path = tmp_path / "plain.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 25, (W, H))
    for f in synthetic_frames(10):
        w.write(f.image)
    w.release()
    with ReplaySource(path) as src:
        got = read_all(src)
    assert len(got) == 10
    assert got[1].ts == pytest.approx(1 / 25)


def test_wrong_frame_size_rejected(tmp_path):
    with Recorder(tmp_path / "r", fps=30, size=(W, H)) as rec, pytest.raises(RecorderError):
        rec.write(Frame(image=np.zeros((10, 10, 3), np.uint8), ts=0.0, index=0))


def test_missing_video_is_error(tmp_path):
    with pytest.raises(ReplayError):
        ReplaySource(tmp_path)


def test_read_before_start_is_error(tmp_path):
    out = record(tmp_path, synthetic_frames(3))
    with pytest.raises(ReplayError):
        ReplaySource(out).read_latest()
