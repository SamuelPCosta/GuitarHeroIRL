"""Testes das mãos (T1.7): lateralidade, papéis, máscara e (se houver modelo) o MediaPipe."""

from pathlib import Path

import numpy as np
import pytest

from ghirl.app.config import HandsConfig
from ghirl.capture.base import Frame
from ghirl.hands.mp_tracker import Hand, assign_roles_by_label, hand_mask, real_side
from ghirl.render.debug import Label, draw_hands, draw_labels


def make_hand(cx, cy, side="left", score=0.9, size=40):
    rng = np.random.default_rng(int(cx))
    pts = np.array([cx, cy], np.float32) + rng.uniform(-size, size, (21, 2)).astype(np.float32)
    return Hand(landmarks_px=pts, landmarks_z=np.zeros(21, np.float32), handedness_raw="Left",
                score=score, side=side)


@pytest.mark.parametrize(
    "raw,swap,expected",
    [("Left", False, "left"), ("Right", False, "right"), ("Left", True, "right"), ("Right", True, "left")],
)
def test_real_side(raw, swap, expected):
    assert real_side(raw, swap) == expected


def test_roles_by_label():
    hands = [make_hand(100, 100, "left"), make_hand(300, 100, "right")]
    assign_roles_by_label(hands, fretting="left")
    assert [h.role for h in hands] == ["notas", "ritmo"]


def test_roles_conflict_resolved_by_score():
    hands = [make_hand(100, 100, "left", score=0.6), make_hand(300, 100, "left", score=0.95)]
    assign_roles_by_label(hands, fretting="left")
    assert hands[1].role == "notas" and hands[0].role == "ritmo"


def test_hand_mask_covers_hand_and_dilates():
    hand = make_hand(200, 150)
    m0 = hand_mask([hand], (300, 400), dilate_px=0, scale=1.0)
    m1 = hand_mask([hand], (300, 400), dilate_px=15, scale=1.0)
    assert m0.shape == (300, 400) and m0.dtype == np.uint8
    for x, y in hand.landmarks_px.astype(int):
        assert m0[y, x] == 255
    assert (m1 > 0).sum() > (m0 > 0).sum()
    assert hand_mask([], (300, 400)).max() == 0


def test_hand_mask_low_res_close_to_full_res():
    hand = make_hand(200, 150)
    full = hand_mask([hand], (300, 400), dilate_px=12, scale=1.0) > 0
    fast = hand_mask([hand], (300, 400), dilate_px=12, scale=0.25) > 0
    iou = (full & fast).sum() / (full | fast).sum()
    assert iou > 0.85


def test_draw_hands_and_mirrored_labels():
    img = np.zeros((200, 300, 3), np.uint8)
    hand = make_hand(60, 80)
    hand.role = "notas"
    labels = draw_hands(img, [hand])
    assert img.any() and len(labels) == 1
    view = np.zeros_like(img)
    draw_labels(view, [Label((20.0, 100.0), "abc")], mirrored=True)
    cols = np.where(view.any(axis=(0, 2)))[0]
    assert cols.min() > 150  # texto foi para o lado espelhado (direita)


MODEL = Path("models/hand_landmarker.task")


@pytest.mark.skipif(not MODEL.exists(), reason="modelo do MediaPipe não baixado")
def test_tracker_runs_on_blank_and_increasing_timestamps():
    from ghirl.hands.mp_tracker import HandTracker

    trk = HandTracker(HandsConfig())
    try:
        img = np.zeros((240, 320, 3), np.uint8)
        assert trk.detect(Frame(img, 0.0, 0)) == []
        assert trk.detect(Frame(img, 0.0, 1)) == []  # ts repetido não pode quebrar o MediaPipe
        assert trk.detect(Frame(img, 0.033, 2)) == []
    finally:
        trk.close()


def test_missing_model_message(tmp_path):
    from ghirl.hands.mp_tracker import HandTracker, HandTrackerError

    with pytest.raises(HandTrackerError, match="curl"):
        HandTracker(HandsConfig(model_path=str(tmp_path / "nao.task")))
