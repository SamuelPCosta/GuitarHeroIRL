"""Testes do rastreador LK + homografia (T2.5) com sequências sintéticas de homografias conhecidas."""

import cv2
import numpy as np
import pytest

from ghirl.capture.base import Frame
from ghirl.neck.base import Keyframe, NeckDetection
from ghirl.neck.feature_tracker import FeatureTracker, homography_scale
from ghirl.utils.geometry import AxisFrame, Line, apply_homography

W, H = 1280, 720
REGION = np.array([[250, 260], [1050, 230], [1060, 450], [240, 470]], np.int32)  # "violão"


def texture(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 90, np.uint8)
    for _ in range(900):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), int(rng.integers(2, 9)), c, -1)
        else:
            cv2.line(img, (x, y), (x + int(rng.integers(-25, 25)), y + int(rng.integers(-25, 25))), c, 2)
    return cv2.GaussianBlur(img, (3, 3), 0)


def h_at(k, n, rot_deg=12.0, shift=(70, -40), persp=2e-5):
    """Homografia verdadeira K -> quadro k (movimento suave em torno do centro)."""
    a = np.radians(rot_deg) * k / n
    c, s = np.cos(a), np.sin(a)
    cx, cy = W / 2, H / 2
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    Sc = np.diag([1 + 0.1 * k / n, 1 + 0.1 * k / n, 1])
    P = np.array([[1, 0, 0], [0, 1, 0], [persp * k / n, 0, 1]])
    T2 = np.array([[1, 0, cx + shift[0] * k / n], [0, 1, cy + shift[1] * k / n], [0, 0, 1]])
    return T2 @ P @ Sc @ R @ T1


def make_keyframe(img):
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [REGION], 255)
    det = NeckDetection(0, mask, AxisFrame.make([650, 350], [1, 0]), Line.make([0, 260], [1, 0]),
                        Line.make([0, 450], [1, 0]), (-400, 400), full_mask=mask)
    return Keyframe(0, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), det, mask)


def corner_error(H_est, H_true):
    return float(np.mean(np.linalg.norm(apply_homography(H_est, REGION) - apply_homography(H_true, REGION), axis=1)))


def run_sequence(n=40, occluder=False, seed=0):
    base = texture(seed)
    kf = make_keyframe(base)
    trk = FeatureTracker()
    trk.init(Frame(base, 0.0, 0), kf)
    errors, poses = [], []
    for k in range(1, n + 1):
        Ht = h_at(k, n)
        img = cv2.warpPerspective(base, Ht, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=(90, 90, 90))
        excl = None
        if occluder:  # "mão" que passa por cima do braço
            x = 300 + 15 * k
            cv2.rectangle(img, (x, 220), (x + 160, 480), (40, 120, 200), -1)
            excl = np.zeros((H, W), np.uint8)
            cv2.rectangle(excl, (x - 15, 205), (x + 175, 495), 255, -1)
        pose = trk.update(Frame(img, k / 30, k), excl)
        poses.append(pose)
        if pose.H is not None:
            errors.append(corner_error(pose.H, Ht))
    return errors, poses, trk


def test_tracks_smooth_motion_accurately():
    errors, poses, _ = run_sequence()
    assert all(p.status == "ok" for p in poses)
    assert np.mean(errors) < 1.0 and max(errors) < 2.0


def test_tracks_with_moving_occluder_and_excludes_it():
    errors, poses, trk = run_sequence(occluder=True)
    assert sum(p.status != "perdido" for p in poses) == len(poses)
    assert np.mean(errors) < 2.0
    # nenhum ponto rastreado dentro do último retângulo de exclusão
    _, cur = trk.points
    x = 300 + 15 * 40
    inside = (cur[:, 0] > x - 15) & (cur[:, 0] < x + 175) & (cur[:, 1] > 205) & (cur[:, 1] < 495)
    assert not inside.any()


def test_lost_when_texture_disappears():
    base = texture()
    trk = FeatureTracker()
    trk.init(Frame(base, 0.0, 0), make_keyframe(base))
    blank = np.full_like(base, 90)
    statuses = [trk.update(Frame(blank, k / 30, k)).status for k in range(1, 6)]
    assert statuses[-1] == "perdido"


def test_replenishes_points():
    _, _, trk = run_sequence(occluder=True)
    k_pts, cur = trk.points
    assert len(cur) >= trk.p.min_points // 2
    assert k_pts.shape == cur.shape


def test_update_before_init_raises():
    with pytest.raises(RuntimeError):
        FeatureTracker().update(Frame(texture(), 0.0, 0))


def test_deterministic():
    e1, _, _ = run_sequence(n=10)
    e2, _, _ = run_sequence(n=10)
    assert e1 == e2


def test_homography_scale():
    assert homography_scale(np.eye(3), np.array([10.0, 10.0])) == pytest.approx(1.0)
    assert homography_scale(np.diag([2.0, 2.0, 1.0]), np.array([5.0, 5.0])) == pytest.approx(2.0)
