"""Testes de geometria e tipos do braço (T2.1)."""

import numpy as np
import pytest

from ghirl.neck.base import NeckDetection, NeckPose
from ghirl.utils.geometry import (
    AxisFrame,
    Line,
    apply_homography,
    intersect,
    normalize,
    perp,
    undirected_angle_deg,
)

RNG = np.random.default_rng(42)


def random_homography():
    H = np.eye(3) + RNG.normal(0, 0.05, (3, 3))
    H[2, :2] = RNG.normal(0, 1e-4, 2)
    H[:2, 2] = RNG.uniform(-50, 50, 2)
    return H / H[2, 2]


def test_homography_roundtrip():
    H = random_homography()
    pts = RNG.uniform(0, 1280, (50, 2))
    back = apply_homography(np.linalg.inv(H), apply_homography(H, pts))
    assert np.allclose(back, pts, atol=1e-6)


def test_homography_identity_and_translation():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(apply_homography(np.eye(3), pts), pts)
    T = np.array([[1, 0, 10], [0, 1, -5], [0, 0, 1]], float)
    assert np.allclose(apply_homography(T, pts), pts + [10, -5])


def test_homography_point_at_infinity_raises():
    H = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], float)
    with pytest.raises(ValueError):
        apply_homography(H, [[0.0, 5.0]])


def test_axis_frame_roundtrip_and_orthonormal():
    ax = AxisFrame.make([640, 360], [3, -1])
    pts = RNG.uniform(0, 1280, (100, 2))
    st = ax.to_local(pts)
    assert np.allclose(ax.to_image(st), pts, atol=1e-6)
    assert abs(np.dot(ax.u, ax.n)) < 1e-12 and np.isclose(np.linalg.norm(ax.n), 1)


def test_axis_frame_local_values():
    ax = AxisFrame.make([100, 100], [1, 0])
    st = ax.to_local([[150, 100], [100, 130]])
    assert np.allclose(st, [[50, 0], [0, 30]])  # n = +y (para baixo na imagem)


def test_axis_frame_flip_and_shift():
    ax = AxisFrame.make([0, 0], [1, 0])
    p = [[10.0, 5.0]]
    assert np.allclose(ax.flipped().to_local(p), [[-10, -5]])
    assert np.allclose(ax.shifted(4).to_local(p), [[6, 5]])


def test_line_distance_projection_and_intersection():
    l1 = Line.make([0, 0], [1, 0])
    assert np.allclose(l1.signed_distance([[3, 2], [3, -2]]), [2, -2])
    assert np.allclose(l1.project([[3, 2]]), [[3, 0]])
    l2 = Line.through([5, -5], [5, 5])
    assert np.allclose(intersect(l1, l2), [5, 0])
    assert intersect(l1, Line.make([0, 1], [2, 0])) is None


def test_line_transformed_by_homography():
    H = random_homography()
    line = Line.through([10, 20], [200, 80])
    moved = line.transformed(H)
    mid = apply_homography(H, [[105, 50]])  # ponto da reta original
    assert abs(moved.signed_distance(mid)[0]) < 1e-6


def test_angles_and_helpers():
    assert undirected_angle_deg([1, 0], [-1, 0]) == pytest.approx(0, abs=1e-6)
    assert undirected_angle_deg([1, 0], [0, 1]) == pytest.approx(90)
    assert undirected_angle_deg([1, 0], [1, 1]) == pytest.approx(45)
    assert np.allclose(perp([1, 0]), [0, 1])
    with pytest.raises(ValueError):
        normalize([0, 0])


def test_neck_detection_width_and_quad():
    ax = AxisFrame.make([500, 300], [1, 0])
    det = NeckDetection(
        frame_index=0, mask=None, axis=ax,
        edge_a=Line.through([0, 280], [1000, 270]),  # afunila: larguras diferentes
        edge_b=Line.through([0, 320], [1000, 330]),
        extent=(-400, 300),
    )
    assert det.width_at(0) == pytest.approx(50, abs=0.1)  # em x=500: y=275 e y=325
    assert det.width_at(-500) < det.width_at(500)  # as bordas se afastam com x
    q = det.quad()
    assert q.shape == (4, 2)
    assert q[0, 0] == pytest.approx(100) and q[1, 0] == pytest.approx(800)


def test_neck_pose_ok_flag():
    assert not NeckPose(frame_index=0, H=None).ok
    assert NeckPose(frame_index=0, H=np.eye(3), status="ok").ok
    assert not NeckPose(frame_index=0, H=np.eye(3), status="perdido").ok
