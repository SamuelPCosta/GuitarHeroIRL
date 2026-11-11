"""Testes da geometria a partir da máscara (T2.3) com violões sintéticos."""

import cv2
import numpy as np
import pytest

from ghirl.neck.mask_geometry import analyze_mask, longest_run
from ghirl.utils.geometry import undirected_angle_deg

W, H = 1280, 720


def _ellipse_pts(cx, cy, ax, ay, n=60):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + ax * np.cos(t), cy + ay * np.sin(t)], axis=1)


def synth_guitar(angle_deg=0.0, center=(640, 360), neck_only=False, bite=False, scale=1.0):
    """Violão em coordenadas locais (x ao longo do braço, cabeça em +x), girado e transladado.

    Proporções aproximadas de um folk (em px a ~1 px/mm): braço 380 de comprimento, 43 -> 54 de
    largura; cabeça 110 x 80; corpo ~300 x 280.
    """
    a = np.radians(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = np.asarray(center, float)

    def tf(p):
        return (np.asarray(p, float) * scale) @ R.T + c

    neck_x0, neck_x1 = -150.0, 230.0
    w_body, w_nut = 54.0, 43.0
    neck = [[neck_x0, -w_body / 2], [neck_x1, -w_nut / 2], [neck_x1, w_nut / 2], [neck_x0, w_body / 2]]
    head = [[neck_x1, -40], [neck_x1 + 110, -40], [neck_x1 + 110, 40], [neck_x1, 40]]
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [tf(neck).astype(np.int32)], 255)
    if not neck_only:
        cv2.fillPoly(m, [tf(head).astype(np.int32)], 255)
        cv2.fillPoly(m, [tf(_ellipse_pts(-260, 0, 120, 120)).astype(np.int32)], 255)
        cv2.fillPoly(m, [tf(_ellipse_pts(-400, 0, 110, 140)).astype(np.int32)], 255)
    if bite:  # mão sobre o braço "morde" a borda superior
        bx, by = tf([[60, -w_body / 2]])[0]
        cv2.circle(m, (int(bx), int(by)), int(14 * scale), 0, -1)
    truth = {
        "u": R @ np.array([1.0, 0.0]),
        "edge_pts": tf([[x, s * (w_body / 2 + (w_nut - w_body) / 2 * (x - neck_x0) / (neck_x1 - neck_x0))]
                        for x in np.linspace(-100, 180, 8) for s in (-1, 1)]),
        "neck_len": (neck_x1 - neck_x0) * scale,
    }
    return m, truth


def edge_error(det, truth):
    d = [min(abs(det.edge_a.signed_distance([p])[0]), abs(det.edge_b.signed_distance([p])[0]))
         for p in truth["edge_pts"]]
    return max(d)


@pytest.mark.parametrize("angle", [0.0, 12.0, -25.0, 40.0, 170.0])
def test_full_guitar_axis_edges_and_head_side(angle):
    m, truth = synth_guitar(angle)
    det = analyze_mask(m)
    assert det is not None
    assert undirected_angle_deg(det.axis.u, truth["u"]) < 2.0
    assert edge_error(det, truth) < 3.0
    sign = 1 if np.dot(det.axis.u, truth["u"]) > 0 else -1
    assert det.nut_side_hint == sign  # a cabeça fica em +x local
    length = det.extent[1] - det.extent[0]
    assert 0.8 * truth["neck_len"] < length < 1.15 * truth["neck_len"]
    assert det.width_at(0.0) == pytest.approx(48.5, abs=4)
    assert det.confidence > 0.5


def test_neck_only_mask_has_no_head_hint():
    m, truth = synth_guitar(18.0, neck_only=True)
    det = analyze_mask(m)
    assert det is not None
    assert undirected_angle_deg(det.axis.u, truth["u"]) < 2.0
    assert det.nut_side_hint is None


def test_hand_occlusion_bite_is_tolerated():
    m, truth = synth_guitar(-10.0, bite=True)
    det = analyze_mask(m)
    assert det is not None
    assert undirected_angle_deg(det.axis.u, truth["u"]) < 3.0
    assert edge_error(det, truth) < 4.0


def test_smaller_guitar_far_from_camera():
    m, truth = synth_guitar(5.0, scale=0.55)
    det = analyze_mask(m)
    assert det is not None
    assert undirected_angle_deg(det.axis.u, truth["u"]) < 3.0


def test_edges_are_on_opposite_sides_and_mask_cut():
    m, _ = synth_guitar(30.0)
    det = analyze_mask(m)
    ta = det.axis.to_local([det.edge_a.point])[0, 1]
    tb = det.axis.to_local([det.edge_b.point])[0, 1]
    assert ta < 0 < tb
    assert 0 < (det.mask > 0).sum() < (m > 0).sum()  # só a faixa do braço


@pytest.mark.parametrize("kind", ["empty", "tiny", "disk"])
def test_rejects_non_neck_shapes(kind):
    m = np.zeros((H, W), np.uint8)
    if kind == "tiny":
        cv2.circle(m, (100, 100), 5, 255, -1)
    elif kind == "disk":
        cv2.circle(m, (640, 360), 200, 255, -1)
    assert analyze_mask(m) is None


def test_longest_run_with_gap_fill():
    f = np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1], bool)
    assert longest_run(f) == (3, 5)
    assert longest_run(f, gap_fill=1) == (0, 5)
    assert longest_run(np.zeros(4, bool)) is None
