"""Testes do modelo de trastes e dos solvers de calibração (T3.1, T3.2)."""

import numpy as np
import pytest

from ghirl.calib.fret_model import (
    AxisCalibration,
    CalibrationError,
    StringModel,
    finger_g,
    fret_g,
    g_to_fret,
    solve_affine,
    solve_projective,
    solve_taper,
    validate,
)

BETA = 0.75


def test_fret_formula_known_values():
    assert fret_g(0) == 0
    assert fret_g(12) == pytest.approx(0.5)
    assert fret_g(24) == pytest.approx(0.75)
    assert 648 * fret_g(1) == pytest.approx(36.37, abs=0.01)  # casa 1 de uma escala de 648 mm


def test_fret_inverse_roundtrip():
    n = np.linspace(0, 15, 61)
    assert np.allclose(g_to_fret(fret_g(n)), n)


def test_finger_position_values():
    assert finger_g(1, BETA) == pytest.approx(0.04210, abs=1e-5)
    assert finger_g(5, BETA) == pytest.approx(0.23970, abs=1e-5)
    assert finger_g(12, BETA) == pytest.approx(0.49257, abs=1e-5)
    assert fret_g(4) < finger_g(5, BETA) < fret_g(5)


def test_axis_calibration_roundtrip_and_nut():
    cal = AxisCalibration(a=-600.0, b=300.0, c=0.2, model="3p")
    g = np.linspace(0, 0.5, 11)
    assert np.allclose(cal.g_of_s(cal.s_of_g(g)), g)
    assert cal.nut_s == 300.0 and cal.direction == -1
    assert len(cal.fret_s()) == 13


def true_projective(theta_deg=0.0, L_px=650.0, nut=-300.0, Z=800.0, ell=325.0):
    """Mapa verdadeiro g -> s para um braço girado ``theta`` (guinada), visto a distância Z (mm)."""
    sin = np.sin(np.radians(theta_deg))
    # x_img ∝ X·Z/(Z + X·sinθ), X = g·L (mm, L = 2·ell); normalizado para L_px no traste 12 sem giro
    def s_of_g(g):
        X = np.asarray(g) * 2 * ell
        return nut + L_px * (X * Z / (Z + X * sin)) / (2 * ell)
    return s_of_g


@pytest.mark.parametrize("theta", [0.0, 15.0, 30.0])
def test_projective_is_exact_without_noise(theta):
    s_true = true_projective(theta)
    meas = [(n, float(s_true(finger_g(n, BETA)))) for n in (1, 12, 5)]
    cal = solve_projective(meas, BETA)
    frets = fret_g(np.arange(13))
    assert np.allclose(cal.s_of_g(frets), s_true(frets), atol=0.01)


def test_affine_exact_when_frontal_and_degrades_with_yaw():
    frets = fret_g(np.arange(13))
    errs = {}
    for theta in (0.0, 15.0, 30.0):
        s_true = true_projective(theta)
        cal = solve_affine([(n, float(s_true(finger_g(n, BETA)))) for n in (1, 12)], BETA)
        errs[theta] = np.abs(cal.s_of_g(frets) - s_true(frets)).max()
    assert errs[0.0] < 1e-6
    assert errs[0.0] < errs[15.0] < errs[30.0]
    # valores da tabela de design §8.2 (px ≈ mm aqui): ~5,9 a 15°, ~9,9 a 30°
    assert 5.0 < errs[15.0] < 7.0 and 9.0 < errs[30.0] < 11.0


def test_taper_model_recovers_perspective():
    theta, Z, ell = 25.0, 800.0, 325.0
    s_true = true_projective(theta, Z=Z, ell=ell)
    sin = np.sin(np.radians(theta))

    def width_img(g):  # largura física com afunilamento 43 -> 54, ampliada como 1/(Z + X sinθ)
        wf = 43 + (54 - 43) * (g / 0.5)
        return wf * Z / (Z + g * 2 * ell * sin)

    ns = (1, 12)
    meas = [(n, float(s_true(finger_g(n, BETA)))) for n in ns]
    widths = [float(width_img(finger_g(n, BETA))) for n in ns]
    cal = solve_taper(meas, widths, BETA, 43.0, 54.0)
    frets = fret_g(np.arange(13))
    aff = solve_affine(meas, BETA)
    err_taper = np.abs(cal.s_of_g(frets) - s_true(frets)).max()
    err_aff = np.abs(aff.s_of_g(frets) - s_true(frets)).max()
    assert err_taper < 0.5 < err_aff


def test_least_squares_with_extra_measures_and_errors():
    s_true = true_projective(0.0)
    meas = [(n, float(s_true(finger_g(n, BETA)))) for n in (1, 5, 7, 12)]
    assert solve_affine(meas, BETA).c == 0.0
    with pytest.raises(CalibrationError):
        solve_affine(meas[:1], BETA)
    with pytest.raises(CalibrationError):
        solve_projective(meas[:2], BETA)


def test_validate_accepts_good_and_rejects_bad():
    s_true = true_projective(0.0, L_px=650.0, nut=-300.0)
    meas = [(1, float(s_true(finger_g(1, BETA)))), (12, float(s_true(finger_g(12, BETA))))]
    cal = solve_affine(meas, BETA)
    width = 48.0  # px; 293/48 ≈ 6,1
    assert validate(cal, meas, width, extent=(-320, 80), nut_side_hint=-1) == []
    # pestanas trocadas em relação à cabeça
    assert any("cabeça" in i for i in validate(cal, meas, width, nut_side_hint=+1))
    # distância implausível em relação à largura
    assert any("largura" in i for i in validate(cal, meas, 200.0))
    # perspectiva absurda
    assert any("perspectiva" in i for i in validate(AxisCalibration(600, 0, -5.0), meas, width))


def test_string_model_roundtrip_and_extremes():
    sm = StringModel(margin=0.09)
    assert sm.w_of_string(6) == pytest.approx(0.09)
    assert sm.w_of_string(1) == pytest.approx(0.91)
    k = np.linspace(1, 6, 11)
    assert np.allclose(sm.string_of_w(sm.w_of_string(k)), k)
