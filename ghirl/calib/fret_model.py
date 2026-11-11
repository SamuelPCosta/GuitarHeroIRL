"""Modelo de trastes, cordas e calibração ao longo do eixo (T3.1–T3.2; design §8).

Coordenadas:
    g = u / L ∈ [0, 0.5]   posição física ao longo do braço como fração da escala (0 = nut, 0.5 = traste 12)
    s                      coordenada ao longo do eixo do braço no quadro-chave K (px)
    w ∈ [0, 1]             posição através do braço (0 = borda dos bordões, 1 = borda das primas)
    n̂ (contínuo)           casa: traste n em n̂ = n; a casa n é o intervalo (n-1, n]
    k̂ (contínuo)           corda: 6 (Mi grave) ... 1 (Mi aguda)

Calibração ao longo do eixo: s = (a·g + b) / (c·g + 1). Afim quando c = 0. Nut em s = b.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_FRETS = 12


def fret_g(n) -> np.ndarray:
    """Posição do traste n (fração da escala): 1 - 2^(-n/12)."""
    return 1.0 - 2.0 ** (-np.asarray(n, np.float64) / 12.0)


def g_to_fret(g) -> np.ndarray:
    """Inverso de ``fret_g``: casa contínua n̂ a partir de g (g < 1)."""
    g = np.minimum(np.asarray(g, np.float64), 1.0 - 1e-9)
    return -12.0 * np.log2(1.0 - g)


def finger_g(n, beta: float) -> np.ndarray:
    """Posição típica do dedo na casa n: entre os trastes n-1 e n, a uma fração ``beta`` (0..1)."""
    n = np.asarray(n, np.float64)
    a, b = fret_g(n - 1), fret_g(n)
    return a + beta * (b - a)


@dataclass(frozen=True)
class AxisCalibration:
    """Mapa 1D projetivo g <-> s (em K)."""

    a: float
    b: float
    c: float = 0.0
    model: str = "2p"

    def s_of_g(self, g) -> np.ndarray:
        g = np.asarray(g, np.float64)
        return (self.a * g + self.b) / (self.c * g + 1.0)

    def g_of_s(self, s) -> np.ndarray:
        s = np.asarray(s, np.float64)
        return (s - self.b) / (self.a - self.c * s)

    @property
    def nut_s(self) -> float:
        return float(self.b)

    def fret_s(self, n_max: int = N_FRETS) -> np.ndarray:
        return self.s_of_g(fret_g(np.arange(n_max + 1)))

    @property
    def direction(self) -> int:
        """+1 se s cresce do nut para o corpo, -1 caso contrário."""
        return 1 if self.a > 0 else -1


class CalibrationError(ValueError):
    pass


def solve_affine(measures: list[tuple[int, float]], beta: float) -> AxisCalibration:
    """2 pontos (ou mais, mínimos quadrados): s = a·g + b."""
    if len(measures) < 2:
        raise CalibrationError("o modelo afim precisa de pelo menos 2 casas")
    g = finger_g([n for n, _ in measures], beta)
    s = np.array([v for _, v in measures], np.float64)
    A = np.stack([g, np.ones_like(g)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, s, rcond=None)
    return AxisCalibration(float(a), float(b), 0.0, "2p")


def solve_projective(measures: list[tuple[int, float]], beta: float) -> AxisCalibration:
    """3 pontos (ou mais): a·g + b − c·g·s = s (linear em a, b, c)."""
    if len(measures) < 3:
        raise CalibrationError("o modelo projetivo precisa de pelo menos 3 casas")
    g = finger_g([n for n, _ in measures], beta)
    s = np.array([v for _, v in measures], np.float64)
    A = np.stack([g, np.ones_like(g), -g * s], axis=1)
    (a, b, c), *_ = np.linalg.lstsq(A, s, rcond=None)
    return AxisCalibration(float(a), float(b), float(c), "3p")


def solve_taper(measures: list[tuple[int, float]], widths_px: list[float], beta: float,
                width_nut_mm: float, width_12_mm: float) -> AxisCalibration:
    """2 pontos + larguras medidas nesses pontos e o afunilamento físico conhecido.

    Sob a homografia, a ampliação transversal é proporcional a 1/(c·g + 1):
        w_img(g)·(c·g + 1) = k·w_fis(g)   e   s·(c·g + 1) = a·g + b
    -> 4 equações lineares em (a, b, c, k).
    """
    if len(measures) != len(widths_px) or len(measures) < 2:
        raise CalibrationError("precisa de 2 casas com as respectivas larguras")
    g = finger_g([n for n, _ in measures], beta)
    s = np.array([v for _, v in measures], np.float64)
    wi = np.asarray(widths_px, np.float64)
    wf = width_nut_mm + (width_12_mm - width_nut_mm) * (g / 0.5)  # afunilamento linear nut -> traste 12
    rows, rhs = [], []
    for gi, si, wii, wfi in zip(g, s, wi, wf, strict=True):
        rows.append([gi, 1.0, -gi * si, 0.0])
        rhs.append(si)
        rows.append([0.0, 0.0, wii * gi, -wfi])
        rhs.append(-wii)
    (a, b, c, _k), *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return AxisCalibration(float(a), float(b), float(c), "2p_taper")


def validate(cal: AxisCalibration, measures: list[tuple[int, float]], neck_width_px: float,
             extent: tuple[float, float] | None = None, nut_side_hint: int | None = None,
             ratio_range: tuple[float, float] = (3.0, 10.0), c_max: float = 1.5) -> list[str]:
    """Regras de design §8.5. Devolve a lista de problemas (vazia = ok)."""
    issues = []
    by_n = dict(measures)
    if 1 in by_n and 12 in by_n:
        span = by_n[12] - by_n[1]
        if neck_width_px > 0:
            ratio = abs(span) / neck_width_px
            if not ratio_range[0] <= ratio <= ratio_range[1]:
                issues.append(f"distância casa 1 → casa 12 é {ratio:.1f}× a largura do braço "
                              f"(esperado {ratio_range[0]:.0f}–{ratio_range[1]:.0f}×)")
        if nut_side_hint is not None and np.sign(span) != 0 and np.sign(span) == nut_side_hint:
            issues.append("a casa 12 ficou do lado da cabeça do violão (pestanas trocadas?)")
    gg = np.linspace(0, 0.55, 23)
    if np.any(cal.c * gg + 1.0 <= 0.05) or abs(cal.c) > c_max:
        issues.append("perspectiva implausível no ajuste (c fora da faixa)")
    if extent is not None:
        lo, hi = min(extent), max(extent)
        margin = 0.25 * (hi - lo)
        if not lo - margin <= cal.nut_s <= hi + margin:
            issues.append("o nut previsto cai fora do braço detectado")
    if cal.a == 0:
        issues.append("escala nula")
    return issues


@dataclass(frozen=True)
class StringModel:
    """Cordas entre as bordas do braço, com margem ``margin`` (fração da largura) em cada lado."""

    margin: float = 0.09

    def w_of_string(self, k) -> np.ndarray:
        k = np.asarray(k, np.float64)
        return self.margin + (6.0 - k) / 5.0 * (1.0 - 2.0 * self.margin)

    def string_of_w(self, w) -> np.ndarray:
        w = np.asarray(w, np.float64)
        return 6.0 - 5.0 * (w - self.margin) / (1.0 - 2.0 * self.margin)
