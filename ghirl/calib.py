"""Modelo do braço e calibração por pestana.

Coordenadas: g = fração da escala (0 = nut, 0.5 = traste 12); s = posição no eixo em K (px);
w ∈ [0, 1] através do braço (0 = bordões); n̂ = casa contínua (casa n = (n-1, n]); k̂ = corda (6..1).
Calibração: s = (a·g + b) / (c·g + 1) (afim se c = 0).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from ghirl.config import CalibConfig
from ghirl.geometry import Line, NeckDetection, apply_homography, intersect, undirected_angle_deg
from ghirl.hands import INDEX_DIP, INDEX_MCP, INDEX_PIP, INDEX_TIP, Hand

# ==================================================================================================
# trastes e calibração ao longo do eixo
# ==================================================================================================

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


def fit_axis(g, s, projective: bool = False) -> AxisCalibration:
    """Mínimos quadrados de s = (a·g + b)/(c·g + 1). Afim (c = 0) com 2+ pontos; projetivo com 3+."""
    g, s = np.asarray(g, np.float64), np.asarray(s, np.float64)
    if len(g) < (3 if projective else 2):
        raise CalibrationError("pontos insuficientes para o ajuste")
    cols = [g, np.ones_like(g)] + ([-g * s] if projective else [])
    sol, *_ = np.linalg.lstsq(np.stack(cols, axis=1), s, rcond=None)
    c = float(sol[2]) if projective else 0.0
    return AxisCalibration(float(sol[0]), float(sol[1]), c, "trastes" if projective else "pestanas")


def solve_affine(measures: list[tuple[int, float]], beta: float) -> AxisCalibration:
    """Pestanas (casa, s): o dedo fica a uma fração ``beta`` da casa."""
    return fit_axis(finger_g([n for n, _ in measures], beta), [v for _, v in measures])


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


# ==================================================================================================
# modelo do braço em K
# ==================================================================================================

@dataclass
class NeckGeometry:
    """Eixo e bordas do braço em K, com a borda dos bordões identificada."""

    detection: NeckDetection
    bass_is_a: bool  # True se edge_a é a borda dos bordões (corda 6)

    @staticmethod
    def from_detection(det: NeckDetection, bass_edge: str = "top") -> NeckGeometry:
        # compara as bordas no meio do braço: "top" = menor y na imagem
        s_mid = 0.5 * (det.extent[0] + det.extent[1])
        cut = Line(det.axis.to_image([[s_mid, 0.0]])[0], det.axis.n)
        pa, pb = intersect(cut, det.edge_a), intersect(cut, det.edge_b)
        ya = pa[1] if pa is not None else det.edge_a.point[1]
        yb = pb[1] if pb is not None else det.edge_b.point[1]
        a_is_top = ya < yb
        return NeckGeometry(det, bass_is_a=a_is_top if bass_edge == "top" else not a_is_top)

    @property
    def bass_edge(self) -> Line:
        return self.detection.edge_a if self.bass_is_a else self.detection.edge_b

    @property
    def treble_edge(self) -> Line:
        return self.detection.edge_b if self.bass_is_a else self.detection.edge_a

    def edge_points_at(self, s) -> tuple[np.ndarray, np.ndarray]:
        """Pontos (N, 2) em K das bordas (bordões, primas) na posição s do eixo."""
        s = np.atleast_1d(np.asarray(s, np.float64))
        ax = self.detection.axis
        base = ax.to_image(np.stack([s, np.zeros_like(s)], axis=1))
        out_b, out_t = [], []
        for p in base:
            cut = Line(p, ax.n)
            b = intersect(cut, self.bass_edge)
            t = intersect(cut, self.treble_edge)
            out_b.append(b if b is not None else p)
            out_t.append(t if t is not None else p)
        return np.array(out_b), np.array(out_t)

    def width_at(self, s: float) -> float:
        b, t = self.edge_points_at(s)
        return float(np.linalg.norm(b[0] - t[0]))


@dataclass
class NeckModel:
    geometry: NeckGeometry
    calibration: AxisCalibration
    strings: StringModel
    keyframe_id: int = 0

    # ------------------------------------------------------------------ braço -> K

    def point_K(self, n_hat, w) -> np.ndarray:
        """Ponto em K para casa contínua n̂ e posição transversal w (0 = bordões, 1 = primas)."""
        n_hat = np.atleast_1d(np.asarray(n_hat, np.float64))
        w = np.broadcast_to(np.asarray(w, np.float64), n_hat.shape)
        s = self.calibration.s_of_g(fret_g(n_hat))
        b, t = self.geometry.edge_points_at(s)
        return b + w[:, None] * (t - b)

    def fret_segment_K(self, n: int) -> np.ndarray:
        """Traste n como segmento (2, 2) de borda a borda, em K."""
        return self.point_K([n, n], [0.0, 1.0])

    def string_polyline_K(self, k: int, n_from: float = 0.0, n_to: float = N_FRETS, steps: int = 13) -> np.ndarray:
        n = np.linspace(n_from, n_to, steps)
        return self.point_K(n, self.strings.w_of_string(k))

    def cell_center_K(self, k: int, n: int, beta: float = 0.6) -> np.ndarray:
        """Centro da célula (corda k, casa n), um pouco mais perto do traste n (onde se aperta)."""
        n_hat = (n - 1) + beta if n > 0 else -0.3
        return self.point_K([n_hat], self.strings.w_of_string(k))[0]

    # ------------------------------------------------------------------ K -> braço

    def to_musical(self, pts_K) -> np.ndarray:
        """Pontos em K -> (n̂, k̂) contínuos (casa, corda)."""
        pts = np.atleast_2d(np.asarray(pts_K, np.float64))
        ax = self.geometry.detection.axis
        s = ax.to_local(pts)[:, 0]
        n_hat = g_to_fret(self.calibration.g_of_s(s))
        b, t = self.geometry.edge_points_at(s)
        d = t - b
        w = np.einsum("ij,ij->i", pts - b, d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-9)
        return np.stack([n_hat, self.strings.string_of_w(w)], axis=1)


# ==================================================================================================
# detecção da pestana
# ==================================================================================================

@dataclass
class BarreParams:
    min_joint_deg: float = 150.0
    max_perp_dev_deg: float = 25.0
    min_coverage: float = 0.5
    # fração da extensão detectada tolerada além das pontas. Generosa porque a máscara costuma cobrir
    # só parte do braço (violão escuro na frente de roupa escura); as bordas, que são retas, valem
    # para o braço inteiro.
    extent_margin: float = 0.8


@dataclass
class BarreObservation:
    valid: bool
    s: float | None  # coordenada ao longo do eixo (K) do cruzamento do indicador com a linha central
    reasons: list[str] = field(default_factory=list)
    coverage: float = 0.0
    perp_dev_deg: float = 90.0
    joint_deg: tuple[float, float] = (0.0, 0.0)


def _angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Ângulo ABC em graus (180 = reto)."""
    v1, v2 = a - b, c - b
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    if den < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / den, -1, 1))))


def observe_barre(hand: Hand, H: np.ndarray, geom: NeckGeometry, params: BarreParams | None = None) -> BarreObservation:
    p = params or BarreParams()
    det = geom.detection
    ax = det.axis
    pts = apply_homography(np.linalg.inv(H), hand.landmarks_px[[INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP]])
    mcp, pip, dip, tip = pts
    reasons = []

    j1, j2 = _angle_at(mcp, pip, dip), _angle_at(pip, dip, tip)
    if min(j1, j2) < p.min_joint_deg:
        reasons.append("indicador dobrado")

    finger = pts[1:]  # 6, 7, 8
    center = finger.mean(axis=0)
    _, _, vt = np.linalg.svd(finger - center)
    fdir = vt[0]
    perp_dev = undirected_angle_deg(fdir, ax.n)
    if perp_dev > p.max_perp_dev_deg:
        reasons.append("indicador inclinado em relação ao traste")

    st = ax.to_local(finger)
    s_mid = float(np.mean(st[:, 0]))
    width = geom.width_at(s_mid)
    coverage = float((st[:, 1].max() - st[:, 1].min()) / max(width, 1e-6))
    if coverage < p.min_coverage:
        reasons.append("indicador não atravessa o braço")

    cross = intersect(Line.make(center, fdir), ax.line)
    s_cross = None
    if cross is None:
        reasons.append("indicador paralelo ao braço")
    else:
        s_cross = float(ax.to_local([cross])[0, 0])
        lo, hi = det.extent
        m = p.extent_margin * (hi - lo)
        if not (lo - m <= s_cross <= hi + m):
            reasons.append("fora do braço")
    return BarreObservation(not reasons, s_cross, reasons, coverage, perp_dev, (j1, j2))


def fret_profile(gray: np.ndarray, geom: NeckGeometry, s_lo: float, s_hi: float,
                 w_range: tuple[float, float] = (0.15, 0.85), n_w: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Perfil de "trasteza" ao longo do eixo: os trastes são linhas finas perpendiculares ao braço.

    Retifica a faixa do braço (s × w), tira a média local ao longo de s (sobra só o que é fino) e usa a
    mediana através do braço (uma corda ou um dedo cobrindo parte da largura não vira pico).
    """
    s = np.arange(min(s_lo, s_hi), max(s_lo, s_hi), 1.0)
    b, t = geom.edge_points_at(s)
    ws = np.linspace(*w_range, n_w)
    pts = (b[:, None, :] + ws[None, :, None] * (t - b)[:, None, :]).astype(np.float32)
    strip = cv2.remap(gray, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR).astype(np.float32)
    ridge = np.abs(strip - cv2.blur(strip, (1, 9)))  # (len s, n_w); filtro ao longo de s
    # as duas bordas do fio do traste viram 2 picos colados; a suavização junta num só
    prof = cv2.GaussianBlur(np.median(ridge, axis=1).reshape(-1, 1), (1, 9), 2.0).ravel()
    return s, prof


def refine_with_frets(gray: np.ndarray, geom: NeckGeometry, cal: AxisCalibration,
                      n_max: int = N_FRETS, min_frets: int = 5) -> tuple[AxisCalibration, list] | None:
    """Ajusta a escala pelos trastes visíveis: procura um pico perto de cada traste previsto pelas pestanas
    e refaz o ajuste (projetivo, com 3 parâmetros) com todos os trastes achados. None se achou poucos."""
    fs = cal.fret_s(n_max)
    spacing = np.abs(np.diff(fs))
    s, prof = fret_profile(gray, geom, fs.min() - spacing[0], fs.max() + spacing[-1])
    floor = np.median(prof) + 2.0 * np.median(np.abs(prof - np.median(prof)))
    current, found = cal, []
    for win in (0.3, 0.15):  # 2ª passada mais estreita, em torno do ajuste refeito
        pred = current.fret_s(n_max)
        found = []
        for n in range(n_max + 1):
            sp = spacing[max(n - 1, 0)]
            sel = np.abs(s - pred[n]) <= win * sp
            if sel.sum() < 3:
                continue
            i = int(np.argmax(np.where(sel, prof, -1.0)))
            if prof[i] > floor and 0 < i < len(s) - 1:
                found.append((n, float(s[i])))
        if len(found) < min_frets:
            return None
        n_arr = np.array([n for n, _ in found])
        s_arr = np.array([v for _, v in found])
        current = fit_axis(fret_g(n_arr), s_arr, projective=True)
        res = np.abs(current.s_of_g(fret_g(n_arr)) - s_arr)
        keep = res < 0.12 * spacing[np.maximum(n_arr - 1, 0)]
        if keep.sum() >= min_frets and not keep.all():
            found = [f for f, k in zip(found, keep, strict=True) if k]
            current = fit_axis(fret_g(n_arr[keep]), s_arr[keep], projective=True)
    return current, found


def transfer_calibration(model: NeckModel, H_old_new: np.ndarray, geom: NeckGeometry, gray: np.ndarray,
                         keyframe_id: int, margin: float) -> tuple[NeckModel, list] | None:
    """Leva a calibração de um quadro-chave antigo para um novo e a reajusta pelos trastes do novo.

    Os trastes previstos no K antigo vão para o novo pela homografia; o ajuste pelos trastes corrige o
    resto. Sem trastes suficientes na imagem nova, devolve None (aí é preciso refazer as pestanas).
    """
    g = fret_g(np.arange(N_FRETS + 1))
    old_s = model.calibration.s_of_g(g)
    pts = apply_homography(H_old_new, model.geometry.detection.axis.to_image(np.stack([old_s, 0 * old_s], 1)))
    guess = fit_axis(g, geom.detection.axis.to_local(pts)[:, 0], projective=True)
    ref = refine_with_frets(gray, geom, guess, min_frets=6)
    if ref is None:
        return None
    cal, found = ref
    # um conjunto de trastes deslocado de uma casa inteira também se encaixa bem: só aceita se o ajuste
    # mexeu cada traste menos de 1/3 da casa (senão o registro entre os quadros-chave estava ruim demais)
    spacing = np.abs(np.diff(guess.fret_s(N_FRETS)))
    if np.any(np.abs(cal.fret_s(N_FRETS) - guess.fret_s(N_FRETS))[1:] > spacing / 3) or validate(cal, [], 0):
        return None
    return NeckModel(geom, cal, StringModel(margin), keyframe_id), found


class Calibrator:
    """Calibração em etapas acionadas por botão: pestana na casa 1 -> [Capturar] -> casa 12 -> [Capturar].

    A cada quadro, ``observe`` mede a pestana de cada mão (a que for válida). Ao clicar, ``capture``
    usa a mediana das medidas válidas dos últimos ``window_s`` segundos. Sem contagem regressiva:
    o usuário troca de casa no tempo dele.
    """

    def __init__(self, cfg: CalibConfig, geom: NeckGeometry, keyframe_id: int, window_s: float = 0.4,
                 gray_K: np.ndarray | None = None):
        self.cfg = cfg
        self.gray_K = gray_K  # imagem de K, para o ajuste pelos trastes
        self.frets_found: list[tuple[int, float]] = []
        self.geom = geom
        self.keyframe_id = keyframe_id
        self.window_s = window_s
        self.frets = list(cfg.frets)
        self.measures: list[tuple[int, float]] = []
        self.model: NeckModel | None = None
        self.issues: list[str] = []
        self.last_obs: BarreObservation | None = None
        self._recent: deque[tuple[float, float]] = deque()  # (ts, s) válidos

    @property
    def done(self) -> bool:
        return self.model is not None

    @property
    def current_fret(self) -> int | None:
        return None if self.done else self.frets[len(self.measures)]

    def observe(self, ts: float, hands: list[Hand], H: np.ndarray | None) -> BarreObservation | None:
        obs = None
        if H is not None:
            for h in hands:
                o = observe_barre(h, H, self.geom)
                if obs is None or (o.valid and not obs.valid):
                    obs = o
        self.last_obs = obs
        if obs is not None and obs.valid:
            self._recent.append((ts, obs.s))
        while self._recent and ts - self._recent[0][0] > self.window_s:
            self._recent.popleft()
        return obs

    @property
    def barre_ok(self) -> bool:
        """Há pestana válida recente (o botão fica verde)."""
        return len(self._recent) >= 3

    @property
    def hint(self) -> str:
        if self.barre_ok:
            return "pestana OK"
        if self.last_obs is None:
            return "mão não detectada sobre o braço"
        return ", ".join(self.last_obs.reasons)

    def capture(self) -> str | None:
        """Registra a pestana da etapa atual. Devolve uma mensagem de erro, ou None se capturou."""
        if self.done:
            return "já calibrado"
        if not self.barre_ok:
            return f"pestana não reconhecida: {self.hint}"
        s = float(np.median([v for _, v in self._recent]))
        if self.measures and min(abs(s - v) for _, v in self.measures) < 1.5 * self.geom.width_at(s):
            return f"essa é a mesma posição da casa {self.measures[-1][0]}; mude para a casa {self.current_fret}"
        self.measures.append((self.current_fret, s))
        self._recent.clear()
        if len(self.measures) == len(self.frets):
            return self._finish()
        return None

    def undo(self) -> None:
        self.model = None
        self.issues = []
        if self.measures:
            self.measures.pop()

    def _finish(self) -> str | None:
        cal = solve_affine(self.measures, self.cfg.beta)
        det = self.geom.detection
        mid = 0.5 * (det.extent[0] + det.extent[1])
        self.issues = validate(cal, self.measures, self.geom.width_at(mid), nut_side_hint=det.nut_side_hint)
        if self.issues:
            self.measures.clear()
            return "calibração rejeitada: " + "; ".join(self.issues) + ". Recomece pela casa 1."
        self.frets_found = []
        if self.cfg.refine_frets and self.gray_K is not None:
            ref = refine_with_frets(self.gray_K, self.geom, cal)
            # aceita o ajuste pelos trastes só se ele concordar com as pestanas (meia casa no máximo)
            if ref is not None:
                new, found = ref
                g_bar = finger_g([n for n, _ in self.measures], self.cfg.beta)
                d = np.abs(new.s_of_g(g_bar) - cal.s_of_g(g_bar))
                half = 0.5 * np.abs(np.diff(cal.fret_s(N_FRETS)))[[0, -1]]
                if np.all(d < half) and not validate(new, [], 0, nut_side_hint=None):
                    cal, self.frets_found = new, found
        self.model = NeckModel(self.geom, cal, StringModel(self.cfg.string_margin), self.keyframe_id)
        return None
