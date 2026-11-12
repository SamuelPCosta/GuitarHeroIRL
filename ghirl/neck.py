"""Localização do braço: YOLOE (zero-shot) -> máscara -> geometria; worker assíncrono.

YOLOE prompt-free acha o violão; ``analyze_mask`` extrai eixo, bordas e lado da cabeça. Nada é treinado.
Roda numa thread (``SegWorker``) para não travar o vídeo.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from ghirl.capture import Frame
from ghirl.geometry import AxisFrame, Line, NeckDetection, intersect, normalize

# ==================================================================================================
# segmentadores
# ==================================================================================================

GUITAR_CLASSES = ("guitar", "bass guitar")


@dataclass
class SegResult:
    mask: np.ndarray  # uint8 255 = objeto, tamanho da imagem
    box: tuple[float, float, float, float]  # x0, y0, x1, y1
    confidence: float
    label: str
    latency_ms: float = 0.0


def _sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _to_full_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    m = mask.astype(np.float32)
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return (m > 0.5).astype(np.uint8) * 255


class YoloeGuitarSegmenter:
    name = "yoloe"

    def __init__(self, model_path: str = "models/yoloe-26s-seg-pf.pt", classes=GUITAR_CLASSES,
                 conf: float = 0.15, imgsz: int = 640, device: int | str = 0):
        from ultralytics import YOLOE

        self.model = YOLOE(model_path)
        names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))
        self.class_ids = [k for k, v in names.items() if v in classes]
        if not self.class_ids:
            raise ValueError(f"nenhuma das classes {classes} existe no vocabulário do modelo {model_path}")
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def segment(self, image: np.ndarray) -> list[SegResult]:
        t0 = time.perf_counter()
        r = self.model.predict(image, conf=self.conf, classes=self.class_ids, imgsz=self.imgsz,
                               retina_masks=True, verbose=False, device=self.device)[0]
        _sync()
        dt = (time.perf_counter() - t0) * 1000.0
        out: list[SegResult] = []
        if r.boxes is None or len(r.boxes) == 0 or r.masks is None:
            return out
        masks = r.masks.data.cpu().numpy()
        for i in range(len(r.boxes)):
            out.append(SegResult(
                mask=_to_full_mask(masks[i], image.shape[:2]),
                box=tuple(float(v) for v in r.boxes.xyxy[i].tolist()),
                confidence=float(r.boxes.conf[i]),
                label=r.names[int(r.boxes.cls[i])],
                latency_ms=dt,
            ))
        out.sort(key=lambda s: s.confidence, reverse=True)
        return out


# ==================================================================================================
# máscara -> geometria
# ==================================================================================================

@dataclass
class MaskGeometryParams:
    work_size: int = 640  # maior dimensão da máscara de trabalho (velocidade)
    tau: float = 1.35  # limiar do trecho estreito, em múltiplos de W_ref
    ref_percentile: float = 25.0
    min_elongation: float = 3.0  # comprimento do braço / largura mínimo
    gap_fill_bins: int = 4  # buracos curtos no trecho estreito (oclusão) são ignorados
    edge_trim: float = 0.08  # fração das pontas do braço ignorada no ajuste das bordas
    min_pixels: int = 400


@dataclass
class WidthProfile:
    s: np.ndarray  # centro de cada fatia (px, imagem original)
    tmin: np.ndarray
    tmax: np.ndarray
    valid: np.ndarray  # fatia tem pixels

    @property
    def width(self) -> np.ndarray:
        return np.where(self.valid, self.tmax - self.tmin, 0.0)


def largest_component(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == best).astype(np.uint8)


def pca_axis(pts: np.ndarray) -> AxisFrame:
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    w, v = np.linalg.eigh(cov)
    return AxisFrame.make(mean, v[:, int(np.argmax(w))])


def width_profile(pts: np.ndarray, axis: AxisFrame, bin_px: float) -> WidthProfile:
    st = axis.to_local(pts)
    s0 = st[:, 0].min()
    idx = np.floor((st[:, 0] - s0) / bin_px).astype(int)
    nb = idx.max() + 1
    tmin = np.full(nb, np.inf)
    tmax = np.full(nb, -np.inf)
    np.minimum.at(tmin, idx, st[:, 1])
    np.maximum.at(tmax, idx, st[:, 1])
    valid = np.isfinite(tmin)
    s = s0 + (np.arange(nb) + 0.5) * bin_px
    return WidthProfile(s, np.where(valid, tmin, 0), np.where(valid, tmax, 0), valid)


def _median_filter(x: np.ndarray, k: int = 5) -> np.ndarray:
    if len(x) < k:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, k), axis=1)


def longest_run(flags: np.ndarray, gap_fill: int = 0) -> tuple[int, int] | None:
    """Maior trecho contínuo de True (preenchendo buracos de até ``gap_fill`` itens). Retorna [i0, i1]."""
    f = flags.copy()
    if gap_fill > 0:
        idx = np.flatnonzero(f)
        for a, b in zip(idx[:-1], idx[1:], strict=False):
            if 1 < b - a <= gap_fill + 1:
                f[a:b] = True
    best, cur_start, best_len = None, None, 0
    for i, v in enumerate(np.append(f, False)):
        if v and cur_start is None:
            cur_start = i
        elif not v and cur_start is not None:
            if i - cur_start > best_len:
                best_len, best = i - cur_start, (cur_start, i - 1)
            cur_start = None
    return best


def _fit_line(pts: np.ndarray) -> Line:
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return Line.make([x0, y0], [vx, vy])


def _neck_run(profile: WidthProfile, p: MaskGeometryParams):
    w = _median_filter(profile.width, 5)
    vw = w[profile.valid]
    if len(vw) < 5:
        return None
    w_ref = float(np.percentile(vw, p.ref_percentile))
    if w_ref <= 0:
        return None
    narrow = profile.valid & (w <= p.tau * w_ref)
    run = longest_run(narrow, p.gap_fill_bins)
    return (run, w_ref, w) if run is not None else None


def analyze_mask(mask: np.ndarray, frame_index: int = 0, params: MaskGeometryParams | None = None,
                 source: str = "") -> NeckDetection | None:
    """Extrai a geometria do braço de uma máscara (violão inteiro ou só braço). None se não der."""
    p = params or MaskGeometryParams()
    h, w = mask.shape[:2]
    f = min(1.0, p.work_size / max(h, w))
    small = cv2.resize((mask > 0).astype(np.uint8), (max(1, int(w * f)), max(1, int(h * f))),
                       interpolation=cv2.INTER_NEAREST) if f < 1.0 else (mask > 0).astype(np.uint8)
    small = largest_component(small)
    ys, xs = np.nonzero(small)
    if len(xs) < p.min_pixels * f * f:
        return None
    pts = (np.stack([xs, ys], axis=1).astype(np.float64) + 0.5) / f  # coordenadas originais
    bin_px = 1.0 / f

    axis = pca_axis(pts)
    found = _neck_run(width_profile(pts, axis, bin_px), p)
    if found is None:
        return None
    # Refina: PCA só nos pixels do trecho estreito, depois recalcula o perfil.
    for _ in range(2):
        prof = width_profile(pts, axis, bin_px)
        found = _neck_run(prof, p)
        if found is None:
            return None
        (i0, i1), w_ref, _ = found
        st = axis.to_local(pts)
        sel = (st[:, 0] >= prof.s[i0]) & (st[:, 0] <= prof.s[i1])
        if sel.sum() < 20:
            return None
        new_axis = pca_axis(pts[sel])
        if np.dot(new_axis.u, axis.u) < 0:
            new_axis = new_axis.flipped()
        axis = new_axis

    prof = width_profile(pts, axis, bin_px)
    found = _neck_run(prof, p)
    if found is None:
        return None
    (i0, i1), w_ref, w_smooth = found
    neck_len = prof.s[i1] - prof.s[i0]
    if neck_len < p.min_elongation * w_ref:
        return None

    # Bordas: extremos transversais das fatias do braço (sem as pontas).
    n = i1 - i0 + 1
    trim = int(n * p.edge_trim)
    ii = np.arange(i0 + trim, i1 - trim + 1)
    ii = ii[prof.valid[ii]]
    if len(ii) < 5:
        return None
    top = axis.to_image(np.stack([prof.s[ii], prof.tmin[ii]], axis=1))
    bot = axis.to_image(np.stack([prof.s[ii], prof.tmax[ii]], axis=1))
    edge_top, edge_bot = _fit_line(top), _fit_line(bot)
    da, db = edge_top.direction, edge_bot.direction
    if np.dot(da, axis.u) < 0:
        da = -da
    if np.dot(db, axis.u) < 0:
        db = -db
    edge_top, edge_bot = Line(edge_top.point, da), Line(edge_bot.point, db)

    # Eixo final: bissetriz das bordas, origem no meio do braço.
    u = normalize(da + db)
    s_mid = 0.5 * (prof.s[i0] + prof.s[i1])
    c = axis.to_image([[s_mid, 0.0]])[0]
    cut = Line(c, axis.n)


    pa, pb = intersect(cut, edge_top), intersect(cut, edge_bot)
    origin = 0.5 * (pa + pb) if pa is not None and pb is not None else c
    final = AxisFrame.make(origin, u)
    s_start = float(final.to_local(axis.to_image([[prof.s[i0], 0.0]]))[0, 0])
    s_end = float(final.to_local(axis.to_image([[prof.s[i1], 0.0]]))[0, 0])

    # edge_a = lado t < 0 do eixo final
    if final.to_local([edge_top.point])[0, 1] > final.to_local([edge_bot.point])[0, 1]:
        edge_top, edge_bot = edge_bot, edge_top

    # Lado da cabeça: menos máscara além da ponta do braço.
    valid_idx = np.flatnonzero(prof.valid)
    before = (i0 - valid_idx[0]) * bin_px
    after = (valid_idx[-1] - i1) * bin_px
    hint: int | None = None
    if max(before, after) > 0.15 * neck_len and min(before, after) < 0.6 * max(before, after):
        hint = 1 if after < before else -1  # +1: cabeça no sentido +u

    residual = float(np.std(w_smooth[i0:i1 + 1][prof.valid[i0:i1 + 1]]) / max(w_ref, 1e-6))
    conf = float(np.clip(neck_len / (6.0 * w_ref), 0, 1) * np.clip(1.0 - residual, 0, 1))
    full = mask if mask.dtype == np.uint8 else (mask > 0).astype(np.uint8) * 255
    return NeckDetection(
        frame_index=frame_index, mask=_neck_only_mask(full, final, (s_start, s_end), w_ref),
        axis=final, edge_a=edge_top, edge_b=edge_bot, extent=(s_start, s_end),
        nut_side_hint=hint, confidence=conf, full_mask=full, source=source,
    )


def _neck_only_mask(full: np.ndarray, axis: AxisFrame, extent: tuple[float, float], w_ref: float) -> np.ndarray:
    """Recorta da máscara completa só a faixa do braço (útil para o rastreador e para depuração)."""
    s0, s1 = extent
    half = 1.5 * w_ref
    poly = axis.to_image([[s0, -half], [s1, -half], [s1, half], [s0, half]]).astype(np.int32)
    band = np.zeros(full.shape[:2], np.uint8)
    cv2.fillConvexPoly(band, poly, 255)
    return cv2.bitwise_and((full > 0).astype(np.uint8) * 255, band)


# ==================================================================================================
# localizador
# ==================================================================================================

log = logging.getLogger(__name__)


class NeckLocator:
    """YOLOE prompt-free acha o violão (caixa + máscara) e ``analyze_mask`` tira dela a geometria do braço.

    Um modelo só. Medido na gravação do violão: achou o braço em 100% dos quadros, com o mesmo comprimento
    que a combinação YOLOE + SAM 2.1 (o SAM sozinho falhou em 13% e precisa da caixa do YOLOE como prompt).
    """

    def __init__(self, model: str = "models/yoloe-26s-seg-pf.pt", device: int | str = 0):
        self.yoloe = YoloeGuitarSegmenter(model, device=device)

    def warmup(self, shape=(720, 1280, 3)) -> None:
        self.yoloe.segment(np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8))

    def detect(self, image: np.ndarray, frame_index: int = 0) -> NeckDetection | None:
        t0 = time.perf_counter()
        for r in self.yoloe.segment(image)[:2]:
            det = analyze_mask(r.mask, frame_index, source="yoloe")
            if det is not None:
                det.confidence *= r.confidence ** 0.5
                det.latency_ms = (time.perf_counter() - t0) * 1000.0
                return det
        return None


# ==================================================================================================
# worker assíncrono
# ==================================================================================================

log = logging.getLogger(__name__)


@dataclass
class SegResponse:
    frame: Frame  # o quadro que foi segmentado (a detecção está nas coordenadas dele)
    detection: NeckDetection | None
    latency_ms: float
    error: str | None = None


class SegWorker:
    def __init__(self, locator_factory: Callable[[], object]):
        self._factory = locator_factory
        self._cond = threading.Condition()
        self._pending: Frame | None = None
        self._result: SegResponse | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self.ready = False  # modelo carregado
        self.busy = False
        self.load_error: str | None = None
        self.calls = 0

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="SegWorker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def submit(self, frame: Frame) -> None:
        with self._cond:
            self._pending = frame
            self._cond.notify_all()

    def poll(self) -> SegResponse | None:
        with self._cond:
            r, self._result = self._result, None
            return r

    @property
    def idle(self) -> bool:
        """Sem pedido pendente nem em processamento (bom momento para pedir de novo)."""
        with self._cond:
            return self.ready and not self.busy and self._pending is None

    def _loop(self) -> None:
        try:
            locator = self._factory()
            if hasattr(locator, "warmup"):
                locator.warmup()  # a 1ª inferência na GPU leva ~3 s; melhor pagar antes do 1º pedido
            self.ready = True
            log.info("SegWorker: modelo carregado")
        except Exception as exc:  # noqa: BLE001 - reportado na interface
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.error("SegWorker: falha ao carregar o modelo: %s", self.load_error)
            return
        while True:
            with self._cond:
                while self._running and self._pending is None:
                    self._cond.wait()
                if not self._running:
                    return
                frame = self._pending
                self._pending = None
                self.busy = True
            t0 = time.perf_counter()
            err = None
            det = None
            try:
                det = locator.detect(frame.image, frame_index=frame.index)
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                log.exception("SegWorker: erro na detecção")
            resp = SegResponse(frame, det, (time.perf_counter() - t0) * 1000.0, err)
            with self._cond:
                self._result = resp
                self.busy = False
                self.calls += 1
