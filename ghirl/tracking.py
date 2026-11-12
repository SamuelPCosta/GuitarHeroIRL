"""Rastreamento do braço: pontos LK presos ao quadro-chave K + homografia RANSAC, e o estimador.

Estados do estimador: "procurando" (ainda sem K), "rastreando" (H válida), "perdido" (o violão saiu do
quadro ou ficou coberto). Perdido, o estimador tenta reencontrar o MESMO K (ORB no quadro inteiro e na
região que a segmentação aponta), sem limite de tempo, e a calibração continua valendo. Se o violão está visível (a
segmentação acha) mas não casa com K por ``rekey_after_s``, vira um K novo; o anterior fica em
``prev_keyframe`` para a calibração ser transferida e reajustada pelos trastes. ``reset()`` (pedido de
recalibração) procura um K novo do zero.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from ghirl.capture import Frame
from ghirl.config import NeckConfig
from ghirl.geometry import Keyframe, NeckPose, apply_homography
from ghirl.neck import SegWorker

# ==================================================================================================
# rastreador
# ==================================================================================================

@dataclass
class TrackerParams:
    max_points: int = 300
    min_points: int = 80  # abaixo disso, repõe pontos
    quality_level: float = 0.01
    min_distance: int = 8
    lk_win: int = 21
    lk_levels: int = 3
    fb_max_err: float = 1.0  # px, verificação ida-e-volta
    ransac_px: float = 3.0
    min_inliers: int = 25
    max_rmse: float = 2.0
    scale_range: tuple[float, float] = (0.4, 2.5)
    work_scale: float = 0.5  # LK numa imagem reduzida (velocidade); coordenadas voltam à escala original


class OneEuro:
    """Filtro One Euro (Casiez et al., 2012): passa-baixa cuja frequência de corte cresce com a velocidade.

    Parado: corte baixo (``min_cutoff``), tira a tremedeira. Em movimento: corte sobe (``beta`` × velocidade),
    quase sem atraso.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.1, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x: np.ndarray | None = None
        self.dx: np.ndarray | None = None
        self.t = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        r = 2 * np.pi * cutoff * dt
        return r / (r + 1)

    def reset(self) -> None:
        self.x = None

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float64)
        if self.x is None or t <= self.t:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = t - self.t
        self.dx = self.dx + self._alpha(self.d_cutoff, dt) * ((x - self.x) / dt - self.dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        self.x = self.x + self._alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x


class PoseSmoother:
    """Suaviza a homografia K -> atual filtrando (One Euro) a posição na imagem de 4 cantos do braço."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.1):
        self.filter = OneEuro(min_cutoff, beta)
        self.key = None

    def __call__(self, t: float, H: np.ndarray | None, kf: Keyframe | None, key=None) -> np.ndarray | None:
        if H is None or kf is None:
            self.filter.reset()
            return H
        if key != self.key:  # quadro-chave novo: recomeça
            self.filter.reset()
            self.key = key
        quad = kf.detection.quad().astype(np.float32)
        smooth = self.filter(t, apply_homography(H, quad)).astype(np.float32)
        return cv2.getPerspectiveTransform(quad, smooth)


def to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def homography_scale(H: np.ndarray, at: np.ndarray) -> float:
    """Escala local (raiz do determinante do Jacobiano) da homografia no ponto ``at``."""
    x, y = at
    h = H / H[2, 2]
    w = h[2, 0] * x + h[2, 1] * y + 1.0
    px = (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / w
    py = (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / w
    J = np.array([[h[0, 0] - px * h[2, 0], h[0, 1] - px * h[2, 1]],
                  [h[1, 0] - py * h[2, 0], h[1, 1] - py * h[2, 1]]]) / w
    d = np.linalg.det(J)
    return float(np.sqrt(d)) if d > 0 else 0.0


class FeatureTracker:
    def __init__(self, params: TrackerParams | None = None):
        self.p = params or TrackerParams()
        self.keyframe: Keyframe | None = None
        self._prev_gray: np.ndarray | None = None
        self._pts_k = np.empty((0, 2), np.float32)  # coordenadas em K
        self._pts_cur = np.empty((0, 2), np.float32)  # coordenadas no quadro anterior/atual
        self.H: np.ndarray | None = None
        self.last_pose: NeckPose | None = None

    # ------------------------------------------------------------------ utilitários

    def _small(self, gray: np.ndarray) -> np.ndarray:
        s = self.p.work_scale
        return gray if s == 1.0 else cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    def _detect(self, gray: np.ndarray, mask: np.ndarray, max_n: int) -> np.ndarray:
        if max_n <= 0 or mask is None or not mask.any():
            return np.empty((0, 2), np.float32)
        s = self.p.work_scale
        small_mask = cv2.resize(mask, (int(gray.shape[1] * s), int(gray.shape[0] * s)),
                                interpolation=cv2.INTER_NEAREST) if s != 1.0 else mask
        pts = cv2.goodFeaturesToTrack(self._small(gray), maxCorners=max_n, qualityLevel=self.p.quality_level,
                                      minDistance=max(1, int(self.p.min_distance * s)), mask=small_mask)
        if pts is None:
            return np.empty((0, 2), np.float32)
        return (pts.reshape(-1, 2) / s).astype(np.float32)

    # ------------------------------------------------------------------ API

    def init(self, frame: Frame, keyframe: Keyframe, exclude_mask: np.ndarray | None = None) -> None:
        """Inicia no próprio quadro-chave (H = identidade)."""
        self.keyframe = keyframe
        gray = keyframe.gray
        mask = keyframe.track_mask.copy()
        if exclude_mask is not None:
            mask[exclude_mask > 0] = 0
        pts = self._detect(gray, mask, self.p.max_points)
        self._pts_k = pts.copy()
        self._pts_cur = pts.copy()
        self._prev_gray = gray
        self.H = np.eye(3)
        self.last_pose = NeckPose(frame.index, self.H.copy(), 1.0, len(pts), 0.0, "ok")

    def update(self, frame: Frame, exclude_mask: np.ndarray | None = None) -> NeckPose:
        if self.keyframe is None or self._prev_gray is None:
            raise RuntimeError("FeatureTracker.update() antes de init()")
        p = self.p
        gray = to_gray(frame.image)
        s = p.work_scale
        prev_s, cur_s = self._small(self._prev_gray), self._small(gray)
        n_before = len(self._pts_cur)

        if n_before > 0:
            lk = dict(winSize=(p.lk_win, p.lk_win), maxLevel=p.lk_levels,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
            # 1) previsão quadro a quadro (robusta a movimento rápido)
            p0 = (self._pts_cur * s).reshape(-1, 1, 2)
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_s, cur_s, p0, None, **lk)
            ok_pred = st1.ravel() == 1
            # Movimento GLOBAL quadro a quadro (RANSAC): os trastes são repetitivos e, com movimento
            # rápido, pontos isolados "pulam" para o traste vizinho. A previsão global corrige isso.
            H_pred = self.H
            if ok_pred.sum() >= 8:
                cv2.setRNGSeed(0)
                Hff, _ = cv2.findHomography(p0.reshape(-1, 2)[ok_pred] / s, p1.reshape(-1, 2)[ok_pred] / s,
                                            cv2.RANSAC, p.ransac_px)
                if Hff is not None:
                    H_pred = Hff @ self.H
            # 2) refinamento contra o QUADRO-CHAVE deformado pela pose prevista: cada ponto volta a
            #    ser casado com a aparência original em K, o que elimina a deriva acumulada.
            S = np.diag([s, s, 1.0])
            Hs = S @ H_pred @ np.linalg.inv(S)
            kf_warp = cv2.warpPerspective(self._small(self.keyframe.gray), Hs, (cur_s.shape[1], cur_s.shape[0]))
            q0 = (apply_homography(H_pred, self._pts_k) * s).astype(np.float32).reshape(-1, 1, 2)
            # o próprio q0 é o chute inicial: no quadro-chave deformado o ponto já está "no lugar"
            q1, st3, _ = cv2.calcOpticalFlowPyrLK(kf_warp, cur_s, q0, q0.copy(),
                                                  flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
            q0b, st4, _ = cv2.calcOpticalFlowPyrLK(cur_s, kf_warp, q1, None, **lk)
            fb = np.linalg.norm((q0 - q0b).reshape(-1, 2), axis=1) / s
            good_k = (st3.ravel() == 1) & (st4.ravel() == 1) & (fb < p.fb_max_err)
            # Com borrão de movimento o quadro atual não se parece com o quadro-chave nítido e o
            # casamento acima falha. Nesses pontos vale a posição quadro a quadro, se concordar com o
            # movimento global. A identidade do ponto (coordenada em K) não muda, então não há deriva:
            # quando a imagem volta a ficar nítida, o ponto é recasado com o quadro-chave.
            pred = apply_homography(H_pred, self._pts_k)
            ff = p1.reshape(-1, 2) / s
            good_f = ok_pred & (np.linalg.norm(ff - pred, axis=1) < p.ransac_px + 1.0)
            cur = np.where(good_k[:, None], q1.reshape(-1, 2) / s, ff)
            good = good_k | good_f
            h, w = gray.shape
            inside = (cur[:, 0] >= 0) & (cur[:, 0] < w) & (cur[:, 1] >= 0) & (cur[:, 1] < h)
            good &= inside
            if exclude_mask is not None and good.any():
                xi = np.clip(cur[:, 0].astype(int), 0, w - 1)
                yi = np.clip(cur[:, 1].astype(int), 0, h - 1)
                good &= exclude_mask[yi, xi] == 0
            self._pts_k, self._pts_cur = self._pts_k[good], cur[good].astype(np.float32)

        pose = self._estimate(frame.index)

        # Reposição de pontos na região do braço projetada (menos as mãos).
        if pose.H is not None and pose.status != "perdido" and len(self._pts_cur) < p.min_points:
            self._replenish(gray, pose.H, exclude_mask)

        self._prev_gray = gray
        self.last_pose = pose
        return pose

    def _estimate(self, frame_index: int) -> NeckPose:
        p = self.p
        n = len(self._pts_k)
        if n < 8:
            return NeckPose(frame_index, None, 0.0, 0, float("inf"), "perdido", {"points": n})
        cv2.setRNGSeed(0)  # RANSAC determinístico (RNF-08)
        H, inl = cv2.findHomography(self._pts_k, self._pts_cur, cv2.RANSAC, p.ransac_px, maxIters=2000,
                                    confidence=0.995)
        if H is None:
            return NeckPose(frame_index, None, 0.0, 0, float("inf"), "perdido", {"points": n})
        inl = inl.ravel().astype(bool)
        n_inl = int(inl.sum())
        proj = apply_homography(H, self._pts_k[inl])
        rmse = float(np.sqrt(np.mean(np.sum((proj - self._pts_cur[inl]) ** 2, axis=1)))) if n_inl else np.inf
        center = self._pts_k.mean(axis=0)
        scale = homography_scale(H, center)
        plausible = p.scale_range[0] <= scale <= p.scale_range[1]
        # mantém só os inliers (outliers saem do conjunto rastreado)
        self._pts_k, self._pts_cur = self._pts_k[inl], self._pts_cur[inl]

        q_inl = np.clip(n_inl / (2.0 * p.min_inliers), 0, 1)
        q_ratio = n_inl / n
        q_rmse = np.clip(1.0 - rmse / (2.0 * p.max_rmse), 0, 1)
        quality = float(q_inl * (0.5 + 0.5 * q_ratio) * (0.5 + 0.5 * q_rmse)) if plausible else 0.0
        if n_inl < p.min_inliers // 2 or not plausible:
            status = "perdido"
        elif quality >= 0.6:
            status = "ok"
        elif quality >= 0.3:
            status = "degradado"
        else:
            status = "perdido"
        self.H = H if status != "perdido" else self.H
        return NeckPose(frame_index, H if status != "perdido" else None, quality, n_inl, rmse, status,
                        {"points": n, "scale": scale})

    def _replenish(self, gray: np.ndarray, H: np.ndarray, exclude_mask: np.ndarray | None) -> None:
        kf = self.keyframe
        h, w = gray.shape
        region = cv2.warpPerspective(kf.track_mask, H, (w, h), flags=cv2.INTER_NEAREST)
        if exclude_mask is not None:
            region[exclude_mask > 0] = 0
        # não detecta perto dos pontos que já existem
        for x, y in self._pts_cur.astype(int):
            cv2.circle(region, (int(x), int(y)), self.p.min_distance, 0, -1)
        new = self._detect(gray, region, self.p.max_points - len(self._pts_cur))
        if len(new) == 0:
            return
        # Confere cada ponto novo contra a aparência do quadro-chave: LK do quadro atual para o
        # quadro-chave deformado (ida e volta). Só entram os pontos que existem de fato em K, com a
        # coordenada em K corrigida pelo casamento (e não só pela homografia, que tem erro).
        s = self.p.work_scale
        S = np.diag([s, s, 1.0])
        cur_s = self._small(gray)
        kf_warp = cv2.warpPerspective(self._small(kf.gray), S @ H @ np.linalg.inv(S), (cur_s.shape[1], cur_s.shape[0]))
        lk = dict(winSize=(self.p.lk_win, self.p.lk_win), maxLevel=1,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        a = (new * s).reshape(-1, 1, 2).astype(np.float32)
        b, st1, _ = cv2.calcOpticalFlowPyrLK(cur_s, kf_warp, a, a.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
        a2, st2, _ = cv2.calcOpticalFlowPyrLK(kf_warp, cur_s, b, None, **lk)
        fb = np.linalg.norm((a - a2).reshape(-1, 2), axis=1) / s
        shift = np.linalg.norm((b - a).reshape(-1, 2), axis=1) / s
        ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < self.p.fb_max_err) & (shift < 3.0)
        if not ok.any():
            return
        new_k = apply_homography(np.linalg.inv(H), (b.reshape(-1, 2)[ok] / s)).astype(np.float32)
        self._pts_k = np.vstack([self._pts_k, new_k])
        self._pts_cur = np.vstack([self._pts_cur, new[ok]])

    def relocalize(self, frame: Frame, exclude_mask: np.ndarray | None = None,
                   search_mask: np.ndarray | None = None) -> NeckPose | None:
        """Reencontra o braço por ORB contra o quadro-chave (``orb_register``) e reinicia os pontos LK."""
        if self.keyframe is None:
            return None
        gray = to_gray(frame.image)
        found = orb_register(self.keyframe, gray, search_mask)
        if found is None or not (self.p.scale_range[0] <= found[2] <= self.p.scale_range[1]):
            return None
        H, n_inl, _ = found
        self.H = H
        self._pts_k = np.empty((0, 2), np.float32)
        self._pts_cur = np.empty((0, 2), np.float32)
        self._replenish(gray, H, exclude_mask)
        self._prev_gray = gray
        pose = NeckPose(frame.index, H, 0.5, n_inl, 0.0, "degradado", {"relocalized": True})
        self.last_pose = pose
        return pose

    @property
    def points(self) -> tuple[np.ndarray, np.ndarray]:
        """(pontos em K, pontos no quadro atual) — para depuração."""
        return self._pts_k, self._pts_cur


def orb_register(kf: Keyframe, gray: np.ndarray, mask: np.ndarray | None = None,
                 min_inliers: int = 20) -> tuple[np.ndarray, int, float] | None:
    """Homografia K -> ``gray`` por ORB + teste da razão + RANSAC. Devolve (H, inliers, escala) ou None.

    O teste da razão descarta casamentos ambíguos. Isso importa: as marcações das casas 3, 5, 7 e 9 se
    repetem a cada 2 casas, e um casamento "guiado" por uma pose aproximada chegou a travar 2 casas fora.
    """
    if kf.orb_descriptors is None or len(kf.orb_keypoints) < min_inliers:
        return None
    kps, des = _orb().detectAndCompute(gray, mask)
    if des is None or len(kps) < min_inliers:
        return None
    src_all = np.float32(kf.orb_keypoints)
    dst_all = np.float32([k.pt for k in kps])
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(kf.orb_descriptors, des, k=2)
    good = [m for m, *rest in matches if rest and m.distance < 0.75 * rest[0].distance]
    if len(good) < min_inliers:
        return None
    src = src_all[[m.queryIdx for m in good]]
    dst = dst_all[[m.trainIdx for m in good]]
    cv2.setRNGSeed(0)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or int(inl.sum()) < min_inliers:
        return None
    return H, int(inl.sum()), homography_scale(H, src.mean(axis=0))


def register_keyframes(old: Keyframe, new: Keyframe) -> np.ndarray | None:
    """Homografia do quadro-chave antigo para o novo (para levar a calibração junto)."""
    r = orb_register(old, new.gray, cv2.dilate(new.track_mask, np.ones((31, 31), np.uint8)))
    return r[0] if r is not None and 0.4 <= r[2] <= 2.5 else None


def _orb():
    return cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8, fastThreshold=12)


def build_keyframe(frame: Frame, detection, erode_px: int = 4, exclude_mask: np.ndarray | None = None) -> Keyframe:
    """Quadro-chave a partir de uma detecção: região rastreável = violão (máscara completa, erodida
    para evitar pontos na fronteira com o fundo), menos as mãos; com descritores ORB para relocalizar."""
    mask = detection.full_mask if detection.full_mask is not None else detection.mask
    m = (mask > 0).astype(np.uint8) * 255
    if erode_px > 0:
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)))
    if exclude_mask is not None:
        m[exclude_mask > 0] = 0
    gray = to_gray(frame.image)
    kps, des = _orb().detectAndCompute(gray, m)
    return Keyframe(frame_index=frame.index, gray=gray, detection=detection, track_mask=m,
                    orb_keypoints=tuple(k.pt for k in kps), orb_descriptors=des)


# ==================================================================================================
# estimador
# ==================================================================================================

log = logging.getLogger(__name__)

EstimatorStatus = Literal["procurando", "rastreando", "perdido"]


class NeckEstimator:
    def __init__(self, cfg: NeckConfig, worker: SegWorker, min_confidence: float = 0.35,
                 history_frames: int = 120):
        self.cfg = cfg
        self.worker = worker
        self.min_confidence = min_confidence
        self.tracker = FeatureTracker()
        self.keyframe: Keyframe | None = None
        self.prev_keyframe: Keyframe | None = None  # K anterior, quando o K foi trocado sozinho
        self._unmatched_since: float | None = None  # violão visível mas sem casar com K desde...
        self._unmatched_last = 0.0
        self.status: EstimatorStatus = "procurando"
        self.pose: NeckPose | None = None
        self.keyframe_id = 0  # muda a cada novo K (a calibração depende dele)
        self._bad_frames = 0
        self.last_detection_ms = 0.0
        # Últimos quadros em cinza: a segmentação chega atrasada (0,2–3 s), então o rastreador é
        # iniciado no quadro segmentado e "avançado" por estes quadros até o atual.
        self._history: deque[Frame] = deque(maxlen=history_frames)
        self.max_catchup_steps = 25

    def reset(self) -> None:
        self.keyframe = None
        self.prev_keyframe = None
        self.pose = None
        self.status = "procurando"
        self._bad_frames = 0

    def _catch_up(self, from_index: int, current_index: int) -> bool:
        """Avança o rastreador do quadro segmentado até o anterior ao atual, pelos quadros guardados."""
        pending = [f for f in self._history if from_index < f.index < current_index]
        if not pending:
            return True
        stride = max(1, -(-len(pending) // self.max_catchup_steps))  # ceil
        steps = pending[stride - 1::stride]
        if steps[-1] is not pending[-1]:
            steps.append(pending[-1])
        for f in steps:
            if self.tracker.update(f).status == "perdido":
                return False
        return True

    def update(self, frame: Frame, hand_mask: np.ndarray | None = None) -> NeckPose | None:
        self._history.append(Frame(to_gray(frame.image), frame.ts, frame.index))
        resp = self.worker.poll()
        if resp is not None:
            self.last_detection_ms = resp.latency_ms

        if self.status == "procurando":
            det = resp.detection if resp is not None else None
            if det is not None and det.confidence >= self.min_confidence:
                if not self._new_keyframe(resp, frame):
                    return None
            elif self.worker.idle:
                self.worker.submit(frame)
                return None
            else:
                return None

        if self.status == "rastreando":
            pose = self.tracker.update(frame, hand_mask)
            if pose.status == "perdido":
                self._bad_frames += 1
                if self._bad_frames >= self.cfg.lost_frames:
                    self.status = "perdido"
                    log.info("braço perdido")
            else:
                self._bad_frames = 0
            self.pose = pose if pose.H is not None else self.pose
            return pose if pose.H is not None else None

        # perdido: o quadro-chave (e a calibração) continuam valendo. Tenta sempre:
        # (a) quando a segmentação acha o violão de novo, ORB contra K só na região do violão detectado,
        #     e depois alcança o quadro atual pelos quadros guardados;
        # (b) ORB contra K no quadro inteiro, a cada 3 quadros.
        det = resp.detection if resp is not None else None
        if det is not None and det.confidence >= self.min_confidence:
            # cronômetro de "visível sem casar": recomeça se houve um intervalo sem o violão
            if self._unmatched_since is None or frame.ts - self._unmatched_last > 1.0:
                self._unmatched_since = frame.ts
            self._unmatched_last = frame.ts
            area = det.full_mask if det.full_mask is not None else det.mask
            search = cv2.dilate(area, np.ones((31, 31), np.uint8)) if area is not None else None
            if self.tracker.relocalize(resp.frame, None, search_mask=search) is not None \
                    and self._catch_up(resp.frame.index, frame.index + 1):
                return self._relocked(self.tracker.last_pose, "pela segmentação")
            if frame.ts - self._unmatched_since >= self.cfg.rekey_after_s:
                old = self.keyframe
                if self._new_keyframe(resp, frame):
                    self.prev_keyframe = old
                    log.info("violão visível mas sem casar com o quadro-chave: quadro-chave novo")
                    pose = self.tracker.update(frame, hand_mask)
                    self.pose = pose if pose.H is not None else None
                    return self.pose
        if self.worker.idle:
            self.worker.submit(frame)
        if frame.index % 3 == 0:
            pose = self.tracker.relocalize(frame, hand_mask)
            if pose is not None:
                return self._relocked(pose, "por ORB")
        return None

    def _new_keyframe(self, resp, frame: Frame) -> bool:
        """K novo a partir de uma segmentação (atrasada): inicia o rastreador e alcança o quadro atual."""
        det = resp.detection
        kf = build_keyframe(resp.frame, det, exclude_mask=None)
        self.tracker.init(resp.frame, kf)
        if not self._catch_up(resp.frame.index, frame.index):
            log.info("braço encontrado mas perdido ao alcançar o quadro atual; procurando de novo")
            self.tracker.keyframe = self.keyframe  # continua tentando o K antigo, se havia
            self.worker.submit(frame)
            return False
        self.keyframe = kf
        self.keyframe_id += 1
        self.status = "rastreando"
        self._bad_frames = 0
        self._unmatched_since = None
        log.info("braço encontrado (%s, conf %.2f, %.0f ms, %d quadros de atraso)", det.source,
                 det.confidence, resp.latency_ms, frame.index - resp.frame.index)
        return True

    def _relocked(self, pose: NeckPose, how: str) -> NeckPose:
        self._unmatched_since = None
        self.status = "rastreando"
        self._bad_frames = 0
        self.pose = pose
        log.info("braço reencontrado %s (mesmo quadro-chave, calibração mantida)", how)
        return pose
