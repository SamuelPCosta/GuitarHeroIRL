"""Rastreador do braço: pontos LK presos ao quadro-chave + homografia RANSAC (T2.5; design §7.1).

Cada ponto rastreado guarda sua coordenada em K. A cada quadro, LK piramidal leva os pontos do
quadro anterior ao atual (com verificação ida-e-volta) e a homografia é estimada DIRETAMENTE de K
para o quadro atual (não compondo quadro a quadro), o que limita a deriva. Pontos sobre a mão são
descartados (RF-13) e novos pontos são repostos na região do braço projetada no quadro atual.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ghirl.capture.base import Frame
from ghirl.neck.base import Keyframe, NeckPose
from ghirl.utils.geometry import apply_homography


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
            # 2) refinamento contra o QUADRO-CHAVE deformado pela pose anterior: cada ponto volta a
            #    ser casado com a aparência original em K, o que elimina a deriva acumulada.
            S = np.diag([s, s, 1.0])
            Hs = S @ self.H @ np.linalg.inv(S)
            kf_warp = cv2.warpPerspective(self._small(self.keyframe.gray), Hs, (cur_s.shape[1], cur_s.shape[0]))
            q0 = (apply_homography(self.H, self._pts_k) * s).astype(np.float32).reshape(-1, 1, 2)
            init = np.where(ok_pred[:, None, None], p1, q0).astype(np.float32)
            q1, st3, _ = cv2.calcOpticalFlowPyrLK(kf_warp, cur_s, q0, init.copy(),
                                                  flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
            q0b, st4, _ = cv2.calcOpticalFlowPyrLK(cur_s, kf_warp, q1, None, **lk)
            fb = np.linalg.norm((q0 - q0b).reshape(-1, 2), axis=1) / s
            good = (st3.ravel() == 1) & (st4.ravel() == 1) & (fb < p.fb_max_err)
            cur = q1.reshape(-1, 2) / s
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
        new_k = apply_homography(np.linalg.inv(H), new).astype(np.float32)
        self._pts_k = np.vstack([self._pts_k, new_k])
        self._pts_cur = np.vstack([self._pts_cur, new])

    def relocalize(self, frame: Frame, exclude_mask: np.ndarray | None = None,
                   min_inliers: int = 30) -> NeckPose | None:
        """Reencontra o braço por ORB contra o quadro-chave (T2.7, versão mínima).

        Se der certo, reinicia os pontos rastreados no quadro atual e devolve a pose; senão, None.
        """
        kf = self.keyframe
        if kf is None or kf.orb_descriptors is None or len(kf.orb_keypoints) < min_inliers:
            return None
        gray = to_gray(frame.image)
        orb = _orb()
        kps, des = orb.detectAndCompute(gray, None)
        if des is None or len(kps) < min_inliers:
            return None
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(kf.orb_descriptors, des, k=2)
        good = [m for m, *rest in matches if rest and m.distance < 0.75 * rest[0].distance]
        if len(good) < min_inliers:
            return None
        src = np.float32([kf.orb_keypoints[m.queryIdx] for m in good])
        dst = np.float32([kps[m.trainIdx].pt for m in good])
        cv2.setRNGSeed(0)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None or int(inl.sum()) < min_inliers:
            return None
        scale = homography_scale(H, src.mean(axis=0))
        if not (self.p.scale_range[0] <= scale <= self.p.scale_range[1]):
            return None
        # reinicia os pontos LK na região do braço projetada
        self.H = H
        self._pts_k = np.empty((0, 2), np.float32)
        self._pts_cur = np.empty((0, 2), np.float32)
        self._replenish(gray, H, exclude_mask)
        self._prev_gray = gray
        pose = NeckPose(frame.index, H, 0.5, int(inl.sum()), 0.0, "degradado", {"relocalized": True})
        self.last_pose = pose
        return pose

    @property
    def points(self) -> tuple[np.ndarray, np.ndarray]:
        """(pontos em K, pontos no quadro atual) — para depuração."""
        return self._pts_k, self._pts_cur


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
