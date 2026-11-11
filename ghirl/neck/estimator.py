"""Estimador do braço: localizador assíncrono + rastreador + perda/relocalização (T2.4–T2.7 mínimo).

Estados (``status``):
    "procurando"  sem quadro-chave; pede segmentações ao worker sempre que ele está livre
    "rastreando"  pose válida a cada quadro (H: K -> quadro atual)
    "perdido"     pose inválida por >= ``lost_frames`` quadros; tenta relocalizar por ORB contra o
                  mesmo K (a calibração continua válida); após ``relock_timeout_s`` desiste e volta a
                  "procurando" (novo K -> a calibração precisa ser refeita)

O resto do app só usa ``update(frame, hand_mask) -> NeckPose | None`` e ``keyframe``.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from ghirl.app.config import NeckConfig
from ghirl.capture.base import Frame
from ghirl.neck.base import Keyframe, NeckPose
from ghirl.neck.feature_tracker import FeatureTracker, build_keyframe
from ghirl.neck.seg_worker import SegWorker

log = logging.getLogger(__name__)

EstimatorStatus = Literal["procurando", "rastreando", "perdido"]


class NeckEstimator:
    def __init__(self, cfg: NeckConfig, worker: SegWorker, min_confidence: float = 0.35):
        self.cfg = cfg
        self.worker = worker
        self.min_confidence = min_confidence
        self.tracker = FeatureTracker()
        self.keyframe: Keyframe | None = None
        self.status: EstimatorStatus = "procurando"
        self.pose: NeckPose | None = None
        self.keyframe_id = 0  # muda a cada novo K (a calibração depende dele)
        self._bad_frames = 0
        self._lost_since: float | None = None
        self.last_detection_ms = 0.0

    def reset(self) -> None:
        self.keyframe = None
        self.pose = None
        self.status = "procurando"
        self._bad_frames = 0
        self._lost_since = None

    def update(self, frame: Frame, hand_mask: np.ndarray | None = None) -> NeckPose | None:
        resp = self.worker.poll()
        if resp is not None:
            self.last_detection_ms = resp.latency_ms

        if self.status == "procurando":
            det = resp.detection if resp is not None else None
            if det is not None and det.confidence >= self.min_confidence:
                self.keyframe = build_keyframe(resp.frame, det, exclude_mask=None)
                self.keyframe_id += 1
                self.tracker.init(resp.frame, self.keyframe)
                self.status = "rastreando"
                self._bad_frames = 0
                log.info("braço encontrado (%s, conf %.2f, %.0f ms)", det.source, det.confidence, resp.latency_ms)
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
                    self._lost_since = frame.ts
                    log.info("braço perdido")
            else:
                self._bad_frames = 0
            self.pose = pose if pose.H is not None else self.pose
            return pose if pose.H is not None else None

        # perdido
        if frame.index % 3 == 0:
            pose = self.tracker.relocalize(frame, hand_mask)
            if pose is not None:
                self.status = "rastreando"
                self._bad_frames = 0
                self.pose = pose
                log.info("braço relocalizado")
                return pose
        if self._lost_since is not None and frame.ts - self._lost_since > self.cfg.relock_timeout_s:
            log.info("relocalização expirou; procurando o braço de novo")
            self.reset()
        return None
