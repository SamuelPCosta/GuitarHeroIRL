"""Segmentadores prontos (zero-shot / vocabulário aberto) para achar o violão (T2.2).

- ``YoloeGuitarSegmenter``: YOLOE *prompt-free* (vocabulário embutido de 4585 classes, inclui
  "guitar" e "bass guitar"); devolve caixa + máscara por quadro (~24 ms na RTX 3060).
  Não precisa do codificador de texto CLIP.
- ``Sam2BoxSegmenter``: SAM 2.1 com a caixa do YOLOE como prompt (máscara refinada, ~115 ms).
- ``Sam3TextSegmenter``: SAM 3 com prompt de texto ("guitar neck"), só se ``models/sam3.pt``
  existir (pesos com acesso controlado no Hugging Face).

Nenhum modelo é treinado ou ajustado (restrição R1).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

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


class Sam2BoxSegmenter:
    name = "sam2"

    def __init__(self, model_path: str = "models/sam2.1_t.pt", device: int | str = 0):
        from ultralytics import SAM

        self.model = SAM(model_path)
        self.device = device

    def segment(self, image: np.ndarray, box: tuple[float, float, float, float] | None = None,
                points: np.ndarray | None = None) -> SegResult | None:
        kwargs = {}
        if box is not None:
            kwargs["bboxes"] = [list(box)]
        if points is not None:
            kwargs["points"] = np.asarray(points, np.float32).tolist()
            kwargs["labels"] = [1] * len(kwargs["points"])
        if not kwargs:
            raise ValueError("SAM 2 precisa de caixa ou pontos como prompt")
        t0 = time.perf_counter()
        r = self.model(image, verbose=False, device=self.device, **kwargs)[0]
        _sync()
        dt = (time.perf_counter() - t0) * 1000.0
        if r.masks is None or len(r.masks.data) == 0:
            return None
        mask = _to_full_mask(r.masks.data[0].cpu().numpy(), image.shape[:2])
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        return SegResult(mask=mask, box=(xs.min(), ys.min(), xs.max(), ys.max()), confidence=1.0,
                         label="sam2", latency_ms=dt)


class Sam3TextSegmenter:
    name = "sam3"

    def __init__(self, model_path: str = "models/sam3.pt", prompts=("guitar neck", "guitar"),
                 conf: float = 0.25, device: int | str = 0):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"{model_path} não encontrado. O SAM 3 exige pedir acesso em huggingface.co/facebook/sam3 "
                "e baixar sam3.pt manualmente para models/."
            )
        from ultralytics.models.sam import SAM3SemanticPredictor

        self.prompts = list(prompts)
        self.predictor = SAM3SemanticPredictor(overrides=dict(
            conf=conf, task="segment", mode="predict", model=model_path, half=True, verbose=False,
            device=device))

    def segment(self, image: np.ndarray) -> list[SegResult]:
        t0 = time.perf_counter()
        self.predictor.set_image(image)
        r = self.predictor(text=self.prompts)[0]
        _sync()
        dt = (time.perf_counter() - t0) * 1000.0
        out: list[SegResult] = []
        if r.masks is None:
            return out
        masks = r.masks.data.cpu().numpy()
        for i in range(len(masks)):
            box = r.boxes.xyxy[i].tolist() if r.boxes is not None else (0, 0, 0, 0)
            conf = float(r.boxes.conf[i]) if r.boxes is not None else 0.0
            label = r.names[int(r.boxes.cls[i])] if r.boxes is not None else "sam3"
            out.append(SegResult(_to_full_mask(masks[i], image.shape[:2]), tuple(box), conf, label, dt))
        out.sort(key=lambda s: s.confidence, reverse=True)
        return out
