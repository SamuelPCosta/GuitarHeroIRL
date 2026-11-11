"""Benchmark dos segmentadores no violão (T2.2).

Amostra quadros de uma gravação, roda YOLOE (prompt-free, classe "guitar") e SAM 2.1 (caixa do
YOLOE como prompt; tiny e small) e, se ``models/sam3.pt`` existir, SAM 3 por texto. Mede latência
(com aquecimento) e VRAM e salva sobreposições em ``data/diag/seg/`` para inspeção visual.

Uso:
    python -m ghirl.neck.bench_seg --rec data/rec/violao_01 --frames 5
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import cv2
import numpy as np

from ghirl.capture.replay import ReplaySource
from ghirl.neck.segmenters import Sam2BoxSegmenter, Sam3TextSegmenter, SegResult, YoloeGuitarSegmenter


def sample_frames(rec: str, n: int) -> list[np.ndarray]:
    with ReplaySource(rec) as src:
        total = src.frame_count
        wanted = set(np.linspace(0, max(0, total - 1), n).astype(int).tolist())
        out = []
        i = 0
        while (f := src.read_latest()) is not None:
            if i in wanted:
                out.append(f.image)
            i += 1
    return out


def overlay(img: np.ndarray, res: SegResult | None, color, text: str) -> np.ndarray:
    v = img.copy()
    if res is not None:
        m = res.mask > 0
        v[m] = (v[m] * 0.45 + np.array(color) * 0.55).astype(np.uint8)
        x0, y0, x1, y1 = (int(c) for c in res.box)
        cv2.rectangle(v, (x0, y0), (x1, y1), color, 2)
    cv2.putText(v, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(v, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", default="data/rec/violao_01")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--out", default="data/diag/seg")
    args = ap.parse_args()
    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = sample_frames(args.rec, args.frames)
    print(f"{len(frames)} quadros de {args.rec}")

    yoloe = YoloeGuitarSegmenter()
    sams = {name: Sam2BoxSegmenter(f"models/{name}.pt") for name in ("sam2.1_t", "sam2.1_s")}
    sam3 = None
    try:
        sam3 = Sam3TextSegmenter()
    except FileNotFoundError as exc:
        print(f"(SAM 3 pulado: {exc})")

    # aquecimento
    yoloe.segment(frames[0])
    for s in sams.values():
        s.segment(frames[0], box=(0, 0, frames[0].shape[1] - 1, frames[0].shape[0] - 1))
    torch.cuda.reset_peak_memory_stats()

    lat: dict[str, list[float]] = {k: [] for k in ["yoloe", *sams, "sam3"]}
    found: dict[str, int] = {k: 0 for k in lat}
    for i, img in enumerate(frames):
        dets = yoloe.segment(img)
        best = dets[0] if dets else None
        lat["yoloe"].append(dets[0].latency_ms if dets else float("nan"))
        found["yoloe"] += int(best is not None)
        title = f"YOLOE {best.label} {best.confidence:.2f}" if best else "YOLOE: nada"
        tiles = [overlay(img, best, (0, 200, 255), title)]
        for name, s in sams.items():
            r = s.segment(img, box=best.box) if best else None
            if r is not None:
                lat[name].append(r.latency_ms)
                found[name] += 1
            tiles.append(overlay(img, r, (60, 220, 60), f"{name} {r.latency_ms:.0f} ms" if r else f"{name}: -"))
        if sam3 is not None:
            r3 = sam3.segment(img)
            b3 = r3[0] if r3 else None
            if b3 is not None:
                lat["sam3"].append(b3.latency_ms)
                found["sam3"] += 1
            tiles.append(overlay(img, b3, (255, 120, 60), f"SAM3 {b3.label}" if b3 else "SAM3: nada"))
        small = [cv2.resize(t, (640, 360)) for t in tiles]
        while len(small) % 2:
            small.append(np.zeros_like(small[0]))
        grid = np.vstack([np.hstack(small[j:j + 2]) for j in range(0, len(small), 2)])
        cv2.imwrite(str(out / f"quadro_{i:02d}.jpg"), grid)

    print(f"\n{'modelo':<10}{'achou':>8}{'lat. mediana (ms)':>20}")
    for k, v in lat.items():
        v = [x for x in v if x == x]
        if k == "sam3" and sam3 is None:
            continue
        med = statistics.median(v) if v else float("nan")
        print(f"{k:<10}{found[k]:>5}/{len(frames):<2}{med:>20.1f}")
    print(f"VRAM pico: {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
    print(f"Sobreposições em {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
