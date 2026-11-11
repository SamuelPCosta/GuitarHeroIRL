"""Diagnóstico da webcam (T1.3).

Testa backends (DSHOW, MSMF) com auto-exposição ligada e desligada, mostra o que o driver aceitou
e mede o FPS real. Salva um quadro de amostra de cada combinação em ``data/diag/``.

Uso:
    python -m ghirl.capture.diagnose                 # 10 s por combinação
    python -m ghirl.capture.diagnose --seconds 5 --set camera.index=1
"""

from __future__ import annotations

import argparse
import copy
import statistics
import time
from pathlib import Path

import cv2

from ghirl.app.config import add_config_args, config_from_args
from ghirl.capture.camera import CameraError, CameraSource


def run_one(cam_cfg, seconds: float, out_dir: Path) -> dict:
    src = CameraSource(cam_cfg)
    try:
        src.start()
    except CameraError as exc:
        return {"erro": str(exc)}
    try:
        t_end = time.perf_counter() + 1.0  # aquecimento (auto-exposição/negociação)
        while time.perf_counter() < t_end:
            src.read_latest(timeout=1.0)
        ts, last = [], None
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            f = src.read_latest(timeout=1.0)
            if f is not None:
                ts.append(f.ts)
                last = f
        if last is not None:
            name = f"{cam_cfg.backend}_auto{int(cam_cfg.auto_exposure)}.jpg"
            cv2.imwrite(str(out_dir / name), last.image)
        dts = [b - a for a, b in zip(ts, ts[1:], strict=False)]
        s = src.settings
        return {
            "resolucao": f"{last.image.shape[1]}x{last.image.shape[0]}" if last is not None else "?",
            "fourcc": s.actual.get("fourcc"),
            "fps_driver": round(s.actual.get("fps", 0.0), 1),
            "fps_real": round(len(dts) / sum(dts), 1) if dts else 0.0,
            "intervalo_ms_p50": round(1000 * statistics.median(dts), 1) if dts else None,
            "intervalo_ms_max": round(1000 * max(dts), 1) if dts else None,
            "exposicao": s.actual.get("exposure_now"),
            "brilho_medio": round(float(last.image.mean()), 1) if last is not None else None,
            "recusados": [k for k, ok in s.accepted.items() if not ok],
        }
    finally:
        src.stop()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--backends", default="dshow,msmf")
    parser.add_argument("--exposures", default="", help="lista de exposições manuais extras, ex.: -5,-7,-8")
    args = parser.parse_args()
    cfg = config_from_args(args)
    out_dir = Path("data/diag")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Câmera {cfg.camera.index}: pedido {cfg.camera.width}x{cfg.camera.height}@{cfg.camera.fps} "
          f"{cfg.camera.fourcc}, exposição manual={cfg.camera.exposure}\n")
    for backend in args.backends.split(","):
        for auto in (True, False):
            cam = copy.deepcopy(cfg.camera)
            cam.backend = backend
            cam.auto_exposure = auto
            res = run_one(cam, args.seconds, out_dir)
            print(f"[{backend:5s} auto_exposicao={'sim' if auto else 'não'}]")
            for k, v in res.items():
                print(f"    {k:<18} {v}")
            print()
    for exp in [float(x) for x in args.exposures.split(",") if x.strip()]:
        cam = copy.deepcopy(cfg.camera)
        cam.auto_exposure, cam.exposure = False, exp
        res = run_one(cam, min(args.seconds, 3.0), out_dir)
        print(f"[{cam.backend} exposição manual {exp:+.0f}]  fps_real={res.get('fps_real')}  "
              f"brilho_medio={res.get('brilho_medio')}  exposição lida={res.get('exposicao')}")
    print(f"\nQuadros de amostra em {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
