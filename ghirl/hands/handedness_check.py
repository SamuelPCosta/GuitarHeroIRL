"""Experimento de lateralidade do MediaPipe (T1.7, risco K5).

Mostra instruções na tela e, em três fases, registra o rótulo bruto do MediaPipe para a imagem
ORIGINAL (como o pipeline processa) e para a imagem ESPELHADA:

    1. só a mão ESQUERDA levantada
    2. só a mão DIREITA levantada
    3. segurando o violão normalmente (10 s) — mede o acerto do papel "notas"/"ritmo"

No fim imprime a tabela, recomenda o valor de ``hands.handedness_swap`` e salva
``data/diag/handedness.json``.

Uso:
    python -m ghirl.hands.handedness_check
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path

import cv2

from ghirl.app.config import add_config_args, config_from_args
from ghirl.capture.base import Frame
from ghirl.capture.camera import CameraSource
from ghirl.hands.mp_tracker import WRIST, HandTracker
from ghirl.render.debug import draw_hands
from ghirl.render.hud import draw_panel

PHASES = [
    ("esquerda", "Levante SO a mao ESQUERDA (a que faz as notas)", 5.0),
    ("direita", "Levante SO a mao DIREITA (a do ritmo)", 5.0),
    ("violao", "Segure o VIOLAO normalmente, mao esquerda no braco", 10.0),
]
COUNTDOWN_S = 3.0
WIN = "Experimento de lateralidade"


def _big_text(img, text, y, scale=1.0, color=(255, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = max(10, (img.shape[1] - tw) // 2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--fases", default="esquerda,direita,violao",
                        help="fases a rodar, ex.: --fases violao")
    args = parser.parse_args()
    cfg = config_from_args(args)
    wanted = [p.strip() for p in args.fases.split(",") if p.strip()]
    phases = [p for p in PHASES if p[0] in wanted]

    cfg_raw = copy.deepcopy(cfg.hands)
    cfg_raw.handedness_swap = False  # queremos o rótulo bruto
    trk_orig = HandTracker(cfg_raw)
    trk_flip = HandTracker(cfg_raw)
    trk_app = HandTracker(cfg.hands)  # com a correção configurada, para a fase 3

    results: dict = {}
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    aborted = False
    with CameraSource(cfg.camera) as cam:
        for key, instruction, duration in phases:
            raw_orig: Counter = Counter()
            raw_flip: Counter = Counter()
            n_hands: Counter = Counter()
            roles_seen: Counter = Counter()
            role_ok = role_total = 0
            t_start = time.perf_counter()
            while True:
                frame = cam.read_latest(timeout=1.0)
                if frame is None:
                    continue
                elapsed = time.perf_counter() - t_start
                collecting = elapsed >= COUNTDOWN_S
                if elapsed >= COUNTDOWN_S + duration:
                    break
                hands = trk_orig.detect(frame)
                flipped = Frame(cv2.flip(frame.image, 1), frame.ts, frame.index)
                hands_f = trk_flip.detect(flipped)
                hands_app = trk_app.detect(frame)
                if collecting:
                    if key in ("esquerda", "direita"):
                        if len(hands) == 1:
                            raw_orig[hands[0].handedness_raw] += 1
                        if len(hands_f) == 1:
                            raw_flip[hands_f[0].handedness_raw] += 1
                    else:
                        n_hands[len(hands_app)] += 1
                        for h in hands_app:
                            roles_seen[h.role] += 1
                    if key == "violao" and len(hands_app) == 2:
                        # destro de frente p/ câmera, imagem original: a mão esquerda (braço do violão)
                        # aparece à DIREITA na imagem (x maior) -> deve ser a de "notas"
                        right_in_img = max(hands_app, key=lambda h: h.landmarks_px[WRIST, 0])
                        role_total += 1
                        role_ok += int(right_in_img.role == "notas"
                                       and min(hands_app, key=lambda h: h.landmarks_px[WRIST, 0]).role == "ritmo")

                view = frame.image.copy()
                draw_hands(view, hands_app)
                view = cv2.flip(view, 1)
                _big_text(view, instruction, 60, 1.0)
                if collecting:
                    left = COUNTDOWN_S + duration - elapsed
                    _big_text(view, f"gravando... {left:4.1f} s", 110, 0.9, (80, 80, 255))
                else:
                    _big_text(view, f"comeca em {COUNTDOWN_S - elapsed:3.1f} s", 110, 0.9, (80, 255, 80))
                draw_panel(view, [f"maos detectadas: {len(hands)}", "ESC cancela"], (10, view.shape[0] - 70))
                cv2.imshow(WIN, view)
                if cv2.waitKey(1) & 0xFF == 27:
                    aborted = True
                    break
            results[key] = {"original": dict(raw_orig), "espelhada": dict(raw_flip),
                            "maos_por_quadro": dict(n_hands), "papeis_vistos": dict(roles_seen),
                            "papel_ok": role_ok, "papel_total": role_total}
            if aborted:
                break
    cv2.destroyAllWindows()
    for t in (trk_orig, trk_flip, trk_app):
        t.close()
    if aborted:
        print("Cancelado.")
        return 1

    def majority(c: dict) -> str:
        return max(c, key=c.get) if c else "?"

    swap = None
    if "esquerda" in results and "direita" in results:
        print("\nRótulo bruto do MediaPipe (quadros com exatamente 1 mão):")
        print(f"{'mão real':<10}{'imagem original':<28}{'imagem espelhada':<28}")
        for key in ("esquerda", "direita"):
            r = results[key]
            print(f"{key:<10}{str(r['original']):<28}{str(r['espelhada']):<28}")
        left_orig = majority(results["esquerda"]["original"])
        right_orig = majority(results["direita"]["original"])
        if left_orig.lower().startswith("r") and right_orig.lower().startswith("l"):
            swap = True
        elif left_orig.lower().startswith("l") and right_orig.lower().startswith("r"):
            swap = False
        print(f"\nRecomendação: hands.handedness_swap = {swap}  (None = inconclusivo)")
    acc = float("nan")
    if "violao" in results:
        r3 = results["violao"]
        acc = r3["papel_ok"] / r3["papel_total"] if r3["papel_total"] else float("nan")
        total = sum(r3["maos_por_quadro"].values())
        dist = ", ".join(f"{k} mão(s): {v}" for k, v in sorted(r3["maos_por_quadro"].items()))
        print(f"\nCom o violão ({total} quadros): {dist}")
        print(f"  papéis vistos: {r3['papeis_vistos']}")
        print(f"  papel correto em {r3['papel_ok']}/{r3['papel_total']} quadros com 2 mãos "
              f"({100 * acc:.1f}%) usando handedness_swap={cfg.hands.handedness_swap}")
    out = Path("data/diag/handedness.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fases": results, "swap_recomendado": swap,
                               "swap_usado_fase3": cfg.hands.handedness_swap,
                               "acerto_papel_fase3": acc}, indent=2), encoding="utf-8")
    print(f"Salvo em {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
