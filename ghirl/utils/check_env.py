"""Diagnóstico do ambiente (T1.1).

Uso:  python -m ghirl.utils.check_env

Imprime as versões das dependências principais e verifica se o PyTorch enxerga a GPU.
Sai com código 1 se algo obrigatório faltar.
"""

from __future__ import annotations

import importlib
import platform
import sys

REQUIRED = ["numpy", "cv2", "mediapipe", "yaml", "torch"]


def _version(mod) -> str:
    return getattr(mod, "__version__", "?")


def main() -> int:
    ok = True
    print(f"Python      {platform.python_version()}  ({sys.executable})")
    print(f"Sistema     {platform.system()} {platform.release()}")
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            print(f"{name:<11} {_version(mod)}")
        except Exception as exc:  # noqa: BLE001 - diagnóstico: reporta qualquer falha de import
            print(f"{name:<11} FALTANDO ({exc})")
            ok = False

    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"CUDA        disponível={cuda}  build={torch.version.cuda}")
        if cuda:
            props = torch.cuda.get_device_properties(0)
            print(f"GPU         {props.name}  {props.total_memory / 2**30:.1f} GiB")
        else:
            ok = False
    except ImportError:
        pass

    print("OK" if ok else "FALHOU")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
