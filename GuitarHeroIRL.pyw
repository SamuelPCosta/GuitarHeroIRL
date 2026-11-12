"""Abre o Guitar Hero IRL com dois cliques (sem janela de console).

Usa o Python do .venv do projeto; erros vão para logs/app.log e aparecem numa caixa de mensagem.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv" / "Scripts" / "pythonw.exe"

# aberto pelo Python do sistema (associação do .pyw): reabre com o Python do .venv
if VENV.exists() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
    subprocess.Popen([str(VENV), str(Path(__file__).resolve())], cwd=ROOT)
    sys.exit(0)

import os  # noqa: E402

os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
(ROOT / "logs").mkdir(exist_ok=True)
sys.stdout = sys.stderr = open(ROOT / "logs" / "app.log", "w", encoding="utf-8", buffering=1)  # noqa: SIM115

try:
    from ghirl.app import main

    main([])
except Exception:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    from PySide6.QtWidgets import QApplication, QMessageBox

    app = QApplication.instance() or QApplication([])
    QMessageBox.critical(None, "Guitar Hero IRL", traceback.format_exc()[-1500:] + "\n\n(log em logs/app.log)")
