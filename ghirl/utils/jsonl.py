"""Log JSONL por quadro (T1.6, RF-70): uma linha JSON por quadro processado."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np


def _default(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")
        self.lines = 0

    @classmethod
    def in_dir(cls, directory: str | Path, prefix: str = "run") -> JsonlLogger:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls(Path(directory) / f"{prefix}_{stamp}.jsonl")

    def write(self, record: dict[str, Any]) -> None:
        self._f.write(json.dumps(record, default=_default, ensure_ascii=False) + "\n")
        self.lines += 1

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> JsonlLogger:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
