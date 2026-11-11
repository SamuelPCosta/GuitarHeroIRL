"""Cronômetro por estágio do pipeline (T1.6, RF-70, RNF-01).

Uso:
    prof = Profiler()
    prof.begin_frame()
    with prof.stage("maos"):
        ...
    times = prof.end_frame()        # {"maos": 12.3, ..., "total": 20.1}  (ms)
    print(prof.report())            # tabela P50/P95/média por estágio
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np


class Profiler:
    def __init__(self, keep_last: int = 10_000):
        self.keep_last = keep_last
        self._history: dict[str, list[float]] = defaultdict(list)
        self._order: list[str] = []
        self._current: dict[str, float] = {}
        self._t_frame: float | None = None
        self.frames = 0

    def begin_frame(self) -> None:
        self._current = {}
        self._t_frame = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._current[name] = self._current.get(name, 0.0) + dt_ms
            if name not in self._order:
                self._order.append(name)

    def end_frame(self) -> dict[str, float]:
        if self._t_frame is None:
            raise RuntimeError("end_frame() sem begin_frame()")
        total = (time.perf_counter() - self._t_frame) * 1000.0
        times = dict(self._current)
        times["total"] = total
        for k, v in times.items():
            h = self._history[k]
            h.append(v)
            if len(h) > self.keep_last:
                del h[: len(h) - self.keep_last]
        self.frames += 1
        self._t_frame = None
        return times

    @property
    def last(self) -> dict[str, float]:
        return {k: v[-1] for k, v in self._history.items() if v}

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for name in [*self._order, "total"]:
            vals = self._history.get(name)
            if not vals:
                continue
            a = np.asarray(vals)
            out[name] = {"n": len(a), "media": float(a.mean()), "p50": float(np.percentile(a, 50)),
                         "p95": float(np.percentile(a, 95)), "max": float(a.max())}
        return out

    def report(self) -> str:
        s = self.summary()
        if not s:
            return "(sem medições)"
        lines = [f"{'estágio':<14}{'média':>9}{'P50':>9}{'P95':>9}{'máx':>9}   (ms, {self.frames} quadros)"]
        for name, st in s.items():
            lines.append(f"{name:<14}{st['media']:>9.2f}{st['p50']:>9.2f}{st['p95']:>9.2f}{st['max']:>9.2f}")
        if "total" in s and s["total"]["media"] > 0:
            lines.append(f"{'FPS (1/média)':<14}{1000.0 / s['total']['media']:>9.1f}")
        return "\n".join(lines)
