"""Configuração do app (T1.2, RF-80).

Os valores padrão vivem nas dataclasses abaixo e são espelhados em ``configs/default.yaml``
(um teste garante que os dois coincidem). A configuração final é:

    padrões das dataclasses  <-  arquivo YAML  <-  sobrescritas ``--set secao.chave=valor``

Chaves desconhecidas são erro (evita erros de digitação silenciosos).
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class ConfigError(ValueError):
    """Erro de configuração (chave inexistente, tipo ou valor inválido)."""


# --------------------------------------------------------------------------- seções


@dataclass
class CameraConfig:
    index: int = 0
    backend: Literal["dshow", "msmf", "any"] = "dshow"
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "MJPG"
    auto_exposure: bool = False
    exposure: float | None = -6.0  # DSHOW: log2 de segundos (-6 ≈ 1/64 s)
    gain: float | None = None
    latency_ms: float = 70.0  # latência estimada da câmera (exposição -> leitura)


@dataclass
class DisplayConfig:
    mirror: bool = True  # espelha só a exibição; o processamento usa a imagem original
    debug: bool = False
    window_name: str = "Guitar Hero IRL"


@dataclass
class HandsConfig:
    max_hands: int = 2
    min_conf: float = 0.5
    handedness_swap: bool = False  # T1.7: nesta câmera o rótulo do MediaPipe já vem correto na imagem original
    fretting: Literal["left", "right"] = "left"
    model_path: str = "models/hand_landmarker.task"


@dataclass
class SamConfig:
    version: Literal[2, 3] = 3
    prompts: list[str] = field(default_factory=lambda: ["guitar neck", "fretboard", "guitar"])


@dataclass
class NeckConfig:
    locator: Literal["sam", "yolo", "classic"] = "sam"
    tracker: Literal["lk", "sam2video", "per_frame"] = "lk"
    reseg_period_s: float = 2.0
    iou_reset: float = 0.7
    min_inliers: int = 25
    lost_frames: int = 5
    relock_timeout_s: float = 5.0
    sam: SamConfig = field(default_factory=SamConfig)


@dataclass
class CalibConfig:
    mode: Literal["2p", "2p_taper", "3p"] = "2p"
    frets: list[int] = field(default_factory=lambda: [1, 12])
    beta: float = 0.75
    hold_s: float = 0.6  # tempo de pestana estável para capturar
    scale_mm: float = 648.0
    width_nut_mm: float = 43.0
    width_12_mm: float = 54.0
    string_margin: float = 0.09
    bass_edge: Literal["top", "bottom"] = "top"


@dataclass
class GameConfig:
    lead_s: float = 1.5
    style: Literal["ring", "slide"] = "ring"
    w_before_ms: float = 600.0
    w_after_ms: float = 200.0
    eps_fret: float = 0.15
    string_mode: Literal["zones", "exact"] = "zones"
    eps_string: float = 0.5
    debounce_frames: int = 2
    contact_gamma: float = 0.2
    user_offset_ms: float = 0.0
    count_in_beats: int = 4


@dataclass
class LogConfig:
    jsonl: bool = True
    dir: str = "logs"


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    hands: HandsConfig = field(default_factory=HandsConfig)
    neck: NeckConfig = field(default_factory=NeckConfig)
    calib: CalibConfig = field(default_factory=CalibConfig)
    game: GameConfig = field(default_factory=GameConfig)
    log: LogConfig = field(default_factory=LogConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- validação de tipos


def _check_value(value: Any, tp: Any, path: str) -> Any:
    """Valida (e converte de forma conservadora) ``value`` para o tipo anotado ``tp``."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{path}: valor {value!r} inválido; opções: {list(args)}")
        return value

    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _check_value(value, arg, path)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(f"{path}: {value!r} não corresponde a {tp}")

    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: esperado lista, recebido {type(value).__name__}")
        (item_tp,) = args
        return [_check_value(v, item_tp, f"{path}[{i}]") for i, v in enumerate(value)]

    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: esperado booleano, recebido {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: esperado inteiro, recebido {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{path}: esperado número, recebido {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: esperado texto, recebido {value!r}")
        return value
    raise ConfigError(f"{path}: tipo não suportado {tp}")


def _merge(obj: Any, data: dict[str, Any], prefix: str) -> None:
    """Aplica ``data`` sobre a dataclass ``obj`` recursivamente, validando chaves e tipos."""
    if not isinstance(data, dict):
        raise ConfigError(f"{prefix or 'raiz'}: esperado mapeamento, recebido {type(data).__name__}")
    hints = typing.get_type_hints(type(obj))
    fields = {f.name for f in dataclasses.fields(obj)}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in fields:
            raise ConfigError(f"chave desconhecida: {path!r} (válidas: {sorted(fields)})")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            _merge(current, value, path)
        else:
            setattr(obj, key, _check_value(value, hints[key], path))


# --------------------------------------------------------------------------- API pública


def parse_override(text: str) -> tuple[list[str], Any]:
    """``"camera.width=640"`` -> (["camera", "width"], 640). O valor é lido como YAML."""
    if "=" not in text:
        raise ConfigError(f"sobrescrita sem '=': {text!r} (formato: secao.chave=valor)")
    key, raw = text.split("=", 1)
    parts = [p for p in key.strip().split(".") if p]
    if not parts:
        raise ConfigError(f"sobrescrita sem chave: {text!r}")
    value = yaml.safe_load(raw) if raw.strip() else ""
    return parts, value


def _nest(parts: list[str], value: Any) -> dict[str, Any]:
    out: Any = value
    for part in reversed(parts):
        out = {part: out}
    return out


def load_config(path: str | Path | None = DEFAULT_CONFIG_PATH, overrides: list[str] | None = None) -> Config:
    """Carrega a configuração: padrões <- YAML (se ``path``) <- sobrescritas."""
    cfg = Config()
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"arquivo de configuração não encontrado: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        _merge(cfg, data, "")
    for text in overrides or []:
        parts, value = parse_override(text)
        _merge(cfg, _nest(parts, value), "")
    return cfg


def add_config_args(parser) -> None:
    """Adiciona ``--config`` e ``--set`` a um ``argparse.ArgumentParser``."""
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="arquivo YAML de configuração")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="CHAVE=VALOR",
        help="sobrescreve uma chave, ex.: --set camera.width=640 (pode repetir)",
    )


def config_from_args(args) -> Config:
    return load_config(args.config, args.overrides)
