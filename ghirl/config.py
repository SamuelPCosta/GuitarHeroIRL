"""Configuração: padrões nas dataclasses <- ``config.yaml`` <- ``--set secao.chave=valor``.

Chaves desconhecidas e tipos errados são erro (evita erros de digitação silenciosos).
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class ConfigError(ValueError):
    pass


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


@dataclass
class DisplayConfig:
    mirror: bool = True  # espelha só a exibição
    debug: bool = False  # pontos rastreados e esqueleto das mãos
    show_frets: bool = False  # trastes e todas as notas do exercício (baixa opacidade)
    smooth: bool = True  # filtro One Euro na pose do braço (nota não treme com o violão parado)
    smooth_min_cutoff: float = 1.0  # Hz: menor = mais suave parado
    smooth_beta: float = 0.1  # maior = menos atraso (medido: tremedeira 1,74 -> 0,50 px, atraso 2,7 px)


@dataclass
class HandsConfig:
    max_hands: int = 2
    min_conf: float = 0.5
    model_path: str = "models/hand_landmarker.task"


@dataclass
class GestureConfig:
    enabled: bool = True
    advance_max_s: float = 1.5  # fechar e abrir a mão em até X s = avançar
    reset_hold_s: float = 3.0  # mão fechada por X s = recomeçar a calibração


@dataclass
class NeckConfig:
    lost_frames: int = 5
    rekey_after_s: float = 2.0  # violão visível sem casar com o quadro-chave: troca o K


@dataclass
class CalibConfig:
    frets: list[int] = field(default_factory=lambda: [1, 12])
    beta: float = 0.75  # posição do dedo na casa (0 = traste anterior, 1 = traste da casa)
    refine_frets: bool = True  # ajusta a escala pelos trastes visíveis na imagem
    string_margin: float = 0.09
    bass_edge: Literal["top", "bottom"] = "top"


@dataclass
class GameConfig:
    exercises_dir: str = "exercises"
    exercise: str = "pentatonica_am_casa5"
    hold_frames: int = 2  # quadros com o dedo na nota
    eps_fret: float = 0.25  # folga ao longo do braço (casas)
    eps_string: float = 0.5  # folga através do braço (cordas)
    require_pluck: bool = True  # a nota só vale com um dedo da outra mão tocando a corda
    pluck_window_s: float = 0.35  # dedo na nota e corda tocada a menos de X s
    pluck_min_fret: float = 13.0  # mão que dedilha: além desta "casa" (sobre o corpo)
    pluck_direction: Literal["down", "any"] = "down"  # down = em direção ao chão (corda 6 -> 1)


@dataclass
class LogConfig:
    jsonl: bool = False
    dir: str = "logs"


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    hands: HandsConfig = field(default_factory=HandsConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    neck: NeckConfig = field(default_factory=NeckConfig)
    calib: CalibConfig = field(default_factory=CalibConfig)
    game: GameConfig = field(default_factory=GameConfig)
    log: LogConfig = field(default_factory=LogConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _check(value: Any, tp: Any, path: str) -> Any:
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{path}: {value!r} inválido; opções: {list(args)}")
        return value
    if args and type(None) in args:  # X | None
        return None if value is None else _check(value, next(a for a in args if a is not type(None)), path)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: esperado lista")
        return [_check(v, args[0], path) for v in value]
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, tp) or (tp is int and isinstance(value, bool)):
        raise ConfigError(f"{path}: esperado {tp.__name__}, recebido {value!r}")
    return value


def _merge(obj: Any, data: dict, prefix: str = "") -> None:
    if not isinstance(data, dict):
        raise ConfigError(f"{prefix or 'raiz'}: esperado mapeamento")
    hints = typing.get_type_hints(type(obj))
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in hints:
            raise ConfigError(f"chave desconhecida: {path!r}")
        if dataclasses.is_dataclass(getattr(obj, key)):
            _merge(getattr(obj, key), value, path)
        else:
            setattr(obj, key, _check(value, hints[key], path))


def load_config(path: str | Path | None = DEFAULT_CONFIG_PATH, overrides: list[str] | None = None) -> Config:
    cfg = Config()
    if path is not None:
        if not Path(path).exists():
            raise ConfigError(f"arquivo de configuração não encontrado: {path}")
        _merge(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    for text in overrides or []:
        if "=" not in text:
            raise ConfigError(f"sobrescrita sem '=': {text!r} (formato: secao.chave=valor)")
        key, raw = text.split("=", 1)
        nested: Any = yaml.safe_load(raw) if raw.strip() else ""
        for part in reversed(key.strip().split(".")):
            nested = {part: nested}
        _merge(cfg, nested)
    return cfg


def add_config_args(parser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="arquivo YAML de configuração")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="CHAVE=VALOR",
                        help="sobrescreve uma chave, ex.: --set camera.index=1 (pode repetir)")


def config_from_args(args) -> Config:
    return load_config(args.config, args.overrides)
