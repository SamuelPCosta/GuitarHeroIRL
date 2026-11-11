"""Testes da configuração (T1.2, RF-80)."""

import argparse

import pytest

from ghirl.app.config import (
    DEFAULT_CONFIG_PATH,
    Config,
    ConfigError,
    add_config_args,
    config_from_args,
    load_config,
    parse_override,
)


def test_default_yaml_matches_dataclass_defaults():
    assert load_config(DEFAULT_CONFIG_PATH).to_dict() == Config().to_dict()


def test_load_without_file_gives_defaults():
    assert load_config(None).to_dict() == Config().to_dict()


def test_yaml_file_overrides_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("camera:\n  width: 640\n  height: 480\nneck:\n  sam:\n    version: 2\n", encoding="utf-8")
    cfg = load_config(p)
    assert (cfg.camera.width, cfg.camera.height) == (640, 480)
    assert cfg.neck.sam.version == 2
    assert cfg.camera.fps == 30  # não mencionado: mantém o padrão


def test_cli_overrides_apply_after_file(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("camera:\n  width: 640\n", encoding="utf-8")
    cfg = load_config(p, ["camera.width=800", "display.mirror=false", "calib.frets=[1, 12, 5]"])
    assert cfg.camera.width == 800
    assert cfg.display.mirror is False
    assert cfg.calib.frets == [1, 12, 5]


def test_override_int_promoted_to_float():
    cfg = load_config(None, ["calib.beta=1"])
    assert cfg.calib.beta == 1.0 and isinstance(cfg.calib.beta, float)


def test_optional_field_accepts_null():
    cfg = load_config(None, ["camera.exposure=null"])
    assert cfg.camera.exposure is None


@pytest.mark.parametrize(
    "override",
    [
        "camera.widht=640",  # erro de digitação
        "cameraa.width=640",  # seção inexistente
        "neck.sam.foo=1",  # chave aninhada inexistente
    ],
)
def test_unknown_key_is_error(override):
    with pytest.raises(ConfigError, match="chave desconhecida"):
        load_config(None, [override])


@pytest.mark.parametrize(
    "override",
    [
        "camera.width=abc",
        "camera.width=12.5",
        "display.mirror=1",
        "calib.mode=4p",
        "calib.frets=5",
        "camera=3",
    ],
)
def test_invalid_value_is_error(override):
    with pytest.raises(ConfigError):
        load_config(None, [override])


def test_unknown_key_in_yaml_is_error(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("game:\n  lead: 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="game.lead"):
        load_config(p)


def test_missing_file_is_error(tmp_path):
    with pytest.raises(ConfigError, match="não encontrado"):
        load_config(tmp_path / "nao_existe.yaml")


def test_parse_override_requires_equals():
    with pytest.raises(ConfigError):
        parse_override("camera.width")
    assert parse_override("neck.sam.prompts=[a, b]") == (["neck", "sam", "prompts"], ["a", "b"])


def test_argparse_integration():
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    args = parser.parse_args(["--set", "camera.fps=15", "--set", "game.style=slide"])
    cfg = config_from_args(args)
    assert cfg.camera.fps == 15 and cfg.game.style == "slide"


def test_loaded_configs_are_independent():
    a = load_config(None, ["calib.frets=[3]"])
    b = load_config(None)
    assert b.calib.frets == [1, 12]
    assert a.calib.frets == [3]
