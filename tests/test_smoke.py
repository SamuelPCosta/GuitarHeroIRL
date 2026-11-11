"""Testes de fumaça do ambiente (T1.1)."""

import importlib

import pytest


@pytest.mark.parametrize("name", ["numpy", "cv2", "mediapipe", "yaml"])
def test_dependency_imports(name):
    importlib.import_module(name)


def test_package_imports():
    import ghirl

    assert ghirl.__version__


def test_single_opencv_build():
    """mediapipe traz opencv-contrib-python; opencv-python junto quebraria o cv2."""
    from importlib.metadata import PackageNotFoundError, version

    with pytest.raises(PackageNotFoundError):
        version("opencv-python")
