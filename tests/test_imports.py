"""Every module imports.

A syntax error in ui/main_window.py once passed the whole suite green, because
nothing imported it. These are the cheapest tests here and they catch the class
of mistake that is otherwise found by launching the app by hand.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import fantasia_core


def _modules(package) -> list:
    return sorted(
        m.name for m in pkgutil.walk_packages(package.__path__, package.__name__ + ".")
    )


@pytest.mark.parametrize("name", _modules(fantasia_core))
def test_core_module_imports(name):
    importlib.import_module(name)


def _ui_modules() -> list:
    """Discovered, not listed: a new UI module is covered the day it lands."""
    import ui

    return _modules(ui)


UI_MODULES = _ui_modules()


@pytest.mark.parametrize("name", UI_MODULES)
def test_ui_module_imports(name):
    pytest.importorskip("PySide6")
    importlib.import_module(name)
