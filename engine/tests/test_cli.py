import importlib

import pytest
from typer.testing import CliRunner

from motionly_engine import __version__
from motionly_engine.cli import app


def test_version():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


@pytest.mark.parametrize("name", ["generate", "measure", "fit", "benchmark", "spec", "llm"])
def test_subpackages_import(name):
    importlib.import_module(f"motionly_engine.{name}")
