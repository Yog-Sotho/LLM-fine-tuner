"""Logging instead of print, and the single version number."""

import ast
import logging
import pathlib

import pytest
from typer.testing import CliRunner

from config.constants import APP_NAME, APP_VERSION

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_PACKAGES = ("cli", "config", "core", "data", "export", "inference", "training", "ui")


# ── Version ────────────────────────────────────────────────────────────────


def test_pyproject_reads_the_version_from_constants():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in text
    assert 'version = { attr = "config.constants.APP_VERSION" }' in text
    # setuptools reads it without importing the module only when it is a literal.
    tree = ast.parse((ROOT / "config" / "constants.py").read_text(encoding="utf-8"))
    (node,) = [n for n in tree.body if isinstance(n, ast.Assign)
               and getattr(n.targets[0], "id", None) == "APP_VERSION"]  # fmt: skip
    assert ast.literal_eval(node.value) == APP_VERSION
    assert APP_VERSION in APP_NAME


def test_cli_version_flag():
    from cli.commands import app

    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0 and result.output.strip() == APP_VERSION


def test_main_version_prints_only_the_version(monkeypatch, capsys):
    import main as entry

    monkeypatch.setattr(entry.sys, "argv", ["main.py", "--version"])
    with pytest.raises(SystemExit) as exc:
        entry.main()
    assert exc.value.code == 0 and capsys.readouterr().out.strip() == APP_VERSION


def test_run_record_names_the_app_version():
    from core.run_config import library_versions

    assert library_versions()["llm-fine-tuner"] == APP_VERSION


# ── Logging ────────────────────────────────────────────────────────────────


def test_application_modules_do_not_print():
    """Library code logs; only main.py (the console entry point) prints."""
    offenders = []
    for package in APP_PACKAGES:
        for path in (ROOT / package).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print":
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, offenders


@pytest.fixture
def restore_app_log_levels():
    levels = {name: logging.getLogger(name).level for name in APP_PACKAGES}
    yield
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)


def test_configure_logging_sets_the_app_level(restore_app_log_levels, caplog):
    import main as entry

    entry._configure_logging("DEBUG")
    assert all(logging.getLogger(n).level == logging.DEBUG for n in APP_PACKAGES)
    entry._configure_logging("LOUD")  # not a level: warns and uses INFO
    assert logging.getLogger("training").level == logging.INFO
    assert "LFT_LOG_LEVEL='LOUD' is not a level" in caplog.text


def test_training_warnings_are_logged_and_kept_in_the_run_log(caplog):
    from training.sft import _warn

    class Log:
        records: list = []

    with caplog.at_level(logging.WARNING, logger="training.sft"):
        _warn(Log, "Packing skipped.")
    assert "Packing skipped." in caplog.text
    assert Log.records[-1]["note"] == "⚠️ Packing skipped."


def test_ignored_column_mapping_is_logged(tmp_path, caplog):
    from cli.commands import DummyFile
    from data.loader import load_dataset_from_file

    path = tmp_path / "d.csv"
    path.write_text("instruction,output\nhi there,hello back\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="data.loader"):
        load_dataset_from_file(DummyFile(str(path)), "csv", column_mapping={"missing": "text"})
    assert "source columns not found, ignored: ['missing']" in caplog.text
