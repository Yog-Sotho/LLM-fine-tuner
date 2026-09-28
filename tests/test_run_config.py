"""Unit tests (no model downloads) for run configs, fingerprints and run folders."""

import pytest
import yaml
from datasets import Dataset

import core.run_config as rc
from config.constants import RUN_CONFIG_FILENAME, RUN_NAME_PATTERN


def test_fingerprint_is_deterministic_and_content_sensitive():
    a = Dataset.from_dict({"text": ["x", "y"]})
    b = Dataset.from_dict({"text": ["x", "y"]})
    changed = Dataset.from_dict({"text": ["x", "z"]})
    reordered = Dataset.from_dict({"text": ["y", "x"]})
    fp = rc.dataset_fingerprint(a)
    assert fp == rc.dataset_fingerprint(b)
    assert fp["rows"] == 2 and fp["columns"] == ["text"]
    assert fp["sha256"] != rc.dataset_fingerprint(changed)["sha256"]
    assert fp["sha256"] != rc.dataset_fingerprint(reordered)["sha256"]


def test_fingerprint_ignores_column_order():
    a = Dataset.from_dict({"prompt": ["p"], "completion": ["c"]})
    b = Dataset.from_dict({"completion": ["c"], "prompt": ["p"]})
    assert rc.dataset_fingerprint(a) == rc.dataset_fingerprint(b)


def test_save_and_load_round_trip(tmp_path):
    ds = Dataset.from_dict({"text": ["hello"]})
    path = rc.save_run_config(
        str(tmp_path), mode="sft", model="m", dataset=ds, seed=7, hyperparams={"epochs": 1}
    )
    assert path.endswith(RUN_CONFIG_FILENAME)
    record = rc.load_run_config(path)
    assert record["mode"] == "sft" and record["model"] == "m" and record["seed"] == 7
    assert record["hyperparams"] == {"epochs": 1}
    assert record["dataset"] == rc.dataset_fingerprint(ds)
    assert "transformers" in record["libraries"]


def test_load_rejects_non_config(tmp_path):
    path = tmp_path / "other.yaml"
    path.write_text("just: data\n")
    with pytest.raises(ValueError, match="not a run config"):
        rc.load_run_config(str(path))


def test_load_refuses_python_objects(tmp_path):
    path = tmp_path / "evil.yaml"
    path.write_text('mode: sft\nmodel: !!python/object/apply:os.system ["echo pwned"]\n')
    with pytest.raises(yaml.YAMLError):
        rc.load_run_config(str(path))


def test_resolve_report_to():
    assert rc.resolve_report_to(None) == "none"
    assert rc.resolve_report_to(" NONE ") == "none"
    with pytest.raises(ValueError, match="not available"):
        rc.resolve_report_to("definitely-not-a-tracker")


@pytest.mark.parametrize(
    "name", ["../escape", "a/b", "a\\b", "", ".hidden", "-dash", "x" * 65, "sp ace", "a\0b"]
)
def test_run_dir_for_rejects_unsafe_names(name):
    with pytest.raises(ValueError, match="Run name"):
        rc.run_dir_for(name)


def test_run_dir_for_stays_inside_runs_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(rc, "RUNS_DIR", str(tmp_path))
    assert rc.run_dir_for("my-run_1.0") == str(tmp_path / "my-run_1.0")


def test_new_run_name_is_valid():
    import re

    assert re.fullmatch(RUN_NAME_PATTERN, rc.new_run_name("sft"))
