"""One analysis per dataset, each in its own output folder."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from autoanalysis.loading import discover_datasets, load_raw_tables

PROJECT = Path(__file__).resolve().parents[1]


def small_csv(path, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    pd.DataFrame({"group": rng.choice(["a", "b"], 60), "value": rng.normal(size=60)}).to_csv(path, index=False)


def test_each_csv_is_a_dataset_and_each_subfolder_is_one_dataset(tmp_path):
    small_csv(tmp_path / "sales.csv")
    small_csv(tmp_path / "weather.csv")
    (tmp_path / "shop").mkdir()
    small_csv(tmp_path / "shop" / "orders.csv")
    small_csv(tmp_path / "shop" / "customers.csv")
    (tmp_path / "empty_folder").mkdir()
    (tmp_path / ".hidden").mkdir()
    small_csv(tmp_path / ".hidden" / "x.csv")

    found = discover_datasets(tmp_path)
    assert [name for name, _ in found] == ["sales", "shop", "weather"]
    assert set(load_raw_tables(dict(found)["shop"])) == {"orders", "customers"}
    assert set(load_raw_tables(dict(found)["sales"])) == {"sales"}


def test_name_clash_between_file_and_folder_is_an_error(tmp_path):
    small_csv(tmp_path / "shop.csv")
    (tmp_path / "shop").mkdir()
    small_csv(tmp_path / "shop" / "orders.csv")
    with pytest.raises(ValueError, match="shop"):
        discover_datasets(tmp_path)


def test_command_line_writes_one_folder_per_dataset_and_an_index(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    small_csv(data / "first.csv", seed=1)
    small_csv(data / "second.csv", seed=2)
    command = [sys.executable, str(PROJECT / "run_pipeline.py"), "--data", str(data), "--output", str(out)]

    subprocess.run(command, check=True, capture_output=True, cwd=PROJECT)
    assert (out / "first" / "06_report.html").exists()
    assert (out / "second" / "06_report.html").exists()
    index = (out / "index.html").read_text()
    assert 'href="first/06_report.html"' in index and 'href="second/06_report.html"' in index

    # --only re-runs one dataset and leaves the other's results alone.
    before = (out / "second" / "04_rejected.json").stat().st_mtime_ns
    subprocess.run(command + ["--only", "first"], check=True, capture_output=True, cwd=PROJECT)
    assert (out / "second" / "04_rejected.json").stat().st_mtime_ns == before
    assert 'href="second/06_report.html"' in (out / "index.html").read_text()
