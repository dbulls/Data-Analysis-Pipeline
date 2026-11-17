"""Shared fixtures: the synthetic dataset and one full pipeline run on it."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoanalysis import loading                  # noqa: E402
from autoanalysis.config import Config            # noqa: E402
from autoanalysis.pipeline import run_pipeline    # noqa: E402
from make_synthetic_data import write_tables      # noqa: E402


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    return write_tables(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("output")


@pytest.fixture(scope="session")
def run(data_dir, output_dir):
    """The whole pipeline, run once and shared by every end-to-end test."""
    hashes_before = loading.file_hashes(data_dir)
    result = run_pipeline(data_dir, output_dir, Config())
    result["hashes_before"] = hashes_before
    return result
