"""Persistent series (prices, rates): analysed as changes, so drifting levels do not fake findings."""

import numpy as np
import pandas as pd

from autoanalysis import measures, search, timeseries
from autoanalysis.config import Config
from autoanalysis.pipeline import run_pipeline


def daily_rates(seed=0, n=2000):
    """Three random walks, one value per weekday. a and b share daily shocks; c is independent."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    shared = rng.normal(0, 0.01, n)
    a = 1.2 * np.exp(np.cumsum(shared + rng.normal(0, 0.005, n)))
    b = 0.9 * np.exp(np.cumsum(shared + rng.normal(0, 0.005, n)))
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "a_rate": a, "b_rate": b, "c_rate": c})


def test_random_walks_are_detected_and_independent_iid_data_is_not():
    rng = np.random.default_rng(1)
    df = daily_rates()
    df["date"] = pd.to_datetime(df["date"])
    df["iid"] = rng.normal(10, 1, len(df))
    assert timeseries.persistence(df, "date", "a_rate", "M") > 0.5
    assert abs(timeseries.persistence(df, "date", "iid", "M")) < 0.2


def test_event_streams_are_never_treated_as_single_series(run):
    """The orders table has many orders per day, so nothing in it is converted to changes."""
    assert all(not c["table"].endswith(timeseries.CHANGE_TABLE_SUFFIX) for c in run["candidates"])


def test_drifting_levels_do_not_produce_findings_but_shared_moves_do(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    daily_rates().to_csv(data_dir / "rates.csv", index=False)
    run = run_pipeline(data_dir, tmp_path / "out", Config())

    # Levels are profiled but never tested directly.
    assert all(c["table"] == "rates" + timeseries.CHANGE_TABLE_SUFFIX
               for c in run["candidates"] if "rate" in str(c["columns"]))

    correlations = {(c["columns"]["x"][0], c["columns"]["y"][0]): c
                    for c in run["candidates"] if c["kind"] == "correlation"}
    assert correlations[("a", "b")]["status"] == "accepted"      # shared daily shocks: real
    assert correlations[("a", "c")]["status"] == "rejected"      # unrelated walks
    assert correlations[("b", "c")]["status"] == "rejected"

    # A single series has no meaningful row counts or totals.
    assert not [c for c in run["candidates"] if c["kind"] in ("trend_count", "trend_total", "concentration")]


def test_mostly_constant_column_is_not_correlated():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"x": rng.normal(size=200), "y": np.where(rng.random(200) < 0.9, 0.0, rng.normal(size=200))})
    candidate = search.make_candidate("correlation", "t", x="x", y="y")
    result = measures.measure(candidate, df, Config())
    assert not result["ok"] and "same value" in result["reason"]
