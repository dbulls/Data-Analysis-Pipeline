"""The statistical filters, tested on tiny datasets where the right answer is known."""

import numpy as np
import pandas as pd
import pytest

from autoanalysis import filters, measures, search
from autoanalysis.config import Config

CONFIG = Config()
TYPES = {"when": "datetime", "value": "numeric", "other": "numeric", "group": "categorical"}


def evaluate_one(kind, df, **columns):
    """Measure one candidate and run it through every filter. Returns the annotated candidate."""
    candidate = search.make_candidate(kind, "t", **columns)
    candidate["result"] = measures.measure(candidate, df, CONFIG)
    assert candidate["result"]["ok"], candidate["result"]
    filters.evaluate([candidate], {"t": df}, {"t": TYPES}, {"t": set()}, CONFIG)
    return candidate


def rejection_codes(candidate):
    return {r["code"] for r in candidate["rejections"]}


def daily_frame(n_days=720, per_day=3, seed=0):
    rng = np.random.default_rng(seed)
    when = np.repeat(pd.date_range("2022-01-01", periods=n_days, freq="D"), per_day)
    return pd.DataFrame({"when": when, "value": rng.normal(100, 10, len(when)),
                         "other": rng.normal(0, 1, len(when)),
                         "group": rng.choice(["a", "b", "c"], len(when))}), rng


# ---- Multiple comparisons ----------------------------------------------------

def test_benjamini_hochberg_matches_hand_calculation():
    q = filters.benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    # sorted p: .01 .03 .04 .20 -> p*m/rank: .04 .06 .0533 .20 -> monotone: .04 .0533 .0533 .20
    assert q == pytest.approx([0.04, 0.16 / 3, 0.16 / 3, 0.20])


def test_correction_removes_chance_findings_among_pure_noise():
    """200 unrelated pairs: some raw p-values dip under 0.05, none survives correction."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(300, 201)), columns=[f"c{i}" for i in range(201)])
    p_values = []
    for i in range(1, 201):
        candidate = search.make_candidate("correlation", "t", x="c0", y=f"c{i}")
        p_values.append(measures.measure(candidate, df, CONFIG)["p_value"])
    assert sum(p < 0.05 for p in p_values) >= 3                       # chance "findings" exist
    assert (filters.benjamini_hochberg(p_values) <= CONFIG.alpha).sum() == 0


# ---- Real patterns are accepted ---------------------------------------------

def test_real_trend_is_accepted_with_effect_size():
    df, _ = daily_frame()
    df["value"] += np.linspace(0, 30, len(df))
    c = evaluate_one("trend_mean", df, time="when", value="value", freq="M")
    assert c["status"] == "accepted"
    assert c["result"]["direction"] == 1 and c["result"]["effect_size"] > 0.6
    assert all(check["status"] == "pass" for check in c["checks"].values())
    assert c["confidence"] in ("high", "medium")


def test_real_group_difference_is_accepted():
    df, _ = daily_frame()
    df.loc[df["group"] == "c", "value"] += 12
    c = evaluate_one("group_diff", df, group="group", value="value")
    assert c["status"] == "accepted"
    assert c["result"]["summary"]["highest"] == "c"


def test_real_correlation_is_accepted():
    df, rng = daily_frame()
    df["other"] = df["value"] * 0.1 + rng.normal(0, 1, len(df))
    c = evaluate_one("correlation", df, x="value", y="other")
    assert c["status"] == "accepted" and c["result"]["direction"] == 1


def test_real_breakpoint_is_found_at_the_right_time():
    df, _ = daily_frame()
    df.loc[df["when"] >= "2023-03-01", "value"] -= 15
    c = evaluate_one("breakpoint", df, time="when", value="value", freq="M")
    assert c["status"] == "accepted"
    assert c["result"]["summary"]["break_period"] == "2023-03"
    assert c["result"]["direction"] == -1


# ---- False patterns are rejected, for the right reason ----------------------

def test_noise_trend_is_rejected():
    df, _ = daily_frame()
    c = evaluate_one("trend_mean", df, time="when", value="value", freq="M")
    assert c["status"] == "rejected"
    assert "not_significant" in rejection_codes(c)


def test_tiny_but_significant_effect_is_rejected_as_too_small():
    rng = np.random.default_rng(3)
    n = 60000
    df = pd.DataFrame({"value": rng.normal(size=n)})
    df["other"] = df["value"] * 0.05 + rng.normal(size=n)
    df["when"] = pd.Timestamp("2024-01-01")
    c = evaluate_one("correlation", df, x="value", y="other")
    assert c["result"]["p_value"] < 1e-6             # overwhelmingly "significant"
    assert rejection_codes(c) == {"effect_too_small"}


def test_spike_from_a_single_record_is_rejected():
    df, _ = daily_frame()
    spike_row = df.index[df["when"] == "2022-09-10"][0]
    df.loc[spike_row, "value"] = 50000
    periods = measures.find_anomalies(df, {"time": "when", "value": "value", "freq": "M"}, CONFIG, "mean")
    assert periods[0] == "2022-09"
    c = evaluate_one("anomaly_mean", df, time="when", value="value", freq="M", period="2022-09")
    assert c["status"] == "rejected"
    assert "driven_by_single_record" in rejection_codes(c)


def test_unusual_period_made_of_many_records_is_accepted():
    df, _ = daily_frame()
    df.loc[df["when"].dt.strftime("%Y-%m") == "2022-09", "value"] += 25
    c = evaluate_one("anomaly_mean", df, time="when", value="value", freq="M", period="2022-09")
    assert c["status"] == "accepted"


def test_pattern_present_in_only_one_half_is_rejected():
    """A correlation that exists only in the second year does not replicate across halves."""
    df, rng = daily_frame()
    second_year = df["when"] >= "2022-12-27"
    df.loc[second_year, "other"] = df.loc[second_year, "value"] * 0.2 + rng.normal(0, 1, second_year.sum())
    c = evaluate_one("correlation", df, x="value", y="other")
    assert c["q_value"] < 0.05
    assert "does_not_replicate" in rejection_codes(c)


def test_gradual_trend_is_not_reported_as_a_step():
    df, _ = daily_frame()
    df["value"] += np.linspace(0, 30, len(df))
    c = evaluate_one("breakpoint", df, time="when", value="value", freq="M")
    assert "better_explained_by_trend" in rejection_codes(c)


def test_near_duplicate_columns_are_an_artefact_not_a_finding():
    df, _ = daily_frame()
    df["other"] = df["value"] * 2.54
    c = evaluate_one("correlation", df, x="value", y="other")
    assert "data_quality" in rejection_codes(c)


def test_too_little_data_is_not_tested_at_all():
    df, _ = daily_frame(n_days=20, per_day=1)
    candidate = search.make_candidate("correlation", "t", x="value", y="other")
    result = measures.measure(candidate, df, CONFIG)
    assert not result["ok"] and "pairs" in result["reason"]


def test_total_rising_only_because_of_more_rows_is_trivial():
    """Per-record value is flat; row count grows; so the total grows. Count is the finding, total is not."""
    rng = np.random.default_rng(5)
    days = pd.date_range("2022-01-01", periods=720, freq="D")
    weights = np.linspace(1, 3, len(days))
    when = days[np.sort(rng.choice(len(days), 6000, p=weights / weights.sum()))]
    df = pd.DataFrame({"when": when, "value": rng.gamma(4, 5, len(when))})
    candidates = []
    for kind, cols in [("trend_count", {}), ("trend_mean", {"value": "value"}), ("trend_total", {"value": "value"})]:
        candidate = search.make_candidate(kind, "t", time="when", freq="M", **cols)
        candidate["result"] = measures.measure(candidate, df, CONFIG)
        candidates.append(candidate)
    filters.evaluate(candidates, {"t": df}, {"t": TYPES}, {"t": set()}, CONFIG)
    by_kind = {c["kind"]: c for c in candidates}
    assert by_kind["trend_count"]["status"] == "accepted"
    assert by_kind["trend_mean"]["status"] == "rejected"
    assert rejection_codes(by_kind["trend_total"]) == {"trivially_expected"}
