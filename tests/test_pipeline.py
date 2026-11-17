"""End to end on the synthetic dataset: planted patterns are found, planted noise is rejected."""

import filecmp
import json

from autoanalysis.config import Config
from autoanalysis.pipeline import run_pipeline

NOISE_COLUMNS = {"noise_a", "noise_b", "channel", "returned", "customers.region", "customers.age"}


def find(findings, kind, **columns):
    return [f for f in findings if f["kind"] == kind
            and all(f["columns"].get(k) == v for k, v in columns.items())]


def codes(candidate):
    return {r["code"] for r in candidate["rejections"]}


# ---- Planted patterns are found ---------------------------------------------

def test_finds_rising_order_amount(run):
    (finding,) = find(run["accepted"], "trend_mean", value="amount")
    assert finding["result"]["direction"] == 1
    assert 20 < finding["result"]["summary"]["percent_change_over_range"] < 50   # planted: ~35%


def test_finds_segment_difference_through_the_table_join(run):
    (finding,) = find(run["accepted"], "group_diff", group="customers.segment", value="amount")
    assert finding["result"]["summary"]["highest"] == "Enterprise"
    assert finding["result"]["summary"]["lowest"] == "Consumer"


def test_finds_delivery_time_vs_satisfaction_relationship(run):
    (finding,) = find(run["accepted"], "correlation", x="delivery_days", y="satisfaction")
    assert finding["result"]["direction"] == -1


def test_finds_customer_concentration(run):
    (finding,) = [f for f in find(run["accepted"], "concentration", entity="customer_id")
                  if "value" not in f["columns"]]
    assert finding["result"]["effect_size"] > 0.6


def test_finds_delivery_days_breakpoint_at_planted_date(run):
    (finding,) = find(run["accepted"], "breakpoint", value="delivery_days")
    assert finding["result"]["summary"]["break_period"] == "2023-09"
    assert finding["result"]["direction"] == -1


def test_finds_yearly_seasonality_in_quantity(run):
    (finding,) = find(run["accepted"], "seasonality", value="quantity", cycle="month_of_year")
    assert finding["result"]["summary"]["peak"] in ("Nov", "Dec")


def test_finds_discount_anomaly_in_planted_month(run):
    (finding,) = find(run["accepted"], "anomaly_mean", value="discount_pct")
    assert finding["columns"]["period"] == "2023-03"


def test_finds_growth_in_order_volume(run):
    (finding,) = find(run["accepted"], "trend_count", time="order_date")
    assert finding["result"]["direction"] == 1


# ---- Planted traps are rejected ---------------------------------------------

def test_no_accepted_finding_involves_a_noise_column(run):
    for finding in run["accepted"]:
        involved = set(finding["columns"].values())
        assert not (involved & NOISE_COLUMNS), finding["id"]


def test_noise_columns_were_actually_tested(run):
    """The noise is rejected by the filters, not by never being looked at."""
    tested = {v for c in run["rejected"] for v in c["columns"].values() if isinstance(v, str)}
    assert NOISE_COLUMNS <= tested


def test_single_huge_order_is_rejected_as_single_record(run):
    (trap,) = find(run["rejected"], "anomaly_mean", value="amount", period="2022-08")
    assert "driven_by_single_record" in codes(trap)


def test_total_quantity_growth_is_rejected_as_trivial(run):
    (trap,) = find(run["rejected"], "trend_total", value="quantity")
    assert codes(trap) == {"trivially_expected"}


def test_seasonal_peaks_are_not_reported_again_as_anomalies(run):
    assert not find(run["accepted"], "anomaly_mean", value="quantity")


def test_per_customer_totals_are_not_reported_as_separate_concentrations(run):
    assert not [f for f in find(run["accepted"], "concentration", entity="customer_id")
                if "value" in f["columns"]]


# ---- Report and audit trail --------------------------------------------------

def test_every_finding_has_confidence_chart_and_checks(run, output_dir):
    assert run["accepted"]
    for finding in run["accepted"]:
        assert finding["confidence"] in ("high", "medium", "low")
        assert (output_dir / finding["chart"]).exists()
        assert set(finding["checks"]) == {"outliers_removed", "single_record_removed", "split_halves"}
        assert finding["q_value"] <= Config().alpha


def test_every_rejection_states_a_reason(run):
    assert run["rejected"]
    for candidate in run["rejected"]:
        assert candidate["rejections"] and all(r["explanation"] for r in candidate["rejections"])


def test_intermediate_results_are_saved(run, output_dir):
    for name in ["01_profile.json", "01_relationships.json", "02_cleaning_log.json",
                 "03_candidates.json", "03_skipped.json", "04_accepted.json", "04_rejected.json",
                 "06_report.md", "06_report.html", "assumptions.json", "run_manifest.json",
                 "02_cleaned/orders.csv", "02_cleaned/orders__outlier_flags.csv"]:
        assert (output_dir / name).exists(), name
    candidates = json.loads((output_dir / "03_candidates.json").read_text())
    assert len(candidates) == len(run["accepted"]) + len(run["rejected"])


def test_report_has_required_sections(run, output_dir):
    text = (output_dir / "06_report.md").read_text()
    for heading in ["## Top findings", "## Patterns that looked promising but failed the checks",
                    "## Assumptions to check before acting", "**Confidence:", "**To confirm:**"]:
        assert heading in text


def test_same_input_gives_identical_output(run, data_dir, output_dir, tmp_path):
    run_pipeline(data_dir, tmp_path, Config())
    for name in ["01_profile.json", "02_cleaning_log.json", "03_candidates.json",
                 "04_accepted.json", "04_rejected.json", "06_report.md", "06_report.html"]:
        assert filecmp.cmp(output_dir / name, tmp_path / name, shallow=False), name
