"""The cleaning rules: each one does what it says, logs it, and leaves raw data alone."""

import pandas as pd

from autoanalysis import clean, infer, loading
from autoanalysis.config import Config

CONFIG = Config()


def clean_one(raw):
    types = infer.infer_table_types("t", raw, CONFIG)
    cleaned, flags, log = clean.clean_table("t", raw, types, CONFIG)
    return cleaned, flags, log, types


def rules_logged(log, column):
    return {entry["rule"] for entry in log if entry["column"] == column}


def numbered(n):
    return [str(i) for i in range(n)]


def test_types_are_inferred_from_values_not_names():
    raw = pd.DataFrame({
        "a": [str(1000 + i) for i in range(40)],                     # unique whole numbers -> identifier
        "b": [f"{i * 1.5:.1f}" for i in range(40)],                  # decimals -> numeric
        "c": ["2024-01-%02d" % (i % 28 + 1) for i in range(40)],     # dates
        "d": ["red", "green", "blue", "red"] * 10,                   # few repeated labels
        "e": ["yes", "no"] * 20,
        "f": [f"free text number {i} with words" for i in range(40)],
    })
    types = infer.infer_table_types("t", raw, CONFIG)
    assert types == {"a": "identifier", "b": "numeric", "c": "datetime",
                     "d": "categorical", "e": "boolean", "f": "text"}


def test_whitespace_is_trimmed_and_logged():
    raw = pd.DataFrame({"city": ["  Paris", "Paris ", "Rome", "Rome"] * 10})
    cleaned, _, log, _ = clean_one(raw)
    assert set(cleaned["city"]) == {"Paris", "Rome"}
    assert "R1 trim whitespace" in rules_logged(log, "city")


def test_missing_tokens_become_missing_and_are_never_filled():
    raw = pd.DataFrame({"x": ["1.5", "N/A", "null", "", "2.5"] * 8})
    cleaned, _, log, _ = clean_one(raw)
    assert cleaned["x"].isna().sum() == 24          # 3 of every 5 values
    assert cleaned["x"].dropna().tolist() == [1.5, 2.5] * 8
    assert {"R2 missing tokens", "R8 missing left as missing"} <= rules_logged(log, "x")


def test_unparseable_numbers_become_missing_with_a_log_entry():
    raw = pd.DataFrame({"x": [f"{i}.5" for i in range(39)] + ["abc"]})
    cleaned, _, log, types = clean_one(raw)
    assert types["x"] == "numeric"
    assert cleaned["x"].isna().sum() == 1
    entry = next(e for e in log if e["rule"] == "R3 unparseable number")
    assert entry["rows_affected"] == 1


def test_mixed_date_formats_are_standardised():
    raw = pd.DataFrame({"when": ["2024-03-15", "03/16/2024", "2024-03-17", "17 Mar 2024"] * 10})
    cleaned, _, log, types = clean_one(raw)
    assert types["when"] == "datetime"
    assert cleaned["when"].notna().all()
    assert cleaned["when"].iloc[1] == pd.Timestamp("2024-03-16")
    assert cleaned["when"].iloc[3] == pd.Timestamp("2024-03-17")
    assert "R4 standardise date format" in rules_logged(log, "when")


def test_ambiguous_day_month_order_is_flagged():
    raw = pd.DataFrame({"when": ["01/02/2024", "03/04/2024", "05/06/2024", "07/08/2024"] * 10})
    _, _, log, _ = clean_one(raw)
    assert "R4 ambiguous date order" in rules_logged(log, "when")


def test_case_variants_merge_into_most_common_spelling():
    raw = pd.DataFrame({"tier": ["Gold"] * 20 + ["gold"] * 3 + ["Silver"] * 17})
    cleaned, _, log, _ = clean_one(raw)
    assert set(cleaned["tier"]) == {"Gold", "Silver"}
    assert "R5 merge case variants" in rules_logged(log, "tier")


def test_outliers_are_flagged_but_kept():
    raw = pd.DataFrame({"x": [f"{10 + i * 0.1:.1f}" for i in range(49)] + ["9999.5"]})
    cleaned, flags, log, _ = clean_one(raw)
    assert cleaned["x"].max() == 9999.5             # still there
    assert flags["x"].sum() == 1 and flags["x"].iloc[-1]
    assert "R7 flag outliers" in rules_logged(log, "x")


def test_duplicate_rows_dropped_only_when_table_has_a_row_identifier():
    with_id = pd.DataFrame({"row_id": numbered(30) + ["5"], "v": ["a", "b"] * 15 + ["b"]})
    cleaned, _, log, _ = clean_one(with_id)
    assert len(cleaned) == 30
    assert any(e["rule"] == "R6 drop duplicate rows" for e in log)

    without_id = pd.DataFrame({"v": ["a", "b"] * 15, "w": ["x", "y"] * 15})
    cleaned, _, log, _ = clean_one(without_id)
    assert len(cleaned) == 30                       # legitimate repeats are kept
    assert any(e["rule"] == "R6 duplicate rows kept" for e in log)


def test_raw_dataframe_is_not_modified():
    raw = pd.DataFrame({"x": [" 1.5 ", "N/A", "abc", "2.5"] * 10})
    before = raw.copy()
    clean_one(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_raw_files_are_untouched_by_a_full_run(data_dir, run):
    """File hashes after the run equal the hashes taken before it."""
    assert loading.file_hashes(data_dir) == run["hashes_before"]


def test_profile_reports_missing_cardinality_and_duplicates(run):
    orders = run["profile"]["orders"]
    assert orders["exact_duplicate_rows"] == 5
    satisfaction = orders["column_profiles"]["satisfaction"]
    assert 0.10 < satisfaction["missing_rate"] < 0.13
    assert orders["column_profiles"]["channel"]["type"] == "categorical"
    assert orders["column_profiles"]["order_date"]["formats_found"].keys() == {"%Y-%m-%d", "%m/%d/%Y"}


def test_shared_key_between_tables_is_detected(run):
    assert any(r["child_table"] == "orders" and r["parent_table"] == "customers"
               and r["child_column"] == "customer_id" for r in run["relationships"])
