"""Separate real signal from chance, artefacts and trivia.

For every candidate, in this order:
    1. correct p-values for multiple comparisons (Benjamini-Hochberg)
    2. require a minimum effect size
    3. re-measure without outliers
    4. re-measure without the single most influential record (or unit)
    5. re-measure on each half of the data
    6. reject data-quality artefacts, trivially expected results and duplicates
    7. rank what is left

Every check's outcome is stored on the candidate, pass or fail, so the
decision can be audited afterwards.
"""

import math

import numpy as np
import pandas as pd

from . import measures
from .clean import tukey_outliers

COUNT_KINDS = {"trend_count", "anomaly_count"}
ANOMALY_KINDS = {"anomaly_mean", "anomaly_count"}
# Kinds that are themselves about one split in time, so "first half vs second
# half" makes no sense; they are checked on alternating records instead.
NO_TIME_SPLIT = {"breakpoint", "dist_shift"}


# --------------------------------------------------------------------------
# Multiple comparisons
# --------------------------------------------------------------------------

def benjamini_hochberg(p_values):
    """Convert p-values to q-values (false discovery rate adjusted)."""
    p = np.asarray(p_values, float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p, kind="stable")
    scaled = p[order] * m / np.arange(1, m + 1)
    # A q-value can never be smaller than that of a smaller p-value.
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(scaled, 1.0)
    return q


# --------------------------------------------------------------------------
# Subsets used by the robustness checks
# --------------------------------------------------------------------------

def value_columns(candidate, types):
    """The truly numeric columns a candidate measures (yes/no columns have no outliers)."""
    cols = candidate["columns"]
    return [cols[k] for k in ("value", "x", "y") if k in cols and types.get(cols[k]) == "numeric"]


def without_outliers(df, candidate, types, config):
    """Rows where none of the candidate's numeric values is a flagged outlier. None if not applicable."""
    columns = value_columns(candidate, types)
    if not columns or candidate["kind"] in COUNT_KINDS:
        return None, 0
    is_outlier = pd.Series(False, index=df.index)
    for column in columns:
        flagged, _, _ = tukey_outliers(df[column], config.outlier_iqr_multiplier)
        is_outlier |= flagged.fillna(False)
    return df[~is_outlier], int(is_outlier.sum())


def without_most_influential(df, candidate, types):
    """Drop the single record (or, for entity concentration, the single entity)
    most able to create the pattern on its own. None if not applicable."""
    kind, cols = candidate["kind"], candidate["columns"]
    if kind in COUNT_KINDS:
        return None, None

    if kind == "concentration":
        units = measures.concentration_units(df, cols)
        if units.empty:
            return None, None
        top = units.idxmax()
        if "entity" in cols:
            return df[df[cols["entity"]] != top], f"{cols['entity']} = {top}"
        return df.drop(index=top), f"largest {cols['value']} = {units[top]:.6g}"

    columns = [cols[k] for k in ("value", "x", "y") if k in cols]
    if not columns:
        return None, None
    pool = df
    if kind == "anomaly_mean":
        # Only a record inside the anomalous period can be responsible for it.
        in_period = df[cols["time"]].dt.to_period(cols["freq"]).astype(str) == cols["period"]
        pool = df[in_period.fillna(False)]
    # Distance from the overall median, in robust units, summed over the columns.
    distance = pd.Series(0.0, index=pool.index)
    for column in columns:
        centre = df[column].median()
        scale = (df[column] - centre).abs().median() or df[column].std() or 1.0
        distance = distance.add((pool[column] - centre).abs() / scale, fill_value=0)
    distance = distance.dropna()
    if distance.empty:
        return None, None
    worst = distance.idxmax()
    described = ", ".join(f"{c} = {df.loc[worst, c]:.6g}" for c in columns)
    return df.drop(index=worst), described


def split_in_half(df, candidate, time_column):
    """Return (description, half_a, half_b)."""
    use_time = time_column is not None and candidate["kind"] not in NO_TIME_SPLIT
    if use_time:
        time = df[time_column]
        midpoint = time.min() + (time.max() - time.min()) / 2
        return (f"first vs. second half of the {time_column} range (split at {midpoint.date()})",
                df[time < midpoint], df[time >= midpoint])
    ordered = df.sort_values(time_column, kind="stable") if time_column else df
    return ("alternating records (every other row" + (f", ordered by {time_column})" if time_column else ")"),
            ordered.iloc[0::2], ordered.iloc[1::2])


# --------------------------------------------------------------------------
# Does a pattern "hold" on a subset?
# --------------------------------------------------------------------------

def same_direction(full, part):
    """Signs must match; profiles (group means, seasonal shapes) must have a similar shape."""
    if isinstance(full, dict):
        shared = [k for k in full if k in part]
        if len(shared) < 2:
            return False
        a = np.array([full[k] for k in shared], float)
        b = np.array([part[k] for k in shared], float)
        if len(shared) == 2:
            return bool(np.sign(a[0] - a[1]) == np.sign(b[0] - b[1]))
        if a.std() == 0 or b.std() == 0:
            return False
        return bool(np.corrcoef(a, b)[0, 1] >= 0.5)
    return full == part and full != 0


def compare(candidate, subset_df, config, description, retention=None):
    """Re-measure on a subset and say whether the pattern held."""
    retention = config.effect_retention if retention is None else retention
    full = candidate["result"]
    part = measures.measure(candidate, subset_df, config, relaxed=True, need_p=False)
    if not part["ok"]:
        return {"status": "not_assessable", "detail": f"{description}: {part['reason']}"}
    name = full.get("replication_name", full["effect_name"])
    full_size = full.get("replication_value", full["effect_size"])
    part_size = part.get("replication_value", part["effect_size"])

    floor = retention * full_size
    if candidate["kind"] == "concentration":
        # The threshold share is what defines this finding, so subsets must clear it too.
        floor = max(floor, config.effect_thresholds["concentration"][0])
    aligned = same_direction(full["direction"], part["direction"])
    held = aligned and part_size >= floor
    return {
        "status": "pass" if held else "fail",
        "retained": part_size / full_size if full_size else None,
        "n": part["n"], "same_direction": aligned,
        "detail": f"{description}: {name} {part_size:.3g} (full data {full_size:.3g}), "
                  f"{'same' if aligned else 'different'} direction.",
    }


def run_robustness_checks(candidate, df, types, time_column, config):
    checks = {}

    subset, n_removed = without_outliers(df, candidate, types, config)
    if subset is None:
        checks["outliers_removed"] = {"status": "not_applicable",
                                      "detail": "No numeric measured value to have outliers."}
    elif n_removed == 0:
        checks["outliers_removed"] = {"status": "pass", "retained": 1.0,
                                      "detail": "No values were flagged as outliers."}
    else:
        checks["outliers_removed"] = compare(candidate, subset, config,
                                             f"with {n_removed} outlier rows removed")

    subset, what = without_most_influential(df, candidate, types)
    if subset is None:
        checks["single_record_removed"] = {"status": "not_applicable",
                                           "detail": "A count of records cannot be driven by one value."}
    else:
        checks["single_record_removed"] = compare(candidate, subset, config,
                                                  f"with the most extreme record removed ({what})")

    if candidate["kind"] in ANOMALY_KINDS:
        checks["split_halves"] = {
            "status": "not_applicable",
            "detail": "An anomaly is one period. Splitting its records in two only re-samples the same "
                      "period, so it cannot confirm or refute it."}
        return checks

    description, half_a, half_b = split_in_half(df, candidate, time_column)
    first = compare(candidate, half_a, config, "half A", config.effect_retention_in_halves)
    second = compare(candidate, half_b, config, "half B", config.effect_retention_in_halves)
    statuses = {first["status"], second["status"]}
    status = "fail" if "fail" in statuses else ("not_assessable" if "not_assessable" in statuses else "pass")
    retained = [h["retained"] for h in (first, second) if h.get("retained") is not None]
    checks["split_halves"] = {
        "status": status, "split": description,
        "retained": min(retained) if retained else None,
        "detail": f"Split by {description}. {first['detail']} {second['detail']}",
    }
    return checks


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------

def involved_columns(candidate):
    return [v for k, v in candidate["columns"].items() if k in ("time", "value", "group", "x", "y", "entity")]


def reject(candidate, code, explanation):
    candidate["rejections"].append({"code": code, "explanation": explanation})


def apply_individual_rules(candidate, df, config):
    """Rules that look at one candidate at a time."""
    result, kind = candidate["result"], candidate["kind"]
    minimum, _ = config.effect_thresholds[kind]

    if candidate["q_value"] > config.alpha:
        reject(candidate, "not_significant",
               f"Could easily be chance: after correcting for {candidate['tests_in_family']} tests the "
               f"adjusted p-value is {candidate['q_value']:.3g} (raw p = {result['p_value']:.3g}; "
               f"needs to be at most {config.alpha}).")
    if result["effect_size"] < minimum:
        reject(candidate, "effect_too_small",
               f"Too small to matter: {result['effect_name']} is {result['effect_size']:.3g}, "
               f"below the minimum of {minimum}.")

    for column in involved_columns(candidate):
        missing_rate = df[column].isna().mean()
        if missing_rate > config.max_missing_rate:
            reject(candidate, "data_quality",
                   f"{column} is {missing_rate:.0%} missing, so the remaining rows may not be representative.")
    if kind == "correlation":
        strongest = max(abs(result["summary"]["rho"]), abs(result["summary"]["pearson_r"]))
        if strongest >= config.near_duplicate_correlation:
            reject(candidate, "data_quality",
                   f"The two columns are almost perfectly correlated ({strongest:.3f}); one is probably "
                   "computed from the other or is a duplicate, so this is not a discovery.")
    if kind == "breakpoint" and not result["summary"]["step_fits_better_than_line"]:
        reject(candidate, "better_explained_by_trend",
               "A gradual straight-line trend fits the series at least as well as a sudden step.")

    messages = {
        "outliers_removed": ("driven_by_outliers", "Disappears or reverses when flagged outliers are removed"),
        "single_record_removed": ("driven_by_single_record", "Depends on a single record"),
        "split_halves": ("does_not_replicate", "Does not hold in both halves of the data"),
    }
    for name, (code, text) in messages.items():
        check = candidate["checks"][name]
        if check["status"] == "fail":
            reject(candidate, code, f"{text}. {check['detail']}")


def apply_cross_candidate_rules(candidates):
    """Rules that compare findings with each other: trivial totals and duplicates."""
    def survivors(kind):
        return {(c["table"], c["columns"].get("time"), c["columns"].get("value"))
                for c in candidates if c["kind"] == kind and not c["rejections"]}

    rising_counts = {(t, time) for t, time, _ in survivors("trend_count")}
    mean_trends, breakpoints = survivors("trend_mean"), survivors("breakpoint")
    shifted_values = {(t, value) for t, _, value in mean_trends | breakpoints}

    count_concentration = {(c["table"], c["columns"]["entity"]): c["result"]["effect_size"]
                           for c in candidates if c["kind"] == "concentration" and not c["rejections"]
                           and "entity" in c["columns"] and "value" not in c["columns"]}
    seasonal = {(c["table"], c["columns"]["time"], c["columns"]["value"], c["columns"]["cycle"]): c
                for c in candidates if c["kind"] == "seasonality" and not c["rejections"]}

    for c in candidates:
        if c["rejections"]:
            continue
        cols = c["columns"]
        key = (c["table"], cols.get("time"), cols.get("value"))
        if c["kind"] == "trend_total":
            if key in mean_trends or key in breakpoints:
                reject(c, "redundant", f"The total moves because the per-record average of {cols['value']} "
                                       "moves; that is reported as its own finding.")
            elif (c["table"], cols["time"]) in rising_counts:
                reject(c, "trivially_expected",
                       f"The total of {cols['value']} changes only because the number of records changes; "
                       "the per-record average shows no reliable trend.")
        elif c["kind"] == "trend_mean" and key in breakpoints:
            reject(c, "redundant", "The same series is better described by a sudden step, "
                                   "reported as a breakpoint finding.")
        elif c["kind"] == "dist_shift" and (c["table"], cols["value"]) in shifted_values:
            reject(c, "redundant", f"The shift in {cols['value']} is already reported as a trend or breakpoint.")
        elif c["kind"] == "seasonality" and key in breakpoints:
            reject(c, "data_quality",
                   f"{cols['value']} has a one-off step change (reported as a breakpoint). A level shift "
                   "partway through a year looks like a calendar pattern, so this cycle cannot be trusted.")
        elif c["kind"] == "concentration" and "entity" in cols and "value" in cols:
            by_count = count_concentration.get((c["table"], cols["entity"]))
            if by_count is not None and c["result"]["effect_size"] - by_count < 0.10:
                reject(c, "trivially_expected",
                       f"The top {cols['entity']} values hold {c['result']['effect_size']:.0%} of total "
                       f"{cols['value']} simply because they hold {by_count:.0%} of the records; "
                       "that is reported as its own finding.")

    # Second pass: an "unusual" period that is really the edge of a step
    # change, or just the regular seasonal peak or trough.
    break_periods = {(c["table"], c["columns"]["time"], c["columns"]["value"]): c["result"]["summary"]["break_period"]
                     for c in candidates if c["kind"] == "breakpoint" and not c["rejections"]}
    for c in candidates:
        if c["rejections"] or c["kind"] != "anomaly_mean":
            continue
        cols = c["columns"]
        period = pd.Period(cols["period"], freq=cols["freq"])
        start = period.start_time
        break_label = break_periods.get((c["table"], cols["time"], cols["value"]))
        if break_label and abs(period.ordinal - pd.Period(break_label, freq=cols["freq"]).ordinal) <= measures.NEIGHBOURS:
            reject(c, "redundant", f"This period sits next to the step change in {cols['value']} at {break_label} "
                                   "(reported as a breakpoint), so its neighbours on one side are at a different level.")
            continue
        labels = {"month_of_year": measures.MONTH_NAMES[start.month - 1]}
        if cols["freq"] == "D":
            labels["day_of_week"] = measures.DAY_NAMES[start.dayofweek]
        for cycle, label in labels.items():
            cycle_finding = seasonal.get((c["table"], cols["time"], cols["value"], cycle))
            if cycle_finding is None or cycle_finding["rejections"]:
                continue
            bucket_means = cycle_finding["result"]["summary"]["bucket_means"]
            typical = sum(bucket_means.values()) / len(bucket_means)
            if c["rejections"]:
                break
            if measures.sign(bucket_means[label] - typical) == c["result"]["direction"]:
                reject(c, "redundant", f"{label} is regularly {'high' if c['result']['direction'] > 0 else 'low'} "
                                       f"for {cols['value']}; this is the seasonal cycle, reported as its own "
                                       "finding, not a one-off event.")


# --------------------------------------------------------------------------
# Ranking and confidence
# --------------------------------------------------------------------------

def score(candidate, config):
    result, checks = candidate["result"], candidate["checks"]
    minimum, large = config.effect_thresholds[candidate["kind"]]
    size = float(np.clip((result["effect_size"] - minimum) / (large - minimum), 0, 1))

    significance = min(1.0, -math.log10(max(candidate["q_value"], 1e-12)) / 6)
    kept = [min(1.0, check["retained"]) for check in checks.values() if check.get("retained") is not None]
    retention = min(kept) if kept else 0.5
    reliability = 0.5 * significance + 0.5 * retention
    actionability = config.actionability[candidate["kind"]]

    total = (config.weight_size * size + config.weight_reliability * reliability
             + config.weight_actionability * actionability)
    return {"size": size, "reliability": reliability, "actionability": actionability, "total": total}


def confidence_level(candidate, config):
    """high / medium / low, with the reason in words."""
    statuses = [check["status"] for check in candidate["checks"].values()]
    minimum, large = config.effect_thresholds[candidate["kind"]]
    if "not_assessable" in statuses:
        return "low", "At least one robustness check could not be run because a subset was too small."
    if candidate["q_value"] > config.alpha / 2:
        return "low", "Statistically significant, but not by a wide margin."
    if candidate["kind"] in ANOMALY_KINDS:
        return "medium", ("Clearly out of line with neighbouring periods and not caused by one record, but a "
                          "one-off period cannot be replicated within the data.")
    if candidate["q_value"] <= config.alpha / 10 and candidate["result"]["effect_size"] >= (minimum + large) / 2:
        return "high", "Strong statistical evidence, a sizeable effect, and every applicable robustness check passed."
    return "medium", "Passed every applicable check, but the effect is modest or the evidence is not overwhelming."


def mark_top(accepted, config):
    """Choose the findings for the top of the report: best first, at most
    config.max_per_kind_in_top of each kind, so one pattern repeated many
    times (e.g. dozens of similar correlations) cannot crowd out the rest."""
    per_kind = {}
    shown = 0
    for finding in accepted:
        kind = finding["kind"]
        finding["in_top"] = shown < config.top_n_findings and per_kind.get(kind, 0) < config.max_per_kind_in_top
        if finding["in_top"]:
            per_kind[kind] = per_kind.get(kind, 0) + 1
            shown += 1


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def first_time_column(types, joined):
    for column, column_type in types.items():
        if column_type == "datetime" and column not in joined:
            return column
    return None


def evaluate(candidates, views, view_types, joined_columns, config):
    """Annotate every candidate with q-value, checks, rejections, score and confidence.

    Returns (accepted, rejected), accepted sorted best first.
    """
    q_values = benjamini_hochberg([c["result"]["p_value"] for c in candidates])
    for candidate, q in zip(candidates, q_values):
        df, types = views[candidate["table"]], view_types[candidate["table"]]
        time_column = candidate["columns"].get("time") or first_time_column(types, joined_columns[candidate["table"]])
        candidate["q_value"] = float(q)
        candidate["tests_in_family"] = len(candidates)
        candidate["rejections"] = []
        candidate["checks"] = run_robustness_checks(candidate, df, types, time_column, config)
        apply_individual_rules(candidate, df, config)

    apply_cross_candidate_rules(candidates)

    accepted, rejected = [], []
    for candidate in candidates:
        if candidate["rejections"]:
            candidate["status"] = "rejected"
            rejected.append(candidate)
        else:
            candidate["status"] = "accepted"
            candidate["score"] = score(candidate, config)
            candidate["confidence"], candidate["confidence_reason"] = confidence_level(candidate, config)
            accepted.append(candidate)

    accepted.sort(key=lambda c: (-c["score"]["total"], c["id"]))
    for rank, candidate in enumerate(accepted, start=1):
        candidate["rank"] = rank
    mark_top(accepted, config)
    # Rejected: the ones that came closest to being believed first.
    rejected.sort(key=lambda c: (c["q_value"], c["id"]))
    return accepted, rejected
