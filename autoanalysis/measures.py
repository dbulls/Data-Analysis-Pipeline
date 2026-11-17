"""One measurement function per kind of pattern.

Every function takes a DataFrame plus the column names involved and returns
either
    {"ok": True, "n", "test", "p_value", "effect_name", "effect_size",
     "direction", "summary": {...}}
or
    {"ok": False, "reason": "..."}   when the data is too thin to test.

"direction" is what must stay the same for a pattern to count as replicated:
a sign (+1 / -1) or a profile ({label: value}) whose shape must be preserved.
The size compared between full data and subsets is "effect_size", unless the
result supplies a "replication_value" that is fairer to compare on less data.

The filters call these same functions again on subsets of the data (outliers
removed, one record removed, each half), which is why they are pure functions
of the DataFrame they are given.
"""

import math
import zlib

import numpy as np
import pandas as pd
from scipy import stats

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
FREQUENCY_NAMES = {"M": "month", "W": "week", "D": "day"}


def too_thin(reason):
    return {"ok": False, "reason": reason}


def sign(value):
    return 1 if value > 0 else (-1 if value < 0 else 0)


def scaled_minimum(minimum, relaxed, config, floor=2):
    """Minimum sample size; smaller when re-checking on a subset of the data."""
    if not relaxed:
        return minimum
    return max(floor, int(round(minimum * config.subset_minimum_factor)))


def stable_rng(config, candidate_id):
    """Random generator that depends only on the seed and the candidate, not on run order."""
    return np.random.default_rng([config.seed, zlib.crc32(candidate_id.encode())])


# --------------------------------------------------------------------------
# Time series helpers
# --------------------------------------------------------------------------

def choose_frequency(dates, config):
    """Coarsest of month / week / day that still gives enough periods, else None."""
    dates = dates.dropna()
    for freq in ("M", "W", "D"):
        if len(dates) and dates.dt.to_period(freq).nunique() >= config.min_periods:
            return freq
    return None


def period_table(df, time_col, value_col, freq):
    """One row per time period: mean, sum, count, and position (periods since the first)."""
    columns = [time_col] + ([value_col] if value_col else [])
    data = df[columns].dropna()
    if data.empty:
        return pd.DataFrame(columns=["mean", "sum", "count", "position"])
    periods = data[time_col].dt.to_period(freq)
    if value_col:
        table = data.groupby(periods)[value_col].agg(["mean", "sum", "count"])
    else:
        table = data.groupby(periods).size().to_frame("count")
        # Periods with no records at all are real zeros for a count series.
        full_range = pd.period_range(table.index.min(), table.index.max(), freq=freq)
        table = table.reindex(full_range, fill_value=0)
    table = table.sort_index()
    first = table.index[0].ordinal
    table["position"] = [p.ordinal - first for p in table.index]
    return table


def mean_series(df, cols, config, relaxed):
    """Period means, using only periods with enough records behind them."""
    table = period_table(df, cols["time"], cols["value"], cols["freq"])
    needed = scaled_minimum(config.min_records_per_period, relaxed, config)
    return table[table["count"] >= needed]


def count_or_total_series(df, cols, which):
    """Period counts or sums. The first and last periods are dropped because
    they are usually only partly covered by the data. Monthly figures are
    scaled to a month of average length, so February is not "low" by construction."""
    table = period_table(df, cols["time"], cols.get("value"), cols["freq"])
    table = table.iloc[1:-1].copy()
    table["y"] = table[which].astype(float)
    if cols["freq"] == "M" and len(table):
        table["y"] *= 30.4375 / table.index.days_in_month
    return table


# --------------------------------------------------------------------------
# Trends
# --------------------------------------------------------------------------

def measure_trend(df, cols, config, relaxed, which):
    """Monotonic trend of a per-period series: Mann-Kendall test + Theil-Sen slope."""
    if which == "mean":
        table = mean_series(df, cols, config, relaxed).rename(columns={"mean": "y"})
    else:
        table = count_or_total_series(df, cols, "count" if which == "count" else "sum")
    needed = scaled_minimum(config.min_periods, relaxed, config, floor=5)
    if len(table) < needed:
        return too_thin(f"only {len(table)} usable periods; need {needed}")
    x, y = table["position"].to_numpy(float), table["y"].to_numpy(float)
    if np.ptp(y) == 0:
        return too_thin("series is constant")

    tau, p_value = stats.kendalltau(x, y)
    slope, intercept, slope_low, slope_high = stats.theilslopes(y, x)
    start, end = intercept + slope * x[0], intercept + slope * x[-1]
    percent_change = (end - start) / abs(start) * 100 if start != 0 else None
    return {
        "ok": True, "n": len(table),
        "test": "Mann-Kendall trend test (Kendall tau of period value vs. time)",
        "p_value": p_value, "effect_name": "Kendall tau", "effect_size": abs(tau),
        "direction": sign(tau),
        # Tau shrinks on a shorter time span even when the trend is unchanged,
        # so subsets are compared on the rate of change instead.
        "replication_name": "slope per period", "replication_value": abs(slope),
        "summary": {
            "tau": tau, "slope_per_period": slope,
            "slope_95ci": [slope_low, slope_high],
            "fitted_start": start, "fitted_end": end,
            "percent_change_over_range": percent_change,
            "first_period": str(table.index[0]), "last_period": str(table.index[-1]),
            "period": FREQUENCY_NAMES[cols["freq"]],
            "records": int(table["count"].sum()),
        },
    }


# --------------------------------------------------------------------------
# Seasonality
# --------------------------------------------------------------------------

def measure_seasonality(df, cols, config, relaxed):
    """Does the value differ by month-of-year or day-of-week, after removing a linear trend?"""
    data = df[[cols["time"], cols["value"]]].dropna()
    if len(data) < 3:
        return too_thin("no data")
    time, value = data[cols["time"]], data[cols["value"]]
    by_month = cols["cycle"] == "month_of_year"
    cycle_days = 365.25 if by_month else 7
    cycles_needed = 1 if relaxed else (2 if by_month else 8)
    span_days = (time.max() - time.min()).total_seconds() / 86400
    if span_days < cycles_needed * cycle_days:
        return too_thin(f"time span covers fewer than {cycles_needed} full cycles")
    if value.std() == 0:
        return too_thin("value is constant")

    days = (time - time.min()).dt.total_seconds() / 86400
    slope, intercept = np.polyfit(days, value, 1)
    detrended = value - (slope * days + intercept)

    bucket = time.dt.month - 1 if by_month else time.dt.dayofweek
    labels = MONTH_NAMES if by_month else DAY_NAMES
    needed = scaled_minimum(config.min_group_size, relaxed, config, floor=5)
    sizes = bucket.value_counts()
    if len(sizes) < len(labels) or sizes.min() < needed:
        return too_thin(f"not every {'month' if by_month else 'weekday'} has {needed} records")

    groups = [detrended[bucket == b] for b in range(len(labels))]
    statistic, p_value = stats.kruskal(*groups)
    bucket_means = detrended.groupby(bucket).mean()
    spread = bucket_means.max() - bucket_means.min()
    profile = {labels[b]: bucket_means[b] + value.mean() for b in range(len(labels))}
    return {
        "ok": True, "n": len(data),
        "test": "Kruskal-Wallis across calendar buckets, on linearly detrended values",
        "p_value": p_value,
        "effect_name": "seasonal range in standard deviations",
        "effect_size": spread / value.std(),
        "direction": profile,
        "summary": {
            "bucket_means": profile,
            "peak": labels[int(bucket_means.idxmax())],
            "trough": labels[int(bucket_means.idxmin())],
            "range_percent_of_mean": spread / abs(value.mean()) * 100 if value.mean() != 0 else None,
            "span_days": span_days,
        },
    }


# --------------------------------------------------------------------------
# Breakpoints
# --------------------------------------------------------------------------

def strongest_split(y, margin):
    """Best place to cut rows of y into before/after. Works on a 2-D array (one series per row).

    Returns (statistic, split_index) per row; split_index is the first "after" position.
    """
    n = y.shape[1]
    k = np.arange(margin, n - margin + 1)
    cumulative = np.cumsum(y, axis=1)
    left_mean = cumulative[:, k - 1] / k
    right_mean = (cumulative[:, [-1]] - cumulative[:, k - 1]) / (n - k)
    statistic = np.abs(left_mean - right_mean) * np.sqrt(k * (n - k) / n)
    best = statistic.argmax(axis=1)
    return statistic[np.arange(len(y)), best], k[best]


def measure_breakpoint(df, cols, config, relaxed, candidate_id, need_p=True):
    """Single step change in a period-mean series, tested by permutation."""
    table = mean_series(df, cols, config, relaxed)
    needed = scaled_minimum(config.min_periods, relaxed, config, floor=6)
    if len(table) < needed:
        return too_thin(f"only {len(table)} usable periods; need {needed}")
    y = table["mean"].to_numpy(float)
    n = len(y)
    margin = max(2 if relaxed else 3, int(0.15 * n))
    if n - 2 * margin < 1 or np.ptp(y) == 0:
        return too_thin("series too short or constant")

    observed, split = strongest_split(y[None, :], margin)
    observed, split = observed[0], int(split[0])

    p_value = None
    if need_p:
        # Shuffling the periods destroys any step; taking the best split in
        # each shuffle accounts for having searched over all split points.
        rng = stable_rng(config, candidate_id)
        shuffled = rng.permuted(np.tile(y, (config.n_permutations, 1)), axis=1)
        null_statistics, _ = strongest_split(shuffled, margin)
        p_value = (1 + (null_statistics >= observed).sum()) / (config.n_permutations + 1)

    # Is a step a better description than a straight line?
    x = table["position"].to_numpy(float)
    step_fit = np.where(np.arange(n) < split, y[:split].mean(), y[split:].mean())
    line_fit = np.polyval(np.polyfit(x, y, 1), x)
    sse_step, sse_line = ((y - step_fit) ** 2).sum(), ((y - line_fit) ** 2).sum()

    # Effect size from the individual records either side of the break.
    break_start = table.index[split].start_time
    data = df[[cols["time"], cols["value"]]].dropna()
    before = data.loc[data[cols["time"]] < break_start, cols["value"]]
    after = data.loc[data[cols["time"]] >= break_start, cols["value"]]
    pooled_sd = math.sqrt(((len(before) - 1) * before.var() + (len(after) - 1) * after.var())
                          / max(1, len(before) + len(after) - 2))
    if not pooled_sd > 0:
        return too_thin("no variation in records")
    cohens_d = (after.mean() - before.mean()) / pooled_sd
    return {
        "ok": True, "n": n,
        "test": f"Permutation test of the strongest before/after split ({config.n_permutations} shuffles)",
        "p_value": p_value, "effect_name": "Cohen's d (after vs. before)",
        "effect_size": abs(cohens_d), "direction": sign(cohens_d),
        "summary": {
            "break_period": str(table.index[split]), "break_date": break_start,
            "mean_before": before.mean(), "mean_after": after.mean(),
            "percent_change": (after.mean() - before.mean()) / abs(before.mean()) * 100
            if before.mean() != 0 else None,
            "records_before": len(before), "records_after": len(after),
            "step_fits_better_than_line": bool(sse_step < sse_line),
            "sse_step": sse_step, "sse_line": sse_line,
            "period": FREQUENCY_NAMES[cols["freq"]],
        },
    }


# --------------------------------------------------------------------------
# Anomalies
# --------------------------------------------------------------------------

NEIGHBOURS = 3    # periods either side used as the "usual level"


def neighbour_positions(i, n):
    return [j for j in range(max(0, i - NEIGHBOURS), min(n, i + NEIGHBOURS + 1)) if j != i]


def anomaly_scores(df, cols, config, relaxed, which):
    """Robust z-score of each period against the median of its neighbouring periods.

    The period itself is left out of its own baseline, otherwise the baseline
    follows the data and makes ordinary wobble look extreme.
    """
    if which == "mean":
        table = mean_series(df, cols, config, relaxed).rename(columns={"mean": "y"})
    else:
        table = count_or_total_series(df, cols, "count")
    needed = scaled_minimum(config.min_periods, relaxed, config, floor=6)
    if len(table) < needed:
        return None
    table = table.copy()
    y = table["y"].to_numpy(float)
    table["baseline"] = [np.median(y[neighbour_positions(i, len(y))]) for i in range(len(y))]
    residual = table["y"] - table["baseline"]
    scale = 1.4826 * (residual - residual.median()).abs().median()
    if not scale > 0:
        return None
    table["z"] = residual / scale
    return table


def find_anomalies(df, cols, config, which):
    """Period labels of the most extreme periods that clear the z threshold."""
    table = anomaly_scores(df, cols, config, False, which)
    if table is None:
        return []
    extreme = table[table["z"].abs() >= config.anomaly_z_threshold]
    extreme = extreme.reindex(extreme["z"].abs().sort_values(ascending=False, kind="stable").index)
    return [str(period) for period in extreme.index[:config.max_anomalies_per_series]]


def measure_anomaly(df, cols, config, relaxed, which):
    """Is one period out of line with its neighbours?

    Two things must both be true: the period stands out against the usual
    period-to-period wobble (robust z-score, the effect size), and the gap is
    bigger than the noise expected from the records themselves (the p-value).
    """
    table = anomaly_scores(df, cols, config, relaxed, which)
    if table is None:
        return too_thin("too few usable periods, or no variation")
    labels = [str(period) for period in table.index]
    if cols["period"] not in labels:
        return too_thin("the period has too few records in this subset")
    i = labels.index(cols["period"])
    row = table.iloc[i]
    z = row["z"]

    if which == "mean":
        data = df[[cols["time"], cols["value"]]].dropna()
        record_period = data[cols["time"]].dt.to_period(cols["freq"]).astype(str)
        inside = data.loc[record_period == cols["period"], cols["value"]]
        nearby_labels = [labels[j] for j in neighbour_positions(i, len(labels))]
        nearby = data.loc[record_period.isin(nearby_labels), cols["value"]]
        if len(inside) < 2 or len(nearby) < 2:
            return too_thin("too few records to compare the period with its neighbours")
        p_single = stats.ttest_ind(inside, nearby, equal_var=False).pvalue
        test = (f"Welch t-test of the period's records vs. records in the {NEIGHBOURS} periods either side")
    else:
        # Counts: must beat both the usual wobble and plain Poisson counting noise.
        poisson_z = (row["y"] - row["baseline"]) / math.sqrt(max(row["baseline"], 1.0))
        p_single = 2 * stats.norm.sf(min(abs(z), abs(poisson_z)))
        test = "Robust z-score and Poisson counting noise, whichever is less extreme (normal approximation)"
    if not np.isfinite(p_single):
        return too_thin("no variation in the records")
    # Bonferroni for having picked the most extreme of len(table) periods.
    p_value = min(1.0, p_single * len(table))
    return {
        "ok": True, "n": int(row["count"]),
        "test": test + f", Bonferroni-adjusted for {len(table)} periods",
        "p_value": p_value, "effect_name": "robust z-score", "effect_size": abs(z),
        "direction": sign(z),
        "summary": {
            "period_label": cols["period"], "value": row["y"], "baseline": row["baseline"],
            "percent_from_baseline": (row["y"] - row["baseline"]) / abs(row["baseline"]) * 100
            if row["baseline"] != 0 else None,
            "records_in_period": int(row["count"]), "periods_examined": len(table),
            "period": FREQUENCY_NAMES[cols["freq"]],
        },
    }


# --------------------------------------------------------------------------
# Group differences
# --------------------------------------------------------------------------

def group_diff_data(df, cols):
    """Rows to compare, and whether the value is a yes/no rate.

    If the grouping column came from a parent table, records are first
    reduced to one row per parent (its median, or its rate for yes/no values),
    so that a parent with many records does not count many times.
    """
    data = df[[c for c in (cols["group"], cols["value"], cols.get("unit")) if c]].dropna()
    is_rate = set(data[cols["value"]].unique()) <= {0.0, 1.0}
    if cols.get("unit"):
        data = data.groupby(cols["unit"]).agg({cols["group"]: "first",
                                               cols["value"]: "mean" if is_rate else "median"})
    return data, is_rate


def measure_group_diff(df, cols, config, relaxed):
    """Does a numeric (or yes/no) value differ between categories? Kruskal-Wallis."""
    data, is_rate = group_diff_data(df, cols)
    needed = scaled_minimum(config.min_group_size, relaxed, config, floor=5)
    sizes = data[cols["group"]].value_counts()
    kept = sorted(sizes[sizes >= needed].index)
    left_out = sorted(str(g) for g in sizes[sizes < needed].index)
    if len(kept) < 2:
        return too_thin(f"fewer than two groups with at least {needed} records")
    data = data[data[cols["group"]].isin(kept)]
    samples = [data.loc[data[cols["group"]] == g, cols["value"]] for g in kept]
    try:
        statistic, p_value = stats.kruskal(*samples)
    except ValueError:
        return too_thin("all values identical")
    epsilon_squared = min(1.0, max(0.0, statistic / (len(data) - 1)))

    # A yes/no value is summarised by its rate; anything else by its median,
    # which one extreme record cannot move. Groups are ordered by mean rank,
    # which is exactly what the test compares.
    ranks = data[cols["value"]].rank()
    mean_rank = {str(g): ranks[data[cols["group"]] == g].mean() for g in kept}
    groups = {
        str(g): {"n": len(s), "typical": s.mean() if is_rate else s.median(),
                 "mean": s.mean(), "median": s.median(),
                 "standard_error": s.std() / math.sqrt(len(s))}
        for g, s in zip(kept, samples)
    }
    highest, lowest = max(mean_rank, key=mean_rank.get), min(mean_rank, key=mean_rank.get)
    low_value = groups[lowest]["typical"]
    return {
        "ok": True, "n": len(data), "test": "Kruskal-Wallis rank test",
        "p_value": p_value, "effect_name": "epsilon-squared", "effect_size": epsilon_squared,
        "direction": mean_rank,
        "summary": {
            "groups": groups, "highest": highest, "lowest": lowest,
            "typical_is": "rate" if is_rate else "median",
            "compared_per": cols.get("unit"),
            "percent_gap_highest_vs_lowest": (groups[highest]["typical"] - low_value) / abs(low_value) * 100
            if low_value != 0 else None,
            "groups_left_out_as_too_small": left_out,
        },
    }


# --------------------------------------------------------------------------
# Relationships between numeric variables
# --------------------------------------------------------------------------

def correlation_data(df, cols):
    """Complete pairs. If y came from a parent table, one row per parent (median x)."""
    data = df[[c for c in (cols["x"], cols["y"], cols.get("unit")) if c]].dropna()
    if cols.get("unit"):
        data = data.groupby(cols["unit"]).agg({cols["x"]: "median", cols["y"]: "first"})
    return data


def measure_correlation(df, cols, config, relaxed):
    data = correlation_data(df, cols)
    needed = scaled_minimum(config.min_pairs, relaxed, config, floor=10)
    if len(data) < needed:
        return too_thin(f"only {len(data)} complete pairs; need {needed}")
    x, y = data[cols["x"]], data[cols["y"]]
    if x.nunique() < 3 or y.nunique() < 3:
        return too_thin("a variable has fewer than 3 distinct values")
    most_common_share = max(x.value_counts(normalize=True).iloc[0], y.value_counts(normalize=True).iloc[0])
    if most_common_share > 0.5:
        return too_thin(f"a variable has the same value in {most_common_share:.0%} of pairs, "
                        "so a rank correlation would mostly measure ties")
    rho, p_value = stats.spearmanr(x, y)
    pearson_r = stats.pearsonr(x, y)[0]
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "ok": True, "n": len(data), "test": "Spearman rank correlation",
        "p_value": p_value, "effect_name": "Spearman rho", "effect_size": abs(rho),
        "direction": sign(rho),
        "summary": {"rho": rho, "pearson_r": pearson_r, "compared_per": cols.get("unit"),
                    "line_slope": slope, "line_intercept": intercept},
    }


# --------------------------------------------------------------------------
# Concentration
# --------------------------------------------------------------------------

def top_share(values, top_fraction=0.2):
    """Share of the total held by the largest `top_fraction` of units. Rows = samples."""
    values = np.sort(np.atleast_2d(values), axis=1)[:, ::-1]
    k = max(1, math.ceil(top_fraction * values.shape[1]))
    return values[:, :k].sum(axis=1) / values.sum(axis=1)


def gini(values):
    values = np.sort(np.asarray(values, float))
    n = len(values)
    return (2 * (np.arange(1, n + 1) * values).sum() / (n * values.sum())) - (n + 1) / n


def concentration_units(df, cols):
    """The per-unit totals being compared: per entity if one is given, else per record."""
    entity, value = cols.get("entity"), cols.get("value")
    if entity and value:
        return df[[entity, value]].dropna().groupby(entity)[value].sum()
    if entity:
        return df[entity].dropna().value_counts().sort_index()
    return df[value].dropna()


def measure_concentration(df, cols, config, relaxed, candidate_id, need_p=True):
    """Do the top 20% of units hold more than the threshold share of the total?"""
    units = concentration_units(df, cols)
    needed = scaled_minimum(config.min_units_concentration, relaxed, config, floor=10)
    if len(units) < needed:
        return too_thin(f"only {len(units)} units; need {needed}")
    values = units.to_numpy(float)
    if (values < 0).any() or values.sum() <= 0:
        return too_thin("values are negative or sum to zero, so shares are undefined")

    threshold = config.effect_thresholds["concentration"][0]
    share = top_share(values)[0]
    p_value, interval = None, [None, None]
    if need_p:
        # Bootstrap: resample units, see how often the share falls to the threshold.
        rng = stable_rng(config, candidate_id)
        resampled = values[rng.integers(0, len(values), (config.n_bootstrap, len(values)))]
        shares = top_share(resampled)
        p_value = (1 + (shares <= threshold).sum()) / (config.n_bootstrap + 1)
        interval = [np.percentile(shares, 2.5), np.percentile(shares, 97.5)]
    return {
        "ok": True, "n": len(values),
        "test": f"Bootstrap test that the top-20% share exceeds {threshold:.0%} "
                f"({config.n_bootstrap} resamples)",
        "p_value": p_value, "effect_name": "share of total held by top 20%",
        "effect_size": share, "direction": 1,
        "summary": {"top_20_percent_share": share, "share_95ci": interval,
                    "gini": gini(values), "units": len(values),
                    "share_if_all_equal": 0.2,
                    "largest_unit_share": values.max() / values.sum()},
    }


# --------------------------------------------------------------------------
# Distribution shift
# --------------------------------------------------------------------------

def measure_dist_shift(df, cols, config, relaxed):
    """Did the distribution of a value change between the first and second half of the time range?"""
    data = df[[cols["time"], cols["value"]]].dropna()
    if data.empty:
        return too_thin("no data")
    time = data[cols["time"]]
    midpoint = time.min() + (time.max() - time.min()) / 2
    first = data.loc[time < midpoint, cols["value"]]
    second = data.loc[time >= midpoint, cols["value"]]
    needed = scaled_minimum(config.min_pairs, relaxed, config, floor=10)
    if min(len(first), len(second)) < needed:
        return too_thin(f"a half has fewer than {needed} records")
    statistic, p_value = stats.ks_2samp(first, second)
    shift = second.median() - first.median()
    if shift == 0:
        shift = second.mean() - first.mean()
    return {
        "ok": True, "n": len(data), "test": "Two-sample Kolmogorov-Smirnov test",
        "p_value": p_value, "effect_name": "KS D (largest gap between the two distributions)",
        "effect_size": statistic, "direction": sign(shift),
        "summary": {"midpoint": midpoint, "median_first_half": first.median(),
                    "median_second_half": second.median(), "mean_first_half": first.mean(),
                    "mean_second_half": second.mean(), "std_first_half": first.std(),
                    "std_second_half": second.std(),
                    "records_first_half": len(first), "records_second_half": len(second)},
    }


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------

def measure(candidate, df, config, relaxed=False, need_p=True):
    """Run the right measurement for a candidate on the given data."""
    kind, cols, cid = candidate["kind"], candidate["columns"], candidate["id"]
    if kind == "trend_mean":
        return measure_trend(df, cols, config, relaxed, "mean")
    if kind == "trend_count":
        return measure_trend(df, cols, config, relaxed, "count")
    if kind == "trend_total":
        return measure_trend(df, cols, config, relaxed, "total")
    if kind == "seasonality":
        return measure_seasonality(df, cols, config, relaxed)
    if kind == "breakpoint":
        return measure_breakpoint(df, cols, config, relaxed, cid, need_p)
    if kind == "anomaly_mean":
        return measure_anomaly(df, cols, config, relaxed, "mean")
    if kind == "anomaly_count":
        return measure_anomaly(df, cols, config, relaxed, "count")
    if kind == "group_diff":
        return measure_group_diff(df, cols, config, relaxed)
    if kind == "correlation":
        return measure_correlation(df, cols, config, relaxed)
    if kind == "concentration":
        return measure_concentration(df, cols, config, relaxed, cid, need_p)
    if kind == "dist_shift":
        return measure_dist_shift(df, cols, config, relaxed)
    raise ValueError(f"Unknown kind of finding: {kind}")
