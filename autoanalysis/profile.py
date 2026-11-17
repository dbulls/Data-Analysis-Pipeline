"""Describe every table and column: type, range, distribution, missing, duplicates."""

import numpy as np

from . import infer


def profile_column(column_name, raw_series, column_type, config):
    trimmed = infer.trim(raw_series)
    missing = infer.is_missing(trimmed, config)
    values = trimmed[~missing]
    n_total = len(raw_series)

    profile = {
        "type": column_type,
        "rows": n_total,
        "missing": int(missing.sum()),
        "missing_rate": float(missing.mean()) if n_total else 0.0,
        "distinct": int(values.nunique()),
        "values_with_extra_whitespace": int((raw_series.astype(str) != trimmed).sum()),
    }

    if column_type in ("numeric", "boolean"):
        numbers = infer.parse_numbers(values) if column_type == "numeric" else infer.parse_booleans(values)
        if column_type == "boolean" and numbers.notna().sum() == 0:
            numbers = infer.parse_numbers(values)   # 0/1 booleans
        profile["unparseable"] = int(numbers.isna().sum())
        numbers = numbers.dropna()
        if len(numbers):
            counts, edges = np.histogram(numbers, bins=10)
            profile.update({
                "min": numbers.min(), "max": numbers.max(),
                "mean": numbers.mean(), "median": numbers.median(),
                "std": numbers.std(), "q25": numbers.quantile(0.25),
                "q75": numbers.quantile(0.75), "skew": numbers.skew(),
                "histogram": {"bin_edges": edges, "counts": counts},
            })
    elif column_type == "datetime":
        dates, info = infer.parse_dates(values)
        profile["unparseable"] = int(dates.isna().sum())
        profile["formats_found"] = info["formats_used"]
        profile["ambiguous_day_month_order"] = info["ambiguous"]
        dates = dates.dropna()
        if len(dates):
            profile.update({"min": dates.min(), "max": dates.max(),
                            "span_days": (dates.max() - dates.min()).days})
    elif column_type in ("categorical", "identifier"):
        top = values.value_counts().head(10)
        profile["top_values"] = {str(k): int(v) for k, v in top.items()}
        profile["duplicated_values"] = int(values.duplicated().sum())
        if column_type == "categorical":
            lowered = values.str.lower()
            profile["case_variants"] = int(values.nunique() - lowered.nunique())
    elif column_type == "text":
        lengths = values.str.len()
        profile.update({"min_length": lengths.min(), "max_length": lengths.max(),
                        "mean_length": lengths.mean()})
    return profile


def profile_tables(raw_tables, config):
    """Return (profile, types) where types is {table: {column: type}}."""
    profile, types = {}, {}
    for name, raw_df in raw_tables.items():
        types[name] = infer.infer_table_types(name, raw_df, config)
        profile[name] = {
            "rows": len(raw_df),
            "columns": len(raw_df.columns),
            "exact_duplicate_rows": int(raw_df.duplicated().sum()),
            "column_profiles": {
                column: profile_column(column, raw_df[column], types[name][column], config)
                for column in raw_df.columns
            },
        }
    return profile, types
