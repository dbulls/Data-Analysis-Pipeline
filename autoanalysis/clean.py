"""Conservative, logged cleaning.

Rules (each one writes a line to the cleaning log when it changes anything):
    R1  trim whitespace
    R2  turn missing-value tokens ("", "N/A", "null", ...) into real missing values
    R3  convert numeric / boolean columns; unparseable values become missing
    R4  standardise dates to ISO format; unparseable values become missing
    R5  merge category spellings that differ only by letter case
    R6  drop exact duplicate rows, only in tables that have a unique row identifier
    R7  flag outliers (never delete them)
    R8  missing values are never filled in; each analysis uses the rows it has

The raw DataFrames passed in are not modified.
"""

import pandas as pd

from . import infer


def tukey_outliers(values, multiplier):
    """Return (is_outlier, low_fence, high_fence) using Tukey's IQR fences.

    If the middle half of the data has no spread, nothing is flagged.
    """
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    spread = multiplier * (q3 - q1)
    if not spread > 0:
        return pd.Series(False, index=values.index), q1, q3
    return (values < q1 - spread) | (values > q3 + spread), q1 - spread, q3 + spread


def clean_table(table_name, raw_df, types, config):
    """Return (clean_df, outlier_flags_df, log_entries)."""
    log = []

    def record(rule, column, rows_affected, detail):
        if rows_affected:
            log.append({"table": table_name, "column": column, "rule": rule,
                        "rows_affected": int(rows_affected), "detail": detail})

    clean = pd.DataFrame(index=raw_df.index)
    for column in raw_df.columns:
        column_type = types[column]
        raw = raw_df[column].astype(str)

        # R1 trim whitespace
        text = infer.trim(raw)
        record("R1 trim whitespace", column, (text != raw).sum(),
               "Removed leading/trailing spaces and collapsed repeated spaces.")

        # R2 missing tokens
        missing = infer.is_missing(text, config)
        blank = (text == "")
        record("R2 missing tokens", column, (missing & ~blank).sum(),
               f"Values matching {sorted(t for t in config.missing_tokens if t)} set to missing.")
        text = text.mask(missing)

        if column_type in ("numeric", "boolean"):
            # R3 numeric conversion
            numbers = infer.parse_numbers(text.fillna(""))
            if column_type == "boolean" and numbers.notna().sum() == 0:
                numbers = infer.parse_booleans(text.fillna(""))
            bad = numbers.isna() & ~missing
            record("R3 unparseable number", column, bad.sum(),
                   "Could not be read as a number; set to missing. Examples: "
                   + ", ".join(sorted(text[bad].unique())[:5]))
            clean[column] = numbers
        elif column_type == "datetime":
            # R4 dates
            dates, info = infer.parse_dates(text)
            bad = dates.isna() & ~missing
            record("R4 unparseable date", column, bad.sum(),
                   "Matched none of the known date formats; set to missing. Examples: "
                   + ", ".join(sorted(text[bad].unique())[:5]))
            non_iso = sum(n for fmt, n in info["formats_used"].items() if not fmt.startswith("%Y-%m-%d"))
            record("R4 standardise date format", column, non_iso,
                   f"Converted to ISO dates. Formats found: {info['formats_used']}.")
            if info["ambiguous"]:
                record("R4 ambiguous date order", column, dates.notna().sum(),
                       f"More than one format fits every value; assumed {info['chosen_first']}. "
                       "Check day/month order.")
            clean[column] = dates
        elif column_type == "categorical":
            # R5 case variants -> most frequent spelling
            counts = text.value_counts()
            canonical = {}
            for spelling in counts.index:           # most frequent first
                canonical.setdefault(spelling.lower(), spelling)
            merged = text.map(lambda v: canonical[v.lower()] if isinstance(v, str) else v)
            changed = (merged != text) & text.notna()
            record("R5 merge case variants", column, changed.sum(),
                   "Spellings differing only by letter case merged into the most common one.")
            clean[column] = merged
        else:
            clean[column] = text

    # R6 exact duplicate rows
    duplicated = clean.duplicated(keep="first")
    kept_once = clean[~duplicated]
    has_row_id = any(types[c] == "identifier" and kept_once[c].notna().all() and kept_once[c].is_unique
                     for c in clean.columns)
    if duplicated.any():
        if has_row_id:
            record("R6 drop duplicate rows", None, duplicated.sum(),
                   "Rows identical in every column, including the row identifier, kept once.")
            clean = clean[~duplicated]
        else:
            record("R6 duplicate rows kept", None, duplicated.sum(),
                   "Identical rows found but the table has no row identifier, so they may be "
                   "legitimate repeats. Kept.")

    # R7 flag outliers (Tukey fences); flags live in a separate table
    flags = pd.DataFrame(index=clean.index)
    for column in clean.columns:
        if types[column] != "numeric":
            continue
        flags[column], low, high = tukey_outliers(clean[column], config.outlier_iqr_multiplier)
        record("R7 flag outliers", column, flags[column].sum(),
               f"Outside {config.outlier_iqr_multiplier} x IQR fences "
               f"[{low:.4g}, {high:.4g}]. Flagged only; values kept.")

    # R8 missing values: nothing is filled in
    for column in clean.columns:
        n_missing = clean[column].isna().sum()
        record("R8 missing left as missing", column, n_missing,
               "Not imputed. Analyses using this column skip these rows and report the row count used.")

    return clean, flags, log


def clean_tables(raw_tables, types, config):
    cleaned, flags, log = {}, {}, []
    for name, raw_df in raw_tables.items():
        cleaned[name], flags[name], entries = clean_table(name, raw_df, types[name], config)
        log.extend(entries)
    return cleaned, flags, log
