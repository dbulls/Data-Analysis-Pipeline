"""Decide what type each column is by looking at its values.

Types: numeric, boolean, datetime, categorical, identifier, text, empty.

The same parsing helpers are used by the profiler and the cleaner, so the
type a column is profiled as is exactly the type it is cleaned as.
"""

import re

import pandas as pd

# Date formats tried, in priority order. When a value fits two formats
# (e.g. 03/04/2023), the format that parses more of the column wins; if they
# tie, the earlier one in this list wins and the ambiguity is logged.
DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%d %b %Y",
    "%b %d, %Y",
    "%Y%m%d",
]

PLAIN_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
THOUSANDS_NUMBER = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")

TRUE_WORDS = {"true", "yes", "y", "t"}
FALSE_WORDS = {"false", "no", "n", "f"}


def trim(series):
    """Strip leading/trailing whitespace and collapse internal runs of spaces."""
    return series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


def is_missing(series, config):
    """True where a (trimmed) text value is one of the missing-value tokens."""
    return series.str.lower().isin(config.missing_tokens)


def parse_numbers(series):
    """Text -> float. Accepts plain numbers and 1,234.5 style. Others become NaN."""
    plain = series.str.match(PLAIN_NUMBER)
    thousands = series.str.match(THOUSANDS_NUMBER)
    cleaned = series.where(~thousands, series.str.replace(",", "", regex=False))
    numbers = pd.to_numeric(cleaned.where(plain | thousands), errors="coerce")
    return numbers.astype(float)


def parse_booleans(series):
    """Text -> 1.0 / 0.0 for yes/no style words. Others become NaN."""
    lowered = series.str.lower()
    result = pd.Series(float("nan"), index=series.index, dtype=float)
    result[lowered.isin(TRUE_WORDS)] = 1.0
    result[lowered.isin(FALSE_WORDS)] = 0.0
    return result


def parse_dates(series):
    """Text -> datetime using the explicit formats above.

    Returns (parsed_series, info) where info records how many values each
    format parsed and whether the column was ambiguous between formats.
    """
    values = series.dropna()
    parsed_by_format = {}
    for fmt in DATE_FORMATS:
        # %Y%m%d would happily "parse" any 8-digit number; require 8 digits.
        candidates = values
        if fmt == "%Y%m%d":
            candidates = values[values.str.match(r"^(19|20)\d{6}$")]
        parsed = pd.to_datetime(candidates, format=fmt, errors="coerce").dropna()
        if len(parsed):
            parsed_by_format[fmt] = parsed

    # Most successful format first; list order breaks ties (stable sort).
    order = sorted(parsed_by_format, key=lambda fmt: -len(parsed_by_format[fmt]))
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    used = {}
    for fmt in order:
        still_empty = result.isna()
        fill = parsed_by_format[fmt]
        fill = fill[still_empty.reindex(fill.index)]
        if len(fill):
            result.loc[fill.index] = fill.astype("datetime64[ns]")
            used[fmt] = int(len(fill))

    ambiguous = False
    if order:
        best = len(parsed_by_format[order[0]])
        ambiguous = any(len(parsed_by_format[fmt]) == best for fmt in order[1:])
    return result, {"formats_used": used, "ambiguous": ambiguous,
                    "chosen_first": order[0] if order else None}


def looks_like_identifier_name(column_name):
    name = column_name.lower()
    return name == "id" or name.endswith(("_id", " id", "_key", "_code", "_uuid")) or name == "uuid"


def infer_column_type(column_name, raw_series, config, override=None):
    """Return one of the type names for a raw text column."""
    if override:
        return override

    values = trim(raw_series)
    values = values[~is_missing(values, config)]
    if len(values) == 0:
        return "empty"

    n = len(values)
    n_unique = values.nunique()
    unique_rate = n_unique / n
    needed = config.parse_success_rate

    numbers = parse_numbers(values)
    is_numeric = numbers.notna().mean() >= needed
    is_whole = is_numeric and (numbers.dropna() % 1 == 0).all()

    # Identifiers: named like a key, or (nearly) unique whole numbers / codes.
    name_hint = looks_like_identifier_name(column_name)
    code_like = (not is_numeric) and not values.str.contains(" ").any()
    if name_hint and (is_whole or not is_numeric):
        return "identifier"
    if n >= 20 and unique_rate >= config.identifier_unique_rate and (is_whole or code_like):
        # A unique date column is still a date, not an identifier.
        dates, _ = parse_dates(values)
        if dates.notna().mean() < needed:
            return "identifier"

    if parse_booleans(values).notna().mean() >= needed and n_unique <= 2:
        return "boolean"
    if is_numeric:
        if set(numbers.dropna().unique()) <= {0.0, 1.0} and n_unique == 2:
            return "boolean"
        return "numeric"

    dates, _ = parse_dates(values)
    if dates.notna().mean() >= needed:
        return "datetime"

    if n_unique <= config.max_categories and unique_rate <= 0.5:
        return "categorical"
    return "text"


def infer_table_types(table_name, raw_df, config):
    """Return {column: type} for one table."""
    types = {}
    for column in raw_df.columns:
        override = config.type_overrides.get(f"{table_name}.{column}")
        types[column] = infer_column_type(column, raw_df[column], config, override)
    return types
