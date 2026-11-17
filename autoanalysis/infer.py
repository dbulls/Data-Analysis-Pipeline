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
    "%d.%m.%Y",
    "%d %b %Y",
    "%d-%b-%Y",
    "%b %d, %Y",
    "%Y%m%d",
]

PLAIN_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
THOUSANDS_NUMBER = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
# "4,99": a comma followed by one or two digits can only be a decimal comma
# ("1,234" with three digits stays a thousands separator, as above).
DECIMAL_COMMA_NUMBER = re.compile(r"^[+-]?\d+,\d{1,2}$")

# A number with a currency or unit around it: "$16.49", "USD 24.95", "45.00 USD", "2 pcs".
CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}
WRAPPED_NUMBER = re.compile(
    r"^(?P<before>[$€£¥]|[A-Z]{3}(?= ))?\s*(?P<number>[+-]?[\d.,]+)\s*(?P<after>[A-Za-z]{1,10}\.?)?$")

TRUE_WORDS = {"true", "yes", "y", "t", "1", "1.0"}
FALSE_WORDS = {"false", "no", "n", "f", "0", "0.0"}


def trim(series):
    """Strip leading/trailing whitespace and collapse internal runs of spaces."""
    return series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


def is_missing(series, config):
    """True where a (trimmed) text value is one of the missing-value tokens."""
    return series.str.lower().isin(config.missing_tokens)


def bare_numbers(series):
    """Text that is only a number -> float: 12.5, 1,234.5 or 4,99. Others become NaN."""
    plain = series.str.match(PLAIN_NUMBER)
    thousands = series.str.match(THOUSANDS_NUMBER)
    decimal_comma = series.str.match(DECIMAL_COMMA_NUMBER)
    cleaned = series.where(~thousands, series.str.replace(",", "", regex=False))
    cleaned = cleaned.where(~decimal_comma, series.str.replace(",", ".", regex=False))
    return pd.to_numeric(cleaned.where(plain | thousands | decimal_comma), errors="coerce").astype(float)


def number_units(series):
    """The currency or unit written around each number ("$" -> "USD", "pcs"), or None.

    Returns (unit_per_value, numeric_part_per_value) for values shaped like a wrapped number.
    """
    parts = series.str.extract(WRAPPED_NUMBER)
    before = parts["before"].map(lambda u: CURRENCY_SYMBOLS.get(u, u), na_action="ignore")
    after = parts["after"].str.rstrip(".")
    after = after.where(after.isna() | ~after.str.upper().isin(CURRENCY_SYMBOLS.values()), after.str.upper())
    unit = before.fillna(after)
    unit = unit.where(~(before.notna() & after.notna()))      # "$5 USD"-style doubles: too unusual to trust
    unit = unit.str.lower().where(~unit.isin(CURRENCY_SYMBOLS.values()), unit)
    return unit, parts["number"]


def parse_numbers(series):
    """Text -> float. Others become NaN.

    Accepts plain numbers, 1,234.5 and 4,99 styles, and numbers written with a
    currency or unit ("$16.49", "45.00 USD", "2 pcs"), but only when the whole
    column uses one currency or unit. With two (say USD and EUR) the amounts
    are not comparable, so values carrying a unit are left unparsed.
    """
    numbers = bare_numbers(series)
    unit, number_part = number_units(series)
    if unit.dropna().nunique() == 1:
        wrapped = bare_numbers(number_part.fillna(""))
        numbers = numbers.fillna(wrapped.where(unit.notna()))
    return numbers


def parse_booleans(series):
    """Text -> 1.0 / 0.0 for yes/no style words and 1/0, in any mix. Others become NaN."""
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
    # Values that two formats read as different dates (03/04/2026: 4 March or 3 April?).
    # Each was read with the format that fits most of the column.
    readings = pd.DataFrame(parsed_by_format)
    two_meanings = int((readings.nunique(axis=1) > 1).sum()) if len(readings.columns) > 1 else 0
    return result, {"formats_used": used, "ambiguous": ambiguous,
                    "chosen_first": order[0] if order else None, "two_meanings": two_meanings}


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

    booleans = parse_booleans(values)
    if booleans.notna().mean() >= needed and booleans.nunique() == 2:
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
