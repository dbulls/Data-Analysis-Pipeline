"""Handle "persistent" series: values that mostly carry over from one record to the next.

Prices, exchange rates, balances and temperatures behave like this: today's
value is yesterday's value plus a small move. Neighbouring records are then
far from independent, and the standard tests treat each of them as fresh
evidence. That makes them badly over-confident: two unrelated series that both
drifted look "correlated", and any wiggle looks "significant".

The standard remedy, used here: analyse the record-to-record % changes
instead of the levels. Changes are close to independent, so the usual tests
apply to them. Questions about the level are still answered, just phrased as
changes: "the franc jumped in August 2011" is an unusual month of changes, and
"the euro and the krone move together" is a correlation of their changes.

How a series is recognised as persistent (only in tables with one record per
time stamp, i.e. a single time series rather than a stream of events):
    1. remove each period's mean (so trends and steps do not count)
    2. take the correlation of each remaining value with the one before it
    3. at or above config.persistence_threshold, the series is persistent
"""

import numpy as np
import pandas as pd

from . import measures

CHANGE_TABLE_SUFFIX = "__changes"
CHANGE_COLUMN_SUFFIX = "_pct_change"
CHANGE_UNIT = "% change from previous record"


def is_single_series(df, time_col):
    """True if (almost) every record has its own time stamp."""
    times = df[time_col].dropna()
    return len(times) > 0 and times.nunique() >= 0.95 * len(times)


def persistence(df, time_col, value_col, freq):
    """Lag-1 autocorrelation of records after removing each period's mean."""
    data = df[[time_col, value_col]].dropna().sort_values(time_col, kind="stable")
    if len(data) < 20 or data[value_col].nunique() < 3:
        return None
    period = data[time_col].dt.to_period(freq)
    within = data[value_col] - data.groupby(period)[value_col].transform("mean")
    if within.std() == 0:
        return None
    return float(within.autocorr())


def percent_changes(df, time_col, value_col):
    """% change from the previous non-missing record, aligned to df's rows.

    A change spanning a gap more than 3x the usual spacing is left missing,
    because it is not comparable with ordinary record-to-record moves.
    """
    data = df[[time_col, value_col]].dropna().sort_values(time_col, kind="stable")
    values = data[value_col]
    if (values > 0).all():
        change = values.pct_change(fill_method=None) * 100
    else:
        change = values.diff()                      # % change is undefined around zero
    gaps = data[time_col].diff()
    change[gaps > 3 * gaps.median()] = np.nan
    return change.reindex(df.index)


def add_change_tables(views, view_types, joined_columns, config):
    """For each table with persistent series, add a derived table of their % changes.

    The persistent columns get the type "persistent_level" in the original
    table, so the search skips them there. Returns (notes, persistent) where
    persistent is {table: {column: autocorrelation}}.
    """
    notes, persistent = [], {}
    for table in list(views):
        view, types = views[table], view_types[table]
        time_columns = [c for c, t in types.items() if t == "datetime" and c not in joined_columns[table]]
        if not time_columns or not is_single_series(view, time_columns[0]):
            continue
        time_col = time_columns[0]
        freq = measures.choose_frequency(view[time_col], config)
        if freq is None:
            continue

        found = {}
        for column, column_type in types.items():
            if column_type != "numeric" or column in joined_columns[table]:
                continue
            score = persistence(view, time_col, column, freq)
            if score is not None and score >= config.persistence_threshold:
                found[column] = score
        if not found:
            continue

        name = table + CHANGE_TABLE_SUFFIX
        changes = pd.DataFrame({time_col: view[time_col]}, index=view.index)
        for column in found:
            changes[column + CHANGE_COLUMN_SUFFIX] = percent_changes(view, time_col, column)
            types[column] = "persistent_level"
        views[name] = changes
        view_types[name] = {time_col: "datetime", **{c: "numeric" for c in changes.columns if c != time_col}}
        joined_columns[name] = {}
        for column in changes.columns[1:]:
            config.units.setdefault(f"{name}.{column}", CHANGE_UNIT)

        persistent[table] = found
        notes.append(
            f"{len(found)} column(s) in `{table}` behave like levels that carry over from one record to the "
            f"next (e.g. prices or rates; within-period autocorrelation >= {config.persistence_threshold}). "
            f"Tests on such levels are badly over-confident, so they were analysed as % changes from the "
            f"previous record, in the derived table `{name}`. Their levels are profiled but not tested.")
    return notes, persistent
