"""Enumerate candidate findings: every sensible column combination, every pattern category.

Nothing is judged here. Each candidate is measured once on the full (cleaned)
data; the filters decide what survives.
"""

from . import measures, timeseries

CATEGORIES = {
    "trend_mean": "Trend over time",
    "trend_count": "Trend over time",
    "trend_total": "Trend over time",
    "seasonality": "Seasonality",
    "breakpoint": "Breakpoint",
    "anomaly_mean": "Anomaly",
    "anomaly_count": "Anomaly",
    "group_diff": "Difference between groups",
    "correlation": "Relationship between variables",
    "concentration": "Concentration",
    "dist_shift": "Distribution shift",
}


def make_candidate(kind, table, **columns):
    columns = {k: v for k, v in columns.items() if v is not None}
    name_parts = [str(columns[k]) for k in ("entity", "group", "time", "x", "y", "value", "cycle", "period")
                  if k in columns]
    return {"id": "|".join([kind, table] + name_parts), "kind": kind,
            "category": CATEGORIES[kind], "table": table, "columns": columns}


def columns_of_type(types, wanted, only=None):
    return [c for c, t in types.items() if t in wanted and (only is None or c in only)]


def enumerate_candidates(table, view, types, joined, entity_columns, config, notes):
    """List every candidate for one table view (not yet measured)."""
    native = set(view.columns) - set(joined)
    numeric = columns_of_type(types, {"numeric"}, native)
    yes_no = columns_of_type(types, {"boolean"}, native)
    joined_numeric = columns_of_type(types, {"numeric"}, joined)
    categorical = columns_of_type(types, {"categorical"})          # native and joined
    time_columns = columns_of_type(types, {"datetime"}, native)
    non_negative = [c for c in numeric if (view[c].dropna() >= 0).all()]

    candidates = []

    # A table with one record per time stamp is a set of measurements over
    # time, not a stream of events. Its row count only reflects the calendar,
    # and adding its values across time stamps (a "total exchange rate") has
    # no meaning, so counts, totals and concentration are not tested there.
    single_series = bool(time_columns) and timeseries.is_single_series(view, time_columns[0])
    if single_series:
        notes.append(f"`{table}` has one record per {time_columns[0]}, so it was treated as measurements "
                     "over time: row counts, totals and concentration were not tested on it.")
        non_negative, entity_columns = [], []

    # --- Patterns over time ------------------------------------------------
    for time in time_columns:
        freq = measures.choose_frequency(view[time], config)
        if freq is None:
            notes.append(f"{table}.{time}: fewer than {config.min_periods} distinct days, "
                         "so no time-based patterns were tested on it.")
            continue
        if not single_series:
            candidates.append(make_candidate("trend_count", table, time=time, freq=freq))
            for period in measures.find_anomalies(view, {"time": time, "freq": freq}, config, "count"):
                candidates.append(make_candidate("anomaly_count", table, time=time, freq=freq, period=period))
        for value in numeric + yes_no:
            base = {"time": time, "value": value, "freq": freq}
            candidates.append(make_candidate("trend_mean", table, **base))
            candidates.append(make_candidate("breakpoint", table, **base))
            for period in measures.find_anomalies(view, base, config, "mean"):
                candidates.append(make_candidate("anomaly_mean", table, period=period, **base))
        for value in numeric:
            candidates.append(make_candidate("dist_shift", table, time=time, value=value))
            for cycle in ("month_of_year", "day_of_week"):
                candidates.append(make_candidate("seasonality", table, time=time, value=value, cycle=cycle))
        for value in non_negative:
            candidates.append(make_candidate("trend_total", table, time=time, value=value, freq=freq))

    # --- Differences between groups ----------------------------------------
    # The measured value must belong to this table; the grouping may come
    # from a joined parent table (e.g. order amount by customer segment).
    # In that case the comparison is made per parent (one value per customer),
    # because several records from the same parent are not independent.
    for group in categorical:
        for value in numeric + yes_no:
            candidates.append(make_candidate("group_diff", table, group=group, value=value,
                                             unit=joined.get(group)))

    # --- Relationships between numeric variables ---------------------------
    # At least one side must be native, otherwise the same parent-table pair
    # would be re-tested on duplicated rows.
    for i, x in enumerate(numeric):
        for y in numeric[i + 1:]:
            candidates.append(make_candidate("correlation", table, x=x, y=y))
        for y in joined_numeric:
            candidates.append(make_candidate("correlation", table, x=x, y=y, unit=joined[y]))

    # --- Concentration -----------------------------------------------------
    for value in non_negative:
        candidates.append(make_candidate("concentration", table, value=value))
    for entity in entity_columns:
        candidates.append(make_candidate("concentration", table, entity=entity))
        for value in non_negative:
            candidates.append(make_candidate("concentration", table, entity=entity, value=value))
    return candidates


def find_entity_columns(table, view, types, relationships):
    """Columns that say which entity a record belongs to (foreign keys, repeated identifiers)."""
    entities = {r["child_column"] for r in relationships if r["child_table"] == table}
    for column, column_type in types.items():
        if column_type == "identifier" and column in view and view[column].duplicated().any():
            entities.add(column)
    return sorted(entities)


def search_all(views, view_types, joined_columns, relationships, config):
    """Return (candidates, skipped, notes).

    candidates: measured successfully on the full data (each has a "result")
    skipped:    could not be tested because the data was too thin, with the reason
    """
    candidates, skipped, notes = [], [], []
    for table in sorted(views):
        view, types = views[table], view_types[table]
        entities = find_entity_columns(table, view, types, relationships)
        for candidate in enumerate_candidates(table, view, types, joined_columns[table],
                                              entities, config, notes):
            result = measures.measure(candidate, view, config)
            if result["ok"]:
                candidate["result"] = result
                candidates.append(candidate)
            else:
                candidate["reason"] = result["reason"]
                skipped.append(candidate)
    return candidates, skipped, notes
