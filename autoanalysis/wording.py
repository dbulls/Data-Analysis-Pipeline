"""Plain-language text for findings: the statement, the evidence, and what would confirm it.

Names of columns and tables are shown through labels.py, so sentences say
"study hours per week", not `study_hours_per_week`.
"""

from .labels import CHANGE_SUFFIX, capitalize_first, pretty_column, pretty_table


def number(value):
    """Format a number for reading, not for precision."""
    if value is None:
        return "n/a"
    size = abs(value)
    if size >= 1000:
        return f"{value:,.0f}"
    if size >= 10:
        return f"{value:.1f}"
    if size >= 1:
        return f"{value:.2f}"
    return f"{value:.3g}"


def percent(value):
    return "n/a" if value is None else f"{abs(value):.0f}%"


def change_statement(candidate, labels=None):
    """Wording for a series of % changes (see timeseries.py): talk in percentage points."""
    kind, cols, table = candidate["kind"], candidate["columns"], candidate["table"]
    s = candidate["result"]["summary"]
    up = candidate["result"]["direction"] == 1

    def label(column):
        return pretty_column(column.removesuffix(CHANGE_SUFFIX), labels, table)

    name = label(cols.get("value", ""))
    if kind == "anomaly_mean":
        return (f"{name} moved unusually in {s['period_label']}: average change per record {s['value']:+.2f}% "
                f"vs. about {s['baseline']:+.2f}% in surrounding {s['period']}s.")
    if kind == "trend_mean":
        return (f"The average change per record in {name} drifted {'up' if up else 'down'} between "
                f"{s['first_period']} and {s['last_period']} (from about {s['fitted_start']:+.3f}% to "
                f"{s['fitted_end']:+.3f}%).")
    if kind == "breakpoint":
        return (f"The average change per record in {name} shifted {'up' if up else 'down'} around "
                f"{s['break_period']} (from {s['mean_before']:+.3f}% to {s['mean_after']:+.3f}%).")
    time = pretty_column(cols.get("time", ""), labels, table)
    if kind == "dist_shift":
        ratio = s["std_second_half"] / s["std_first_half"] if s["std_first_half"] else None
        sizes = f"(typical size {s['std_first_half']:.2f}% vs. {s['std_second_half']:.2f}%)"
        if ratio is None or 0.85 <= ratio <= 1.15:
            return (f"The pattern of moves in {name} changed between the first and second half of the "
                    f"{time} range, though their typical size barely did {sizes}.")
        return (f"Moves in {name} became {'larger' if ratio > 1 else 'smaller'} in the second half of the "
                f"{time} range {sizes}.")
    if kind == "seasonality":
        means = s["bucket_means"]
        cycle = "month of year" if cols["cycle"] == "month_of_year" else "day of week"
        return (f"Changes in {name} differ by {cycle}: on average {means[s['peak']]:+.3f}% per record in "
                f"{s['peak']} vs. {means[s['trough']]:+.3f}% in {s['trough']}.")
    if kind == "correlation":
        return (f"{label(cols['x'])} and {label(cols['y'])} tend to move "
                f"{'together' if up else 'in opposite directions'} from one record "
                f"to the next (rank correlation of their changes {s['rho']:+.2f}).")
    return None


def statement(candidate, labels=None):
    """One sentence saying what was found. Also used as the chart title."""
    return capitalize_first(raw_statement(candidate, labels))


def raw_statement(candidate, labels=None):
    kind, cols, raw_table = candidate["kind"], candidate["columns"], candidate["table"]
    if any(str(v).endswith(CHANGE_SUFFIX) for v in cols.values()):
        special = change_statement(candidate, labels)
        if special:
            return special
    s = candidate["result"]["summary"]
    table = pretty_table(raw_table, labels)

    def label(key):
        return pretty_column(cols[key], labels, raw_table) if cols.get(key) else None

    value, group, time = label("value"), label("group"), label("time")
    up = candidate["result"]["direction"] == 1

    if kind in ("trend_mean", "trend_count", "trend_total"):
        subject = {"trend_mean": f"Average {value}",
                   "trend_count": f"The number of rows in {table}",
                   "trend_total": f"Total {value}"}[kind]
        return (f"{subject} per {s['period']} {'rose' if up else 'fell'} by about "
                f"{percent(s['percent_change_over_range'])} between {s['first_period']} and "
                f"{s['last_period']} (from about {number(s['fitted_start'])} to {number(s['fitted_end'])}).")
    if kind == "seasonality":
        cycle = "yearly" if cols["cycle"] == "month_of_year" else "weekly"
        return (f"{value} follows a {cycle} cycle: highest in {s['peak']}, lowest in {s['trough']}, "
                f"a swing of about {percent(s['range_percent_of_mean'])} of its average.")
    if kind == "breakpoint":
        return (f"Average {value} {'stepped up' if up else 'dropped'} by about {percent(s['percent_change'])} "
                f"around {s['break_period']} (from {number(s['mean_before'])} to {number(s['mean_after'])}) "
                "and stayed at the new level.")
    if kind in ("anomaly_mean", "anomaly_count"):
        subject = f"Average {value}" if kind == "anomaly_mean" else f"The number of rows in {table}"
        return (f"{subject} in {s['period_label']} was {percent(s['percent_from_baseline'])} "
                f"{'above' if up else 'below'} its usual level ({number(s['value'])} vs. about "
                f"{number(s['baseline'])}).")
    if kind == "group_diff":
        groups = s["groups"]
        what = "rate" if s["typical_is"] == "rate" else "median"
        return (f"{value} differs by {group}: highest for {s['highest']} "
                f"({what} {number(groups[s['highest']]['typical'])}), lowest for {s['lowest']} "
                f"({what} {number(groups[s['lowest']]['typical'])}), a gap of "
                f"{percent(s['percent_gap_highest_vs_lowest'])}"
                + (f" (compared per {pretty_column(cols['unit'], labels, raw_table)})." if cols.get("unit") else "."))
    if kind == "correlation":
        return (f"Higher {label('x')} tends to go with {'higher' if up else 'lower'} {label('y')} "
                f"(rank correlation {s['rho']:+.2f}).")
    if kind == "concentration":
        share = f"{s['top_20_percent_share']:.0%}"
        entity = label("entity")
        if entity and value:
            return f"The top 20% of {entity} values account for {share} of total {value}."
        if entity:
            return f"The top 20% of {entity} values account for {share} of all rows in {table}."
        return f"The largest 20% of rows in {table} account for {share} of total {value}."
    if kind == "dist_shift":
        return (f"The distribution of {value} shifted {'up' if up else 'down'} between the first and second "
                f"half of the {time} range (median {number(s['median_first_half'])} to "
                f"{number(s['median_second_half'])}).")
    raise ValueError(kind)


def evidence(candidate):
    """Bullet points: the test, the numbers, and each robustness check."""
    result = candidate["result"]
    lines = [
        f"**Test:** {result['test']}.",
        f"**Effect size:** {result['effect_name']} = {result['effect_size']:.3g}.",
        f"**Significance:** raw p = {result['p_value']:.3g}; adjusted for "
        f"{candidate['tests_in_family']} tests, q = {candidate['q_value']:.3g}.",
        f"**Data used:** n = {result['n']:,} "
        f"({'time periods' if candidate['kind'].startswith(('trend', 'breakpoint')) else 'records or units'}).",
    ]
    names = {"outliers_removed": "Outliers removed", "single_record_removed": "Single record removed",
             "split_halves": "Split in half"}
    for key, label in names.items():
        check = candidate["checks"][key]
        lines.append(f"**{label}:** {check['status'].replace('_', ' ')}. {check['detail']}")
    return lines


def what_would_confirm(candidate):
    """What a person should check before relying on the finding."""
    kind, cols = candidate["kind"], candidate["columns"]
    notes = {
        "trend_mean": "More history, and a check that how this value is recorded did not change over the "
                      "period. The trend test treats periods as independent, which overstates certainty "
                      "when neighbouring periods move together.",
        "trend_count": "Confirm that data collection covered the whole period evenly (no late-arriving or "
                       "back-filled records) before reading this as real growth or decline.",
        "trend_total": "Check whether the driver is more records or bigger records; the pipeline found "
                       "neither explains it alone.",
        "seasonality": "More complete cycles. A cycle seen two or three times can still be coincidence; "
                       "ask whether there is a known reason for the peak.",
        "breakpoint": "Find out what happened at the break date (process change, pricing, system migration, "
                      "definition change). Without a known cause it may be a recording artefact.",
        "anomaly_mean": "Look at the individual records in that period and ask what was different. "
                        "The significance figure relies on a normal approximation.",
        "anomaly_count": "Check for missing or duplicated loads in that period before treating it as a real event.",
        "group_diff": "Check that the groups are comparable in other respects (mix, time period, size). "
                      "This is an observed difference, not proof that the grouping causes it.",
        "correlation": "Correlation is not causation. A third factor may drive both; a controlled comparison "
                       "or an experiment is needed to claim one affects the other.",
        "concentration": "Check that the same entity is not split across several identifiers (or several "
                         "entities merged under one), which would distort the shares.",
        "dist_shift": "Check for a change in how the value is measured or which records are included "
                      "between the two halves.",
    }
    text = notes[kind]
    if cols.get("unit"):
        text += (f" This was tested with one value per {cols['unit']} (its median, or its rate for yes/no "
                 "values), because several records from the same parent are not independent.")
    return text
