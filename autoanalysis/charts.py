"""One chart per finding. The kind of finding decides the kind of chart:

    trends, breakpoints, anomalies   line chart over time
    seasonality                      dot-and-line by calendar bucket
    group differences                dot plot with 95% intervals
    relationships                    scatter plot with fitted line
    concentration                    cumulative share curve
    distribution shift               overlaid histograms

Each chart is titled with the finding stated as a sentence.
"""

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # no display needed; identical output everywhere
import matplotlib.pyplot as plt
import numpy as np

from . import measures
from .wording import statement

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#898781"


def axis_label(table, column, config, prefix=""):
    """Column name plus its unit. Units are not in the data, so they come from config."""
    unit = config.units.get(f"{table}.{column}") or config.units.get(column)
    label = f"{prefix}{column} ({unit})" if unit else f"{prefix}{column} (units as recorded in source)"
    return "\n".join(textwrap.wrap(label, 55))


def new_figure(title):
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    # Title sits on the figure, not the axes, so long axis labels cannot push it off the edge.
    fig.suptitle("\n".join(textwrap.wrap(title, 76)), x=0.02, y=0.98, ha="left", va="top",
                 fontsize=12, fontweight="bold", color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    return fig, ax


def legend(ax):
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)


def time_series(candidate, df, config):
    """Period table for a time-based finding, with real dates on the x axis."""
    kind, cols = candidate["kind"], candidate["columns"]
    if kind in ("trend_count", "anomaly_count"):
        table = measures.count_or_total_series(df, cols, "count")
    elif kind == "trend_total":
        table = measures.count_or_total_series(df, cols, "sum")
    else:
        table = measures.mean_series(df, cols, config, False).rename(columns={"mean": "y"})
    table = table.copy()
    table["date"] = table.index.start_time
    return table


def time_axis_labels(ax, candidate, config):
    kind, cols, table = candidate["kind"], candidate["columns"], candidate["table"]
    period = measures.FREQUENCY_NAMES[cols["freq"]]
    ax.set_xlabel(f"{cols['time']} ({period})")
    if kind in ("trend_count", "anomaly_count"):
        ax.set_ylabel(f"rows in {table} per {period} (count"
                      + (", scaled to an average-length month)" if cols["freq"] == "M" else ")"))
    elif kind == "trend_total":
        ax.set_ylabel(axis_label(table, cols["value"], config, f"total per {period}: "))
    else:
        ax.set_ylabel(axis_label(table, cols["value"], config, f"mean per {period}: "))


def draw_trend(ax, candidate, df, config):
    series, s = time_series(candidate, df, config), candidate["result"]["summary"]
    ax.plot(series["date"], series["y"], color=BLUE, linewidth=2, marker="o", markersize=4,
            label=f"per {s['period']}")
    slope = s["slope_per_period"]
    fitted = s["fitted_start"] + slope * (series["position"] - series["position"].iloc[0])
    ax.plot(series["date"], fitted, color=ORANGE, linewidth=2, linestyle="--", label="fitted trend (Theil-Sen)")
    keep_extremes_from_flattening(ax, series["y"], fitted)
    time_axis_labels(ax, candidate, config)
    legend(ax)


def keep_extremes_from_flattening(ax, y, fitted):
    """If a few periods are wildly off, zoom to the rest and say how many are off the scale."""
    spread = 1.4826 * (y - fitted).abs().median()
    if not spread > 0:
        return
    low, high = fitted.min() - 8 * spread, fitted.max() + 8 * spread
    off_scale = int(((y < low) | (y > high)).sum())
    if off_scale:
        ax.set_ylim(low, high)
        ax.annotate(f"{off_scale} period(s) off the scale (most extreme: {y[(y - fitted).abs().idxmax()]:,.0f})",
                    (0.5, 1.02), xycoords="axes fraction", ha="center", fontsize=9, color=MUTED)


def draw_breakpoint(ax, candidate, df, config):
    series, s = time_series(candidate, df, config), candidate["result"]["summary"]
    ax.plot(series["date"], series["y"], color=BLUE, linewidth=2, marker="o", markersize=4,
            label=f"per {s['period']}")
    before = series["date"] < s["break_date"]
    ax.plot(series["date"][before], [s["mean_before"]] * before.sum(), color=ORANGE, linewidth=2,
            linestyle="--", label="average before / after")
    ax.plot(series["date"][~before], [s["mean_after"]] * (~before).sum(), color=ORANGE, linewidth=2, linestyle="--")
    ax.axvline(s["break_date"], color=NEUTRAL, linewidth=1, linestyle=":")
    ax.annotate(f"break: {s['break_period']}", (s["break_date"], ax.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", fontsize=9, color=MUTED)
    time_axis_labels(ax, candidate, config)
    legend(ax)


def draw_anomaly(ax, candidate, df, config):
    series, s = time_series(candidate, df, config), candidate["result"]["summary"]
    ax.plot(series["date"], series["y"], color=BLUE, linewidth=2, marker="o", markersize=4,
            label=f"per {s['period']}")
    odd = series[[str(p) == s["period_label"] for p in series.index]]
    ax.scatter(odd["date"], odd["y"], s=110, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3,
               label=f"unusual {s['period']}")
    ax.annotate(s["period_label"], (odd["date"].iloc[0], odd["y"].iloc[0]), xytext=(8, 4),
                textcoords="offset points", fontsize=9, color=INK)
    time_axis_labels(ax, candidate, config)
    legend(ax)


def draw_seasonality(ax, candidate, df, config):
    cols, means = candidate["columns"], candidate["result"]["summary"]["bucket_means"]
    ax.plot(list(means), list(means.values()), color=BLUE, linewidth=2, marker="o", markersize=7)
    ax.set_xlabel("month of year" if cols["cycle"] == "month_of_year" else "day of week")
    ax.set_ylabel(axis_label(candidate["table"], cols["value"], config, "mean, trend removed: "))


def draw_group_diff(ax, candidate, df, config):
    cols, s = candidate["columns"], candidate["result"]["summary"]
    groups = s["groups"]
    names = sorted(groups, key=lambda g: groups[g]["typical"])
    labels = [f"{g}  (n={groups[g]['n']:,})" for g in names]
    if s["typical_is"] == "rate":
        # Yes/no value: one dot per group with a 95% interval.
        rates = [groups[g]["mean"] * 100 for g in names]
        errors = [1.96 * groups[g]["standard_error"] * 100 for g in names]
        ax.errorbar(rates, labels, xerr=errors, fmt="o", color=BLUE, elinewidth=2, capsize=4, markersize=8)
        ax.set_xlabel(f"{cols['value']}: share of records that are yes (%, with 95% interval)")
    else:
        data, _ = measures.group_diff_data(df, cols)
        samples = [data.loc[data[cols["group"]].astype(str) == g, cols["value"]] for g in names]
        ax.boxplot(samples, orientation="horizontal", tick_labels=labels, showfliers=False, widths=0.5,
                   patch_artist=True, boxprops={"facecolor": "#cde2fb", "edgecolor": BLUE},
                   medianprops={"color": ORANGE, "linewidth": 2},
                   whiskerprops={"color": BLUE}, capprops={"color": BLUE})
        ax.set_xlabel(axis_label(candidate["table"], cols["value"], config)
                      + "\nbox = middle half, line = median, extreme values not drawn")
    ax.set_ylabel(cols["group"] + (f"\n(one value per {cols['unit']})" if cols.get("unit") else ""))


def draw_correlation(ax, candidate, df, config):
    cols, s = candidate["columns"], candidate["result"]["summary"]
    data = measures.correlation_data(df, cols)
    step = max(1, len(data) // 4000)            # thin very large data, deterministically
    shown = data.iloc[::step]
    ax.scatter(shown[cols["x"]], shown[cols["y"]], s=12, color=BLUE, alpha=0.35, linewidth=0,
               label=(f"one per {cols['unit']}" if cols.get("unit") else "records")
               + (f" (every {step}th shown)" if step > 1 else ""))
    xs = np.array([data[cols["x"]].min(), data[cols["x"]].max()])
    ax.plot(xs, s["line_slope"] * xs + s["line_intercept"], color=ORANGE, linewidth=2,
            label="least-squares line")
    ax.set_xlabel(axis_label(candidate["table"], cols["x"], config))
    ax.set_ylabel(axis_label(candidate["table"], cols["y"], config))
    legend(ax)


def draw_concentration(ax, candidate, df, config):
    cols, s = candidate["columns"], candidate["result"]["summary"]
    values = np.sort(measures.concentration_units(df, cols).to_numpy(float))[::-1]
    unit_share = np.arange(0, len(values) + 1) / len(values) * 100
    total_share = np.concatenate([[0], np.cumsum(values)]) / values.sum() * 100
    ax.plot(unit_share, total_share, color=BLUE, linewidth=2, label="actual")
    ax.plot([0, 100], [0, 100], color=NEUTRAL, linewidth=1.5, linestyle="--", label="if all were equal")
    top = s["top_20_percent_share"] * 100
    ax.scatter([20], [top], s=90, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(f"top 20% hold {top:.0f}%", (20, top), xytext=(10, -14), textcoords="offset points",
                fontsize=10, color=INK)
    unit = cols.get("entity", f"{candidate['table']} records")
    what = cols.get("value", f"{candidate['table']} records")
    ax.set_xlabel(f"{unit}, largest first (cumulative % of units)")
    ax.set_ylabel(f"cumulative share of total {what} (%)")
    legend(ax)


def draw_dist_shift(ax, candidate, df, config):
    cols, s = candidate["columns"], candidate["result"]["summary"]
    data = df[[cols["time"], cols["value"]]].dropna()
    first = data.loc[data[cols["time"]] < s["midpoint"], cols["value"]]
    second = data.loc[data[cols["time"]] >= s["midpoint"], cols["value"]]
    low, high = data[cols["value"]].quantile([0.005, 0.995])
    bins = np.linspace(low, high, 31)
    ax.hist(first, bins=bins, density=True, histtype="stepfilled", color=BLUE, alpha=0.35,
            edgecolor=BLUE, linewidth=2, label=f"first half (n={len(first):,})")
    ax.hist(second, bins=bins, density=True, histtype="stepfilled", color=ORANGE, alpha=0.35,
            edgecolor=ORANGE, linewidth=2, label=f"second half (n={len(second):,})")
    ax.set_xlabel(axis_label(candidate["table"], cols["value"], config))
    ax.set_ylabel("density (share of records per unit)")
    legend(ax)


DRAWERS = {
    "trend_mean": draw_trend, "trend_count": draw_trend, "trend_total": draw_trend,
    "breakpoint": draw_breakpoint, "anomaly_mean": draw_anomaly, "anomaly_count": draw_anomaly,
    "seasonality": draw_seasonality, "group_diff": draw_group_diff, "correlation": draw_correlation,
    "concentration": draw_concentration, "dist_shift": draw_dist_shift,
}


def draw_chart(candidate, df, config, path):
    fig, ax = new_figure(statement(candidate))
    DRAWERS[candidate["kind"]](ax, candidate, df, config)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)


def draw_all(findings, views, config, charts_dir):
    """Draw a chart for each finding in the top list; store its file name on the finding."""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    for old in charts_dir.glob("*.png"):      # stale charts from an earlier run
        old.unlink()
    for finding in findings:
        if not finding.get("in_top"):
            continue
        safe_id = "".join(ch if ch.isalnum() else "_" for ch in finding["id"])[:80]
        name = f"{finding['rank']:02d}_{safe_id}.png"
        draw_chart(finding, views[finding["table"]], config, charts_dir / name)
        finding["chart"] = f"{charts_dir.name}/{name}"
