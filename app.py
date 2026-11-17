"""A browser app for running the analysis and reading its results page by page.

    python3 -m streamlit run app.py

Pick a dataset in the sidebar, press Run, then move through the pages with the
Back / Next buttons. Nothing here decides anything: it reads the files the
pipeline wrote to output/<dataset>/ and shows them with readable names.
"""

import io
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from autoanalysis import wording
from autoanalysis.config import Config
from autoanalysis.labels import capitalize_first, pretty_column, pretty_table, title_column
from autoanalysis.loading import discover_datasets
from autoanalysis.pipeline import run_pipeline
from autoanalysis.report import MOST_SPECIFIC_FIRST, REJECTION_TITLES, write_index
from autoanalysis.results import chart_figure, list_runs, load_run, load_views

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

PAGES = ["Overview", "Top findings", "Chart explorer", "Rejected patterns", "Your data", "Assumptions"]

KIND_NAMES = {
    "trend_mean": "Trend in the average", "trend_count": "Trend in the number of rows",
    "trend_total": "Trend in the total", "seasonality": "Seasonal cycle", "breakpoint": "Sudden shift",
    "anomaly_mean": "Unusual period (average)", "anomaly_count": "Unusual period (row count)",
    "group_diff": "Difference between groups", "correlation": "Two variables move together",
    "concentration": "Concentrated in a few", "dist_shift": "Distribution changed over time",
}
CONFIDENCE_COLOURS = {"high": "green", "medium": "orange", "low": "gray"}
RULE_NAMES = {  # what each cleaning rule means in everyday words
    "R1 trim whitespace": "Removed stray spaces",
    "R2 missing tokens": "Treated placeholders such as N/A as missing",
    "R3 unparseable number": "Could not read as a number (left missing)",
    "R3 remove currency or unit": "Removed a currency or unit written next to the number",
    "R3 decimal comma": "Read a decimal comma as a decimal point (4,99 -> 4.99)",
    "R3 read yes/no values": "Read different yes/no spellings as yes or no",
    "R4 unparseable date": "Could not read as a date (left missing)",
    "R4 standardise date format": "Read dates written in different formats",
    "R4 ambiguous date order": "Dates where day and month could be swapped - check these",
    "R5 merge case variants": "Merged spellings that differ only by capital letters",
    "R6 drop duplicate rows": "Dropped exact duplicate rows",
    "R6 duplicate rows kept": "Found identical rows but kept them (no row identifier)",
    "R7 flag outliers": "Flagged unusually large or small values (kept)",
    "R8 missing left as missing": "Left missing values missing",
}

st.set_page_config(page_title="Data analysis", layout="wide")


# ---------------------------------------------------------------- loading ----

def run_stamp(name):
    """Changes whenever the analysis is re-run, so cached copies are refreshed."""
    manifest = OUTPUT_DIR / name / "run_manifest.json"
    return manifest.stat().st_mtime_ns if manifest.exists() else None


@st.cache_data(show_spinner=False)
def get_run(name, stamp):
    run = load_run(OUTPUT_DIR / name)
    labels = run["config"].labels
    run["statements"] = [wording.statement(f, labels) for f in run["accepted"]]
    return run


@st.cache_resource(show_spinner="Loading the data behind the charts...")
def get_views(name, stamp):
    return load_views(OUTPUT_DIR / name)


def run_analysis(name, path):
    """Run the pipeline on one dataset, showing each stage as it starts."""
    warnings.filterwarnings("ignore", message="Precision loss occurred")
    with st.status(f"Analysing {name}...", expanded=True) as status:
        run_pipeline(path, OUTPUT_DIR / name, Config(), progress=lambda message: status.write(message))
        write_index(OUTPUT_DIR)
        status.update(label=f"Finished {name}", state="complete", expanded=False)
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.slide = 0
    st.session_state.page = PAGES[0]


# ------------------------------------------------------------- navigation ----

def go(page):
    st.session_state.page = page


def move_slide(step, last):
    st.session_state.slide = max(0, min(last, st.session_state.slide + step))


def page_buttons(current):
    """Back / Next between pages."""
    position = PAGES.index(current)
    back, _, forward = st.columns([1, 4, 1])
    if position > 0:
        back.button(f"← {PAGES[position - 1]}", on_click=go, args=(PAGES[position - 1],), key=f"back_{current}")
    if position < len(PAGES) - 1:
        forward.button(f"{PAGES[position + 1]} →", on_click=go, args=(PAGES[position + 1],),
                       key=f"next_{current}", type="primary")


# --------------------------------------------------------------- helpers -----

def effect_in_words(finding, config):
    """'moderate' or 'large', relative to the thresholds in the config."""
    smallest, large = config.effect_thresholds[finding["kind"]]
    size = abs(finding["result"]["effect_size"])
    return "large" if size >= large else "moderate", smallest, large


def show_chart(finding, run, key):
    views = get_views(run["name"], run_stamp(run["name"]))
    fig = chart_figure(finding, views, run["config"])
    st.pyplot(fig, width="stretch")
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    st.download_button("Download chart (PNG)", png.getvalue(), file_name=f"finding_{finding['rank']}.png",
                       mime="image/png", key=f"download_{key}")


def show_details(finding, run):
    """Everything behind a finding, tucked away until asked for."""
    with st.expander("How sure are we? The checks behind this finding"):
        for line in wording.evidence(finding):
            st.markdown(line)
    st.markdown("**What would confirm it:** " + wording.what_would_confirm(finding))


def show_finding(finding, statement, run, key):
    chart_column, text_column = st.columns([3, 2], gap="large")
    with chart_column:
        show_chart(finding, run, key)
    with text_column:
        st.subheader(statement)
        st.badge(f"{finding['confidence'].capitalize()} confidence",
                 color=CONFIDENCE_COLOURS[finding["confidence"]])
        st.caption(finding["confidence_reason"])
        size, smallest, large = effect_in_words(finding, run["config"])
        result = finding["result"]
        st.markdown(f"**{KIND_NAMES[finding['kind']]}** · {size} effect "
                    f"({result['effect_name']} {abs(result['effect_size']):.2f}; "
                    f"worth reporting from {smallest:g}, large from {large:g})")
        show_details(finding, run)


# ------------------------------------------------------------------ pages ----

def page_overview(run):
    profile, manifest = run["profile"], run["manifest"]["counts"]
    st.header("Overview")
    columns = st.columns(5)
    columns[0].metric("Tables", len(profile))
    columns[1].metric("Rows", f"{sum(t['rows'] for t in profile.values()):,}")
    columns[2].metric("Patterns tested", f"{manifest['candidates']:,}")
    columns[3].metric("Passed every check", f"{manifest['accepted']:,}")
    columns[4].metric("Rejected", f"{manifest['rejected']:,}")
    st.info("Each pattern had to survive a correction for testing many things at once, a minimum size, and "
            "re-tests without outliers and on each half of the data. Nothing here is certain, and none of it "
            "shows cause and effect.")
    top = [(f, s) for f, s in zip(run["accepted"], run["statements"]) if f["in_top"]][:3]
    if manifest["candidates"] == 0:
        usable = [title_column(column, run["config"].labels, table)
                  for table, info in profile.items() for column, p in info["column_profiles"].items()
                  if p["type"] in ("numeric", "datetime", "boolean")]
        st.error("No patterns could be tested. "
                 + ("No column could be read as numbers, dates or yes/no values, so there was nothing to "
                    "measure. " if not usable else
                    f"The only measurable columns ({', '.join(usable)}) had nothing to be compared with. ")
                 + "Check the types on the Your data page. If a column was read wrongly, set its type with "
                   "`type_overrides` in a config file.")
        st.button("Go to Your data →", on_click=go, args=("Your data",))
    elif not top:
        st.warning("No pattern passed every check. With this data there is not enough evidence for any of "
                   "the patterns tested.")
    else:
        st.subheader("Strongest findings")
        for finding, statement in top:
            with st.container(border=True):
                st.markdown(f"**{finding['rank']}.** {statement}")
                st.badge(f"{finding['confidence'].capitalize()} confidence",
                         color=CONFIDENCE_COLOURS[finding["confidence"]])
        st.button("See all top findings →", on_click=go, args=("Top findings",))
    st.caption("High / medium / low confidence reflects how well a pattern held up in the re-tests, "
               "not how important it is.")


def page_top_findings(run):
    st.header("Top findings")
    top = [(f, s) for f, s in zip(run["accepted"], run["statements"]) if f["in_top"]]
    if not top:
        st.warning("No patterns could be tested; see the Overview page." if not run["manifest"]["counts"]["candidates"]
                   else "Nothing passed every check, so there are no findings to show.")
        return
    last = len(top) - 1
    st.session_state.slide = min(st.session_state.get("slide", 0), last)
    back, middle, forward = st.columns([1, 4, 1])
    back.button("← Previous", on_click=move_slide, args=(-1, last), disabled=st.session_state.slide == 0)
    middle.markdown(f"<div style='text-align:center'>Finding {st.session_state.slide + 1} of {len(top)}</div>",
                    unsafe_allow_html=True)
    forward.button("Next →", on_click=move_slide, args=(1, last), disabled=st.session_state.slide == last,
                   type="primary")
    finding, statement = top[st.session_state.slide]
    show_finding(finding, statement, run, key=f"slide_{st.session_state.slide}")
    if len(run["accepted"]) > len(top):
        st.caption(f"{len(run['accepted']) - len(top)} more patterns passed. Find them in the Chart explorer.")


def page_chart_explorer(run):
    st.header("Chart explorer")
    st.write("Every pattern that passed, not just the top ones. Pick one to see its chart.")
    items = list(zip(run["accepted"], run["statements"]))
    if not items:
        st.warning("Nothing passed every check.")
        return
    left, middle, right = st.columns(3)
    kinds = left.multiselect("Kind of pattern", sorted({f["kind"] for f, _ in items}),
                             format_func=KIND_NAMES.get)
    confidence = middle.multiselect("Confidence", ["high", "medium", "low"])
    text = right.text_input("Contains the word")
    shown = [(f, s) for f, s in items
             if (not kinds or f["kind"] in kinds) and (not confidence or f["confidence"] in confidence)
             and text.lower() in s.lower()]
    if not shown:
        st.info("No pattern matches these filters.")
        return
    choice = st.selectbox(f"{len(shown)} patterns", range(len(shown)),
                          format_func=lambda i: f"{shown[i][0]['rank']}. {shown[i][1][:120]}")
    finding, statement = shown[choice]
    show_finding(finding, statement, run, key=f"explorer_{finding['rank']}")


def page_rejected(run):
    st.header("Rejected patterns")
    st.write("These were tested and thrown out. Looking at them shows what the checks are protecting you from.")
    config = run["config"]
    only_promising = st.toggle("Only those that looked promising before the checks", value=True)
    by_reason = {}
    for candidate in run["rejected"]:
        result = candidate["result"]
        if only_promising and not (result["p_value"] < config.alpha
                                   or result["effect_size"] >= config.effect_thresholds[candidate["kind"]][0]):
            continue
        codes = {r["code"] for r in candidate["rejections"]}
        main = next(code for code in MOST_SPECIFIC_FIRST if code in codes)
        by_reason.setdefault(main, []).append(candidate)
    if not by_reason:
        st.info("Nothing to show.")
        return
    per_group = st.slider("Show up to this many per reason", 5, 50, 10)
    for code in MOST_SPECIFIC_FIRST:
        group = by_reason.get(code)
        if not group:
            continue
        with st.expander(f"{REJECTION_TITLES[code]} ({len(group)})"):
            for candidate in group[:per_group]:
                try:
                    what = wording.statement(candidate, config.labels)
                except (KeyError, TypeError, ValueError):
                    what = candidate["id"]
                st.markdown(f"**{what}**")
                st.caption(" ".join(r["explanation"] for r in candidate["rejections"]))
            if len(group) > per_group:
                st.caption(f"...and {len(group) - per_group} more. Raise the slider to see them.")


def page_data(run):
    st.header("Your data")
    labels = run["config"].labels
    for table, info in run["profile"].items():
        st.subheader(f"{capitalize_first(pretty_table(table, labels))} · {info['rows']:,} rows")
        rows = []
        for column, p in info["column_profiles"].items():
            if "mean" in p:
                detail = f"{wording.number(p['min'])} to {wording.number(p['max'])}"
            elif p["type"] == "datetime" and "min" in p:
                detail = f"{str(p['min'])[:10]} to {str(p['max'])[:10]}"
            elif "top_values" in p:
                detail = ", ".join(list(p["top_values"])[:3])
            else:
                detail = ""
            rows.append({"Column": title_column(column, labels, table), "Type": p["type"],
                         "Missing": f"{p['missing_rate']:.0%}", "Distinct values": p["distinct"],
                         "Range or examples": detail})
        st.dataframe(rows, hide_index=True, width="stretch")
    for rel in run["relationships"]:
        st.write(f"**{pretty_column(rel['child_column'], labels)}** in {pretty_table(rel['child_table'], labels)} "
                 f"refers to {pretty_column(rel['parent_column'], labels)} in "
                 f"{pretty_table(rel['parent_table'], labels)} ({rel['containment']:.0%} of values match).")
    st.subheader("What was cleaned")
    if not run["cleaning_log"]:
        st.write("Nothing needed changing.")
    else:
        st.dataframe([{"Table": pretty_table(e["table"], labels),
                       "Column": title_column(e["column"], labels, e["table"]) if e["column"] else "(whole row)",
                       "What happened": RULE_NAMES.get(e["rule"], e["rule"]),
                       "Detail": e["detail"],
                       "Rows affected": e["rows_affected"]} for e in run["cleaning_log"]],
                     hide_index=True, width="stretch")
        st.caption("Raw files are never changed. Full detail is in 02_cleaning_log.json.")


def page_assumptions(run):
    st.header("Assumptions and limits")
    st.write("Check these before acting on any finding.")
    for item in run["assumptions"]:
        st.markdown(f"- {item}")
    if run["skipped"]:
        st.subheader("Not tested: data too thin")
        st.write(f"{len(run['skipped'])} patterns could not be tested because there was too little data.")


PAGE_FUNCTIONS = {
    "Overview": page_overview, "Top findings": page_top_findings, "Chart explorer": page_chart_explorer,
    "Rejected patterns": page_rejected, "Your data": page_data, "Assumptions": page_assumptions,
}


# ------------------------------------------------------------------- main ----

def main():
    st.session_state.setdefault("page", PAGES[0])
    st.session_state.setdefault("slide", 0)
    try:
        datasets = dict(discover_datasets(DATA_DIR))
    except FileNotFoundError:
        datasets = {}
    names = sorted(set(datasets) | set(list_runs(OUTPUT_DIR)))

    with st.sidebar:
        st.title("Data analysis")
        if not names:
            st.warning("Put CSV files in the data folder, then reload this page.")
            st.stop()
        name = st.selectbox("Dataset", names)
        if name in datasets:
            label = "Re-run analysis" if run_stamp(name) else "Run analysis"
            if st.button(label, type="primary", width="stretch"):
                run_analysis(name, datasets[name])
        else:
            st.caption("This dataset is no longer in the data folder; showing its saved results.")

    if run_stamp(name) is None:
        st.info(f"{name} has not been analysed yet. Press Run analysis in the sidebar.")
        return

    run = get_run(name, run_stamp(name))
    st.segmented_control("Page", PAGES, key="page", label_visibility="collapsed", required=True)
    current = st.session_state.page
    PAGE_FUNCTIONS[current](run)
    st.divider()
    page_buttons(current)


main()
