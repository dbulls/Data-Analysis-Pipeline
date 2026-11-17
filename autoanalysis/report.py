"""Findings report in Markdown and as a single self-contained HTML file.

The report is built once as a list of simple blocks, then rendered twice.
Blocks: ("h1"|"h2"|"h3"|"p", text), ("ul", [text]), ("table", header, rows), ("img", path, alt).
Inside text, **bold** and `code` are supported.
"""

import base64
import html
import re
from collections import Counter
from pathlib import Path

from . import wording

REJECTION_TITLES = {
    "not_significant": "Could be chance (failed multiple-comparison correction)",
    "effect_too_small": "Statistically detectable but too small to matter",
    "driven_by_outliers": "Driven by outliers",
    "driven_by_single_record": "Driven by a single record",
    "does_not_replicate": "Did not hold in both halves of the data",
    "data_quality": "Data-quality artefact",
    "better_explained_by_trend": "Step change better explained by a gradual trend",
    "trivially_expected": "Trivially expected",
    "redundant": "Duplicate of another finding",
}


MOST_SPECIFIC_FIRST = ["driven_by_single_record", "driven_by_outliers", "data_quality",
                       "trivially_expected", "redundant", "better_explained_by_trend",
                       "does_not_replicate", "effect_too_small", "not_significant"]


def widespread_patterns(run):
    """(table, kind, passed, tested) where at least 10 tests ran and over half passed."""
    tested = Counter((c["table"], c["kind"]) for c in run["candidates"])
    passed = Counter((c["table"], c["kind"]) for c in run["accepted"])
    names = {"correlation": "correlation", "group_diff": "group-difference"}
    return [(table, names[kind], passed[(table, kind)], n) for (table, kind), n in sorted(tested.items())
            if kind in names and n >= 10 and passed[(table, kind)] > n / 2]


def describe_rejected(candidate):
    """Short description of a rejected candidate: what was tested, in neutral words."""
    try:
        return wording.statement(candidate)
    except (KeyError, TypeError, ValueError):
        return candidate["id"]


def build_blocks(run):
    config, accepted, rejected = run["config"], run["accepted"], run["rejected"]
    blocks = [("h1", "Automated findings report")]
    blocks.append(("p", f"Data: `{run['data_dir']}` - {len(run['profile'])} table(s), "
                        f"{sum(t['rows'] for t in run['profile'].values()):,} rows in total. "
                        f"**{len(run['candidates'])}** candidate patterns were tested; "
                        f"**{len(accepted)}** passed every check, **{len(rejected)}** were rejected, and "
                        f"{len(run['skipped'])} could not be tested because the data was too thin."))
    blocks.append(("p", "Nothing here is certain. Each finding carries a confidence level and says what "
                        "would be needed to confirm it. These are patterns in the data as recorded; none "
                        "of them establishes cause and effect."))

    # ---- Top findings -----------------------------------------------------
    top = [f for f in accepted if f["in_top"]]
    rest = [f for f in accepted if not f["in_top"]]
    blocks.append(("h2", f"Top findings ({len(top)} of {len(accepted)} that passed)"))
    blocks.append(("p", f"At most {config.max_per_kind_in_top} findings of each kind are shown here, best first; "
                        "the rest are listed after them."))
    for table, kind, n_passed, n_tested in widespread_patterns(run):
        blocks.append(("p", f"**Note:** {n_passed} of {n_tested} {kind} tests in `{table}` passed. When most "
                            "series move together like this, a common driver is likely (for example, every "
                            "value being measured against the same base, or one market affecting all of them). "
                            "Treat these as one broad pattern rather than many separate discoveries."))
    if not accepted:
        blocks.append(("p", "No pattern passed every check. With this data, the honest conclusion is that "
                            "there is not enough evidence for any of the patterns tested."))
    for finding in top:
        blocks.append(("h3", f"{finding['rank']}. {wording.statement(finding)}"))
        blocks.append(("p", f"**Confidence: {finding['confidence']}.** {finding['confidence_reason']} "
                            f"Category: {finding['category']}. Table: `{finding['table']}`. "
                            f"Ranking score {finding['score']['total']:.2f} "
                            f"(size {finding['score']['size']:.2f}, reliability "
                            f"{finding['score']['reliability']:.2f}, actionability "
                            f"{finding['score']['actionability']:.2f})."))
        if finding.get("chart"):
            blocks.append(("img", finding["chart"], wording.statement(finding)))
        blocks.append(("ul", wording.evidence(finding)))
        blocks.append(("p", "**To confirm:** " + wording.what_would_confirm(finding)))
    if rest:
        blocks.append(("h3", f"Other findings that passed ({len(rest)})"))
        blocks.append(("table", ["Rank", "Finding", "Confidence"],
                       [[f["rank"], wording.statement(f), f["confidence"]] for f in rest]))

    # ---- Rejected ---------------------------------------------------------
    blocks.append(("h2", "Patterns that looked promising but failed the checks"))
    promising = [c for c in rejected
                 if c["result"]["p_value"] < config.alpha
                 or c["result"]["effect_size"] >= config.effect_thresholds[c["kind"]][0]]
    blocks.append(("p", f"{len(promising)} rejected patterns had either a raw p-value below {config.alpha} "
                        "or an effect above the minimum size, so they would have looked like findings "
                        "without the checks. They are grouped by the main reason they were rejected. "
                        f"All {len(rejected)} rejections are in `04_rejected.json`."))
    by_reason = {}
    for candidate in promising:
        # File each one under its most specific reason, not just the first that fired.
        codes = {r["code"] for r in candidate["rejections"]}
        main = next(code for code in MOST_SPECIFIC_FIRST if code in codes)
        by_reason.setdefault(main, []).append(candidate)
    for code in MOST_SPECIFIC_FIRST:
        group = by_reason.get(code)
        if not group:
            continue
        blocks.append(("h3", f"{REJECTION_TITLES[code]} ({len(group)})"))
        shown = group[:8]
        blocks.append(("ul", [f"{describe_rejected(c)} **Why rejected:** "
                              + " ".join(r["explanation"] for r in c["rejections"]) for c in shown]))
        if len(group) > len(shown):
            blocks.append(("p", f"...and {len(group) - len(shown)} more of this kind in `04_rejected.json`."))

    # ---- Not testable -----------------------------------------------------
    blocks.append(("h2", "Not tested: data too thin"))
    if run["skipped"]:
        reasons = Counter(f"{c['category']}: {c['reason']}" for c in run["skipped"])
        blocks.append(("table", ["Reason", "Candidates"], [[r, n] for r, n in sorted(reasons.items())]))
    else:
        blocks.append(("p", "Every candidate had enough data to be tested."))

    # ---- Data profile -----------------------------------------------------
    blocks.append(("h2", "What the data looks like"))
    for table, info in run["profile"].items():
        blocks.append(("h3", f"`{table}` - {info['rows']:,} rows, {info['exact_duplicate_rows']} exact duplicate rows"))
        rows = []
        for column, p in info["column_profiles"].items():
            if "mean" in p:
                detail = f"{wording.number(p['min'])} to {wording.number(p['max'])}, median {wording.number(p['median'])}"
            elif p["type"] == "datetime" and "min" in p:
                detail = f"{str(p['min'])[:10]} to {str(p['max'])[:10]}"
            elif "top_values" in p:
                detail = "most common: " + ", ".join(list(p["top_values"])[:3])
            else:
                detail = ""
            rows.append([column, p["type"], f"{p['missing_rate']:.1%}", p["distinct"], detail])
        blocks.append(("table", ["Column", "Inferred type", "Missing", "Distinct", "Range / examples"], rows))
    if run["relationships"]:
        blocks.append(("h3", "Relationships between tables"))
        blocks.append(("ul", [f"`{r['child_table']}.{r['child_column']}` refers to "
                              f"`{r['parent_table']}.{r['parent_column']}` ({r['containment']:.1%} of values "
                              f"match; {r['orphan_rows']} rows without a match)." for r in run["relationships"]]))

    # ---- Cleaning ---------------------------------------------------------
    blocks.append(("h2", "Cleaning applied"))
    blocks.append(("p", "Raw files were not modified. Rule-by-rule detail is in `02_cleaning_log.json`."))
    blocks.append(("table", ["Table", "Column", "Rule", "Rows affected"],
                   [[e["table"], e["column"] or "(whole row)", e["rule"], e["rows_affected"]]
                    for e in run["cleaning_log"]]))

    # ---- Assumptions ------------------------------------------------------
    blocks.append(("h2", "Assumptions to check before acting"))
    blocks.append(("ul", run["assumptions"]))
    return blocks


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------

def render_markdown(blocks):
    out = []
    for block in blocks:
        kind = block[0]
        if kind in ("h1", "h2", "h3"):
            out.append("#" * int(kind[1]) + " " + block[1])
        elif kind == "p":
            out.append(block[1])
        elif kind == "ul":
            out.append("\n".join(f"- {item}" for item in block[1]))
        elif kind == "img":
            out.append(f"![{block[2]}]({block[1]})")
        elif kind == "table":
            header, rows = block[1], block[2]
            lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            lines += ["| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |" for row in rows]
            out.append("\n".join(lines))
    return "\n\n".join(out) + "\n"


def inline_html(text):
    text = html.escape(str(text))
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", text)


HTML_STYLE = """
:root { --bg:#f9f9f7; --card:#fcfcfb; --ink:#0b0b0b; --muted:#52514e; --line:#e1e0d9; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d0d0d; --card:#1a1a19; --ink:#f0efec; --muted:#c3c2b7; --line:#2c2c2a; } }
body { background:var(--bg); color:var(--ink); font:16px/1.55 system-ui,sans-serif; margin:0; }
main { max-width:900px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:1.9rem; } h2 { margin-top:2.5rem; border-top:1px solid var(--line); padding-top:1.2rem; }
h3 { margin-top:2rem; font-size:1.1rem; } p, li { color:var(--muted); } strong { color:var(--ink); }
img { max-width:100%; border:1px solid var(--line); border-radius:8px; }
.scroll { overflow-x:auto; } table { border-collapse:collapse; width:100%; font-size:0.9rem; }
th, td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
code { background:var(--card); border:1px solid var(--line); border-radius:4px; padding:0 4px; font-size:0.9em; }
"""


def render_html(blocks, output_dir):
    out = []
    for block in blocks:
        kind = block[0]
        if kind in ("h1", "h2", "h3", "p"):
            out.append(f"<{kind}>{inline_html(block[1])}</{kind}>")
        elif kind == "ul":
            out.append("<ul>" + "".join(f"<li>{inline_html(item)}</li>" for item in block[1]) + "</ul>")
        elif kind == "img":
            encoded = base64.b64encode((Path(output_dir) / block[1]).read_bytes()).decode()
            out.append(f'<img alt="{html.escape(block[2])}" src="data:image/png;base64,{encoded}">')
        elif kind == "table":
            head = "".join(f"<th>{inline_html(h)}</th>" for h in block[1])
            body = "".join("<tr>" + "".join(f"<td>{inline_html(c)}</td>" for c in row) + "</tr>"
                           for row in block[2])
            out.append(f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
                       f"<tbody>{body}</tbody></table></div>")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Findings report</title><style>{HTML_STYLE}</style></head>"
            f"<body><main>{''.join(out)}</main></body></html>\n")


def write_report(run, output_dir):
    blocks = build_blocks(run)
    output_dir = Path(output_dir)
    (output_dir / "06_report.md").write_text(render_markdown(blocks))
    (output_dir / "06_report.html").write_text(render_html(blocks, output_dir))


def write_index(output_root):
    """output_root/index.html: one row per analysis folder that has a report."""
    import json
    output_root = Path(output_root)
    rows = []
    for folder in sorted(p for p in output_root.iterdir() if (p / "06_report.html").exists()):
        manifest = json.loads((folder / "run_manifest.json").read_text())
        counts = manifest["counts"]
        rows.append([f'<a href="{html.escape(folder.name)}/06_report.html">{html.escape(folder.name)}</a>',
                     ", ".join(html.escape(name) for name in manifest["input_files_sha256"]),
                     counts["candidates"], counts["accepted"], counts["rejected"]])
    head = "".join(f"<th>{h}</th>" for h in ["Analysis", "Input files", "Patterns tested", "Passed", "Rejected"])
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    page = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Analyses</title><style>{HTML_STYLE} a {{ color: inherit; }}</style></head><body><main>"
            "<h1>Analyses</h1><p>One folder per dataset. Click a name to open its findings report.</p>"
            f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
            "</main></body></html>\n")
    (output_root / "index.html").write_text(page)
