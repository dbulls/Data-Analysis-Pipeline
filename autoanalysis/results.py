"""Read a finished analysis back from its output folder (used by the app).

`pipeline.run_pipeline` returns everything in memory but only saves the pieces
as separate files, so this module puts them back together. It never runs
statistics; it only loads what is already decided.
"""

import json
from pathlib import Path

import pandas as pd

from . import charts
from .config import Config


def read_json(path):
    with open(path) as f:
        return json.load(f)


def list_runs(output_root):
    """Names of the analyses in an output folder, i.e. subfolders with a finished run."""
    output_root = Path(output_root)
    if not output_root.is_dir():
        return []
    return sorted(p.name for p in output_root.iterdir() if (p / "run_manifest.json").exists())


def load_config(manifest):
    """The Config a run used, rebuilt from the manifest (JSON turns tuples into lists)."""
    config = Config()
    for key, value in manifest["config"].items():
        if hasattr(config, key):
            setattr(config, key, tuple(value) if isinstance(getattr(config, key), tuple) else value)
    config.effect_thresholds = {kind: tuple(pair) for kind, pair in config.effect_thresholds.items()}
    return config


def load_run(output_dir):
    """Everything the app shows about one analysis, as plain dicts and lists."""
    output_dir = Path(output_dir)
    manifest = read_json(output_dir / "run_manifest.json")
    return {
        "name": output_dir.name,
        "manifest": manifest,
        "config": load_config(manifest),
        "profile": read_json(output_dir / "01_profile.json"),
        "relationships": read_json(output_dir / "01_relationships.json"),
        "cleaning_log": read_json(output_dir / "02_cleaning_log.json"),
        "accepted": read_json(output_dir / "04_accepted.json"),
        "rejected": read_json(output_dir / "04_rejected.json"),
        "skipped": read_json(output_dir / "03_skipped.json"),
        "assumptions": read_json(output_dir / "assumptions.json"),
    }


def load_views(output_dir):
    """{table: DataFrame}: the tables as the search saw them, which the chart drawers need."""
    views_dir = Path(output_dir) / "02_views"
    return {path.stem: pd.read_pickle(path) for path in sorted(views_dir.glob("*.pkl"))}


DATE_FIELDS = ("break_date", "midpoint")   # summary values that are dates in memory but text in the JSON


def with_dates_restored(finding):
    """A copy of a saved finding whose date values are dates again, as the chart drawers expect."""
    summary = dict(finding["result"]["summary"])
    for key in DATE_FIELDS:
        if summary.get(key) is not None:
            summary[key] = pd.Timestamp(summary[key])
    return {**finding, "result": {**finding["result"], "summary": summary}}


def chart_figure(finding, views, config):
    """A matplotlib Figure for any accepted finding, drawn on demand."""
    finding = with_dates_restored(finding)
    return charts.make_figure(finding, views[finding["table"]], config)
