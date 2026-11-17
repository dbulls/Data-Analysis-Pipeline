"""Run every stage in order and save each intermediate result.

Output files (numbered in the order they are produced):
    01_profile.json          every table and column described
    01_relationships.json    shared keys found between tables
    02_cleaning_log.json     every cleaning rule that changed something
    02_cleaned/              cleaned copies of the tables + outlier flags (+ derived change tables)
    02_persistent_series.json  columns analysed as % changes instead of levels
    03_candidates.json       every pattern tested, with its measurement
    03_skipped.json          patterns that could not be tested (data too thin)
    04_accepted.json         findings that passed every check, ranked
    04_rejected.json         findings that failed, with every reason
    05_charts/               one PNG per accepted finding
    06_report.md / .html     the findings report
    assumptions.json         every assumption the run relied on
    run_manifest.json        input file hashes, config and library versions
"""

from pathlib import Path

import matplotlib
import numpy
import pandas
import scipy

from . import charts, clean, filters, loading, profile, relationships, report, search, timeseries
from .config import Config


def collect_assumptions(config, run):
    """Everything a reader should know was assumed rather than known."""
    items = [
        "Column types were inferred from the values, not declared. Check the 'Inferred type' column in "
        "the profile; override with `type_overrides` in the config if one is wrong.",
        "Units are not recorded in the data. Axis labels say so unless units are supplied in the config.",
        "Missing values were never filled in. Each analysis uses only rows where its columns are present, "
        "which assumes values are not missing for a reason related to the pattern.",
        f"Outliers are values beyond {config.outlier_iqr_multiplier} x the interquartile range. They are "
        "flagged and kept; findings are re-tested without them.",
        f"A finding needs a false-discovery-rate adjusted p-value of at most {config.alpha} (Benjamini-Hochberg "
        "across every pattern tested) and an effect at least as large as the minimum in `effect_thresholds`. "
        "Those minimums are judgement calls.",
        f"A pattern 'holds' on a subset if it points the same way and keeps at least "
        f"{config.effect_retention:.0%} of its full-data size ({config.effect_retention_in_halves:.0%} in each "
        "half, where estimates are noisier). The halves are parts of the same dataset, so this is a "
        "stability check, not independent replication.",
        "Time trends use per-period averages (not totals) so that more records do not look like growth. "
        "Count and total series drop the first and last period, assumed to be partly covered.",
        f"Period averages built from fewer than {config.min_records_per_period} records are ignored; groups "
        f"with fewer than {config.min_group_size} records are left out of group comparisons.",
        "Statistical tests treat records as independent. Where a comparison uses a column joined from a "
        "parent table, records are first reduced to one value per parent to respect this. Other hidden "
        "groupings (repeat measurements, batches) are not known to the pipeline and would make p-values optimistic.",
        "Trend and anomaly tests ignore autocorrelation between neighbouring periods, which makes them "
        "somewhat over-confident.",
        "The ranking's 'actionability' component is a fixed weight per kind of finding, not something "
        "measured from the data.",
        f"Results are reproducible: random steps use seed {config.seed}.",
    ]
    for entry in run["cleaning_log"]:
        if entry["rule"] == "R4 ambiguous date order":
            items.append(f"Dates in `{entry['table']}.{entry['column']}` fit more than one format. {entry['detail']}")
        if entry["rule"] == "R6 drop duplicate rows":
            items.append(f"{entry['rows_affected']} exact duplicate rows in `{entry['table']}` were treated as "
                         "accidental repeats and analysed once.")
    for rel in run["relationships"]:
        items.append(f"`{rel['child_table']}.{rel['child_column']}` was assumed to refer to "
                     f"`{rel['parent_table']}.{rel['parent_column']}` because {rel['containment']:.1%} of its "
                     "values appear there.")
    items.extend(run["search_notes"])
    return items


def run_pipeline(data_dir, output_dir, config=None):
    config = config or Config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and profile
    raw_tables = loading.load_raw_tables(data_dir)
    table_profile, types = profile.profile_tables(raw_tables, config)
    loading.save_json(table_profile, output_dir / "01_profile.json")

    # 2. Clean (raw_tables is left untouched)
    cleaned, outlier_flags, cleaning_log = clean.clean_tables(raw_tables, types, config)
    loading.save_json(cleaning_log, output_dir / "02_cleaning_log.json")
    cleaned_dir = output_dir / "02_cleaned"
    cleaned_dir.mkdir(exist_ok=True)
    for name, df in cleaned.items():
        df.to_csv(cleaned_dir / f"{name}.csv", index=False)
        outlier_flags[name].to_csv(cleaned_dir / f"{name}__outlier_flags.csv", index_label="raw_row")

    # Relationships and joined views
    found = relationships.detect_relationships(cleaned, types, config)
    loading.save_json(found, output_dir / "01_relationships.json")
    views, view_types, joined_columns = relationships.build_views(cleaned, types, found)
    change_notes, persistent = timeseries.add_change_tables(views, view_types, joined_columns, config)
    loading.save_json(persistent, output_dir / "02_persistent_series.json")
    for name in views:
        if name.endswith(timeseries.CHANGE_TABLE_SUFFIX):
            views[name].to_csv(cleaned_dir / f"{name}.csv", index=False)

    # 3. Search for candidate patterns
    candidates, skipped, search_notes = search.search_all(views, view_types, joined_columns, found, config)
    loading.save_json(candidates, output_dir / "03_candidates.json")
    loading.save_json(skipped, output_dir / "03_skipped.json")

    # 4. Filter for real signal
    accepted, rejected = filters.evaluate(candidates, views, view_types, joined_columns, config)

    # 5. Charts
    charts.draw_all(accepted, views, config, output_dir / "05_charts")
    loading.save_json(accepted, output_dir / "04_accepted.json")
    loading.save_json(rejected, output_dir / "04_rejected.json")

    # 6. Report
    run = {
        "data_dir": str(data_dir), "config": config, "profile": table_profile, "types": types,
        "relationships": found, "cleaning_log": cleaning_log, "candidates": candidates,
        "skipped": skipped, "accepted": accepted, "rejected": rejected,
        "joined_columns": joined_columns, "search_notes": change_notes + search_notes,
    }
    run["assumptions"] = collect_assumptions(config, run)
    loading.save_json(run["assumptions"], output_dir / "assumptions.json")
    loading.save_json({
        "input_files_sha256": loading.file_hashes(data_dir),
        "config": config.to_dict(),
        "versions": {"pandas": pandas.__version__, "numpy": numpy.__version__,
                     "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "counts": {"candidates": len(candidates), "skipped": len(skipped),
                   "accepted": len(accepted), "rejected": len(rejected)},
    }, output_dir / "run_manifest.json")
    report.write_report(run, output_dir)
    return run
