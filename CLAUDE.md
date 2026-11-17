# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An automated exploratory analysis pipeline. Given CSV files, it infers the schema itself, cleans the data with logged rules, and searches systematically for patterns: trends, seasonality, breakpoints, anomalies, group differences, correlations, concentration and distribution shifts. The main job is the filtering stage: every candidate pattern must survive multiple-comparison correction, a minimum effect size, and robustness re-tests before it is reported. It outputs a ranked findings report with charts, confidence levels, rejected patterns with reasons, and the stated assumptions. No column names are hard-coded, and nothing is presented as certain. For setup and user-facing usage, see [README.md](README.md).

## Architecture

```
data/            user's datasets (read-only)  ──►  run_pipeline.py (CLI: one analysis per dataset)
autoanalysis/    the pipeline package         ──►  pipeline.run_pipeline (orchestrates the stages)
output/<name>/   one folder per dataset, numbered audit files 01_… → 06_report.html; output/index.html links them
tests/           pytest, built on a synthetic dataset with planted patterns and traps
```

Data flow inside `pipeline.run_pipeline`:

1. **Load:** `loading` reads each CSV as strings only, so nothing is converted silently.
2. **Profile:** `infer` decides each column's type; `profile` describes every column.
3. **Clean:** `clean` applies rules R1–R8 and writes a log entry for each change.
4. **Join:** `relationships` finds shared keys between tables and builds per-table *views* with parent columns joined on.
5. **Change tables:** `timeseries` adds a derived `<table>__changes` view for carried-over series such as prices and rates.
6. **Search:** `search` enumerates *candidates* over the views, and `measures` measures each one once.
7. **Filter:** `filters` applies Benjamini-Hochberg, re-measures on subsets, applies the rejection rules, then ranks, assigns confidence and picks the top list.
8. **Charts:** `charts` draws one PNG per top finding.
9. **Report:** `wording` produces the sentences; `report` writes Markdown, self-contained HTML and the index page.

The **candidate** is the unit everything passes around: `{id, kind, category, table, columns}`, gaining `result`, `q_value`, `checks`, `rejections`, `status`, `score`, `confidence`, `rank` and `in_top` as it moves through the pipeline. Its `id` (e.g. `group_diff|orders|customers.segment|amount`) is stable and serves as both the audit key and part of the RNG seed.

## Key files

- `run_pipeline.py`: CLI. Discovers datasets, gives each a fresh `Config`, runs the pipeline and writes `output/index.html`.
- `autoanalysis/pipeline.py`: stage order, intermediate file names, and the assumptions list in the report.
- `autoanalysis/config.py`: every threshold, in one place. Overridable with `--config file.json`.
- `autoanalysis/measures.py`: one pure `measure_*` function per kind of pattern, plus the `measure()` dispatcher.
- `autoanalysis/filters.py`: significance, robustness checks, rejection rules, ranking, confidence and `mark_top`.
- `autoanalysis/timeseries.py`: detects persistent series and builds the `__changes` tables.
- `autoanalysis/wording.py`: the plain-language statement (also used as the chart title) and the "to confirm" text.
- `autoanalysis/labels.py`: readable names for columns and tables. Everything a person reads goes through it.
- `autoanalysis/results.py`: reads a finished `output/<dataset>/` back (JSON plus `02_views/`) and redraws any finding's chart.
- `app.py`: Streamlit app: run a dataset, then page through results. Reads only what `output/` holds.
- `make_synthetic_data.py`: generates `sample_data/shop/` with documented planted patterns and traps.
- `tests/conftest.py`: session fixtures, which run the full pipeline once on synthetic data (seed 7).

## Run and test

```bash
python3 run_pipeline.py                                    # all datasets in data/ -> output/<dataset>/
python3 run_pipeline.py --only student_performance         # one dataset; other output folders untouched
python3 run_pipeline.py --data sample_data --output sample_output
python3 make_synthetic_data.py                             # regenerate sample_data/shop/
python3 -m streamlit run app.py                            # the GUI (run datasets, browse results)
python3 -m pytest -q                                       # full suite, ~25s
python3 -m pytest tests/test_filters.py::test_noise_trend_is_rejected -q
```

There is no build or lint step. Dependencies are in `requirements.txt`. The exchange-rate dataset takes ~90s to run.

## Conventions

- Readability over cleverness; the user modifies this code. Plain functions and dicts, no classes beyond the `Config` dataclass. Keep each module to one stage.
- Every measurement returns the same dict shape: `{ok: True, n, test, p_value, effect_name, effect_size, direction, summary}` or `{ok: False, reason}`. "Too little data" is a returned reason, never an exception.
- `measure_*` functions are pure functions of `(df, cols, config, relaxed)`, because the filters re-call them on subsets. `relaxed=True` lowers the minimum sample sizes (`scaled_minimum`); `need_p=False` skips expensive p-values.
- `direction` is what replication must preserve: a sign (±1), or a `{label: value}` profile compared by shape. Add `replication_value`/`replication_name` when `effect_size` isn't comparable across subset sizes (trends do this).
- Each rejection is `{code, explanation}` with a code from `report.REJECTION_TITLES`, and the explanation must quote the numbers. Never drop a candidate silently.
- User-facing text never shows raw column names: use `labels.pretty_column` / `pretty_table` (and pass `config.labels`). Candidate ids and file names stay raw.
- Cleaning rules call `record(rule, column, rows_affected, detail)` so every change is logged.
- Thresholds and judgement calls live in `config.py`, never inline, and the assumptions text in `pipeline.collect_assumptions` must quote them.
- Adding a kind of finding means touching all of these: `measure_*` + dispatcher, `search` enumeration + `CATEGORIES`, `effect_thresholds` + `actionability` in config, `wording.statement` and `what_would_confirm`, `charts.DRAWERS`, and a test.

## Constraints and gotchas

- **Never modify `data/`.** It holds the user's real files. `output/`, `sample_output/` and `sample_data/` are generated and safe to regenerate.
- **Determinism is tested.** The same input must give byte-identical JSON and reports. So: no timestamps, randomness only via `measures.stable_rng(config, candidate_id)` (independent of run order), and floats rounded in `loading.to_jsonable`.
- **Never invent or fill data.** Missing values stay missing (rule R8). Outliers are flagged and kept, never deleted.
- **Joined parent columns** are named `parent_table.column`. `joined_columns[table]` maps each to the child key. Candidates that use one carry `unit=<key>` and are aggregated to one row per parent before testing, because records sharing a parent aren't independent.
- **Persistent series.** In tables with one row per timestamp, columns whose within-period lag-1 autocorrelation is ≥ `persistence_threshold` get type `persistent_level`, so search skips them. They're analysed via `<table>__changes` (`*_pct_change` columns) instead. Without this, random-walk data such as FX rates produces hundreds of spurious "findings". Such tables also skip counts, totals and concentration.
- `timeseries` writes into `config.units`, so one `Config` must never be reused across datasets.
- The **split-halves check** uses first/second half of the time range where meaningful, and alternating records otherwise (breakpoints, distribution shifts). Anomalies skip it: a single period can't be split.
- **Calibration.** Tuning was checked across synthetic seeds 1–10: all 9 planted patterns recovered, about 1 false positive in total. Re-check across seeds, not just seed 7, after changing any threshold or rule.
- matplotlib runs headless (`Agg`). Chart titles are figure-level `suptitle` so long labels don't clip them.

## Current state

Updating this section is part of finishing any task: record what changed, what's in progress, and any new known gap.

**Built and tested (65 tests passing):**
- the full pipeline (all stages above);
- one analysis per dataset with `--only` and `output/index.html`;
- linked-table joins with per-parent aggregation;
- persistent-series handling via change tables;
- a per-kind cap on the report's top list;
- a "common driver" note when most correlations pass;
- Streamlit app (`app.py`): run, slide through top findings, chart explorer for all accepted findings, rejected patterns, data and assumptions pages; readable labels in sentences, charts and the app;
- parsing of messy formats: one currency/unit around numbers, decimal commas, mixed yes/no spellings, mixed date formats per column (logged as R3/R4 entries); re-checked on seeds 1–10, unchanged;
- the app explains a run where nothing could be tested instead of saying nothing passed;
- `02_views/*.pkl` saved per run (typed tables the chart drawers need) and an optional `progress` callback on `run_pipeline`.

**Run on the user's data:**
- `exchange_rate_to_usd` (daily FX, 2004–2026, 50 currencies): mostly co-movement of daily changes;
- `student_performance` (100 rows): study hours vs. exam score, rho +0.91;
- `orders_messy` (500 rows): unit price by category, order count rising. Category has unmerged spelling variants (Book/BOOKS, Apparel/Apparell), so groups are split; Shipping stays categorical because of `FREE` words.

**Known gaps / not started:**
- One-day jumps in change series (e.g. devaluations) are rejected as `driven_by_single_record`. Detecting them was offered and deferred by the user.
- No mapping of category synonyms or number words (`FREE` -> 0); would need a config option.
- Not searched: category-by-category associations, interactions among 3+ columns, seasonality in row counts, free text.
- Joins go one level deep only. Input is CSV only.
- Trend tests ignore autocorrelation between periods.
- The Trinidad & Tobago dollar is managed, so it isn't detected as persistent and is tested as a level; its p-values are optimistic.
- App: not tested on very wide or phone layouts; chart axis labels still say "units as recorded in source" unless `units` is set; rejection explanations and check details still quote raw column names.
- Runs made before `02_views/` existed need one re-run before the explorer can chart them.
- Not a git repository yet.
