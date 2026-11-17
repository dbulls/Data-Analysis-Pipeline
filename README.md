# Automated data analysis pipeline

Point it at a folder of CSV files. It works out the schema, cleans the data
with logged rules, searches for patterns, throws away the ones that do not
survive statistical checks, and writes a ranked findings report with charts.
No column names are hard-coded and no questions need to be specified.

## Run it

```bash
pip install -r requirements.txt
```

```bash
python run_pipeline.py
```

That analyses every dataset in `data/`, each into its own folder in
`output/`.

## How datasets map to output folders

| Put this in `data/` | Analysed as | Results in |
|---|---|---|
| `sales.csv` | one dataset, on its own | `output/sales/` |
| `weather.csv` | another dataset, on its own | `output/weather/` |
| a folder `shop/` holding `orders.csv` and `customers.csv` | one dataset of linked tables (links between them are detected) | `output/shop/` |

So: unrelated files go straight into `data/`; tables that belong together
(sharing keys such as `customer_id`) go together in a subfolder. A file and
a folder with the same name is an error.

To re-run just one dataset, leaving the others' results as they are:

```bash
python run_pipeline.py --only sales
```

Re-running a dataset replaces its folder's contents. To keep an earlier run,
use a different output folder: `--output output_before_cleanup`.

## Viewing the output

```bash
open output/index.html
```

That opens a page listing every analysis, with links to each report (macOS;
on Windows use `start`, on Linux `xdg-open`). To go straight to one:

```bash
open output/sales/06_report.html
```

Each report is one self-contained file with the charts embedded, so it can
be emailed or put on a shared drive. Inside each analysis folder, charts are
also saved as PNGs in `05_charts/`, and the JSON files are the audit trail
(see "Auditing a conclusion" below).

## Trying it on synthetic data

Invented data with planted patterns, used by the tests:

```bash
python make_synthetic_data.py
```

```bash
python run_pipeline.py --data sample_data --output sample_output
```

Run the tests:

```bash
python -m pytest -q
```

## Checking results on new data

1. Raw files are only read, never written.
2. Check the **inferred types** in the report's "What the data looks like"
   section first. Everything downstream depends on them.
3. If something is wrong or missing, write a small JSON file and pass it with
   `--config`:

```json
{
  "type_overrides": {"orders.zip_code": "categorical"},
  "units": {"orders.amount": "USD", "delivery_days": "days"},
  "alpha": 0.01
}
```

Keys in `type_overrides` and `units` are `table.column`, where the table
name is the CSV's file name. One config file applies to every dataset in the
run; use `--only` to run a dataset with its own config.

Any field in `autoanalysis/config.py` can be overridden this way. That file
is the single place where every threshold lives.

Input is CSV only. To use a database, export the tables to CSV first, or
replace `load_raw_tables` in `autoanalysis/loading.py` (it just needs to
return `{table_name: DataFrame of strings}`).

## What it does, stage by stage

| Stage | Module | Output |
|---|---|---|
| - | `run_pipeline.py`, `loading.py` | one folder per dataset, `index.html` |
| 1. Profile | `infer.py`, `profile.py`, `relationships.py` | `01_profile.json`, `01_relationships.json` |
| 2. Clean | `clean.py` | `02_cleaning_log.json`, `02_cleaned/` |
| 3. Search | `search.py`, `measures.py` | `03_candidates.json`, `03_skipped.json` |
| 4. Filter | `filters.py` | `04_accepted.json`, `04_rejected.json` |
| 5. Charts | `charts.py` | `05_charts/*.png` |
| 6. Report | `wording.py`, `report.py` | `06_report.md`, `06_report.html` |
| - | `pipeline.py` | `assumptions.json`, `run_manifest.json` |

**Profile.** Each column is typed from its values (numeric, boolean,
datetime, categorical, identifier, text, empty) and described: range,
quartiles, histogram, distinct count, missing rate, duplicates. A column whose
values almost all appear in another table's unique key is treated as a link
between the tables; the parent's descriptive columns are then joined on so
that, for example, order amounts can be compared by customer segment.

**Clean.** Eight rules (R1-R8, listed at the top of `clean.py`): trim
whitespace, convert missing-value tokens, convert numbers and dates (anything
unparseable becomes missing and is logged), merge spellings that differ only
by letter case, drop exact duplicate rows only when the table has a row
identifier, flag outliers without deleting them, and never fill in missing
values.

**Search.** Every sensible column combination is measured once for each
pattern category:

| Category | Test | Effect size |
|---|---|---|
| Trend over time (per-period mean, row count, total) | Mann-Kendall | Kendall tau, Theil-Sen slope |
| Seasonality (month of year, day of week) | Kruskal-Wallis on detrended values | seasonal range in std devs |
| Breakpoint | permutation test of the best before/after split | Cohen's d |
| Anomaly (one unusual period) | Welch t-test vs. neighbouring periods | robust z-score |
| Difference between groups | Kruskal-Wallis | epsilon-squared |
| Relationship between numeric variables | Spearman correlation | rho |
| Concentration | bootstrap of the top-20% share | top-20% share, Gini |
| Distribution shift (first vs. second half) | Kolmogorov-Smirnov | KS D |

**Filter.** A candidate becomes a finding only if all of these hold:

1. Its p-value survives Benjamini-Hochberg correction across *every* pattern
   tested (false discovery rate 1% by default).
2. Its effect size clears a minimum (so huge datasets do not turn trivia into
   findings).
3. It holds when flagged outliers are removed.
4. It holds when the single most influential record (or entity) is removed.
5. It holds in both halves of the data (first vs. second half of the time
   range where that makes sense, otherwise alternating records).
6. It is not a data-quality artefact (column mostly missing, two columns that
   are really the same thing, a "seasonal" pattern caused by a one-off step).
7. It is not trivially expected (totals rising because there are more rows,
   per-customer totals concentrated because orders are) or a duplicate of a
   better description of the same thing.

Survivors are ranked by size, reliability and actionability, and given a
confidence level (high / medium / low). Everything rejected is kept, with
every reason, in `04_rejected.json`.

**Charts.** The kind of finding picks the chart: line for trends, breakpoints
and anomalies; box or dot plot for groups; scatter with fitted line for
relationships; cumulative-share curve for concentration; overlaid histograms
for distribution shifts. Each title is the finding as a sentence.

## Auditing a conclusion

Every candidate has a stable id such as
`group_diff|orders|customers.segment|amount`. Look it up in
`03_candidates.json` for the raw measurement, and in `04_accepted.json` or
`04_rejected.json` for the adjusted p-value, each robustness check with the
numbers it produced, and the rejection reasons. `02_cleaning_log.json` lists
every cleaning rule that changed anything, with row counts.

## Reproducibility

The same input files and config give byte-identical JSON and reports (there
is a test for this). Random steps (permutations, bootstraps) are seeded from
`config.seed` and the candidate id, so results do not depend on run order.
`run_manifest.json` records the SHA-256 of each input file, the full config
and library versions. Outputs contain no timestamps.

## Prices, rates and other carried-over values

Some columns are levels that carry over from one record to the next:
exchange rates, prices, balances. In a table with one record per time stamp,
the pipeline checks each numeric column for this (correlation of each value
with the previous one, after removing each month's average). Such columns
are not tested as levels. Every test here assumes each record is fresh
evidence, which is false for them: two unrelated prices that both drifted
up would look "correlated". Instead the pipeline builds a derived table
`<table>__changes` holding each column's % change from the previous record,
and analyses that. A finding such as "A and B move together" then means
their record-to-record moves are related.

In tables with one record per time stamp, row counts, totals and
concentration are also not tested, since adding up a rate across days means
nothing.

## Changing it

- **A threshold:** `autoanalysis/config.py`.
- **A new cleaning rule:** add it in `clean_table` and call `record(...)` so
  it is logged; add a test in `tests/test_cleaning.py`.
- **A new kind of pattern:** write a `measure_...` function in `measures.py`
  returning the standard result dict, add it to `measure()`, enumerate it in
  `search.py`, add thresholds in `config.py`, a sentence in `wording.py` and
  a drawer in `charts.py`. The filters apply automatically.
- **A new rejection rule:** `apply_individual_rules` (one candidate) or
  `apply_cross_candidate_rules` (comparing candidates) in `filters.py`.

## Known limits

- Halves of the same dataset are not independent replication. The split check
  shows a pattern is stable, not that it will hold in new data.
- Tests assume records are independent. Joined-table comparisons are done per
  parent row to respect this, but other hidden groupings are invisible to the
  pipeline.
- Trend tests ignore autocorrelation and are somewhat over-confident.
- One table is enriched with its direct parents only (one level of joins).
- Not searched: category-by-category associations, interactions between more
  than two columns, seasonality in row counts, free-text content.
- An identifier-like integer column (unique whole numbers) is treated as an
  identifier and not analysed. Override the type if it is really a measure.
- Units are unknown unless supplied in the config.
- Sudden one-record jumps in a change series (e.g. a one-day devaluation)
  are rejected by the "driven by a single record" rule, so they are not
  reported. Look at `04_rejected.json` for `anomaly_mean` candidates on a
  `__changes` table if you need them.
- On the synthetic data, across 10 random seeds, all nine planted patterns
  were recovered every time and one spurious low-confidence finding appeared
  in total. Expect roughly that false-discovery rate, not zero.
