"""Every threshold the pipeline uses, in one place.

Change a value here (or pass a JSON file with --config) and re-run.
All of these are assumptions; they are copied into the report.
"""

from dataclasses import dataclass, field, asdict
import json


@dataclass
class Config:
    # ---- Reproducibility -------------------------------------------------
    seed: int = 20240101           # seeds permutation tests and bootstraps

    # ---- Type inference --------------------------------------------------
    missing_tokens: tuple = ("", "na", "n/a", "nan", "null", "none", "-", "?")
    parse_success_rate: float = 0.95   # share of values that must parse as a type
    max_categories: int = 30           # more distinct values than this is not "categorical"
    identifier_unique_rate: float = 0.95
    # Force a type: {"table.column": "numeric|categorical|datetime|boolean|text|identifier"}
    type_overrides: dict = field(default_factory=dict)
    # Units for axis labels: {"table.column": "USD"}
    units: dict = field(default_factory=dict)

    # Lag-1 autocorrelation (after removing each period's mean) at which a
    # series counts as a carried-over level and is analysed as % changes.
    persistence_threshold: float = 0.5

    # ---- Cleaning --------------------------------------------------------
    outlier_iqr_multiplier: float = 3.0   # Tukey "far out" fence; flags, never deletes
    relationship_min_containment: float = 0.95

    # ---- Minimum data needed before a pattern is even tested -------------
    min_periods: int = 12            # time periods for trends / breakpoints / anomalies
    min_records_per_period: int = 5  # a period mean from fewer records is not used
    min_group_size: int = 20         # groups smaller than this are left out of comparisons
    min_pairs: int = 30              # complete pairs for a correlation
    min_units_concentration: int = 30
    # Robustness re-checks run on about half the data, so their minimums are
    # the values above multiplied by this factor.
    subset_minimum_factor: float = 0.4

    # ---- Significance ----------------------------------------------------
    alpha: float = 0.01              # false discovery rate (Benjamini-Hochberg); deliberately strict
    n_permutations: int = 9999       # breakpoint permutation test (smallest possible p = 0.0001)
    n_bootstrap: int = 4999          # concentration bootstrap (smallest possible p = 0.0002)
    anomaly_z_threshold: float = 4.0 # robust z-score needed to propose an anomaly
    max_anomalies_per_series: int = 2

    # ---- Effect sizes: (smallest worth reporting, "large") ---------------
    effect_thresholds: dict = field(default_factory=lambda: {
        "trend_mean":    (0.30, 0.60),   # |Kendall tau| of period means vs time
        "trend_count":   (0.30, 0.60),
        "trend_total":   (0.30, 0.60),
        "seasonality":   (0.20, 0.80),   # (highest - lowest bucket mean) / std dev
        "breakpoint":    (0.20, 0.80),   # |Cohen's d| of records after vs before
        "anomaly_mean":  (4.00, 8.00),   # |robust z-score|
        "anomaly_count": (4.00, 8.00),
        "group_diff":    (0.02, 0.14),   # epsilon-squared (share of rank variance)
        "correlation":   (0.20, 0.50),   # |Spearman rho|
        "concentration": (0.50, 0.80),   # share of total held by top 20% of units
        "dist_shift":    (0.15, 0.40),   # Kolmogorov-Smirnov D
    })

    # ---- Robustness ------------------------------------------------------
    # A pattern "holds" on a subset if it points the same way and keeps at
    # least this share of its full-data effect size.
    effect_retention: float = 0.5
    # Each half has half the data, so its estimate is noisier; ask for less.
    effect_retention_in_halves: float = 0.3
    max_missing_rate: float = 0.5        # above this a column is too incomplete to trust
    near_duplicate_correlation: float = 0.98  # above this two columns are treated as the same thing

    # ---- Ranking ---------------------------------------------------------
    weight_size: float = 0.4
    weight_reliability: float = 0.4
    weight_actionability: float = 0.2
    # How directly a kind of finding points at something a person could act
    # on. This is a judgement call, not a statistic.
    actionability: dict = field(default_factory=lambda: {
        "group_diff": 0.9, "concentration": 0.9, "breakpoint": 0.8,
        "anomaly_mean": 0.7, "anomaly_count": 0.7, "trend_mean": 0.7,
        "trend_count": 0.6, "trend_total": 0.5, "seasonality": 0.6,
        "correlation": 0.5, "dist_shift": 0.4,
    })
    top_n_findings: int = 15
    # At most this many findings of one kind in the top list, so one pattern
    # repeated many times (e.g. dozens of similar correlations) cannot crowd
    # out everything else. The rest are listed below it.
    max_per_kind_in_top: int = 4

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_file(cls, path):
        """Load overrides from a JSON file; unknown keys are an error."""
        with open(path) as f:
            overrides = json.load(f)
        config = cls()
        for key, value in overrides.items():
            if not hasattr(config, key):
                raise ValueError(f"Unknown config key: {key}")
            current = getattr(config, key)
            if isinstance(current, dict):
                current.update(value)
            else:
                setattr(config, key, type(current)(value) if isinstance(current, tuple) else value)
        return config
