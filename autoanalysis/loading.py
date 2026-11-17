"""Reading raw data and writing results.

Raw files are only ever opened for reading. Every value is loaded as text so
nothing is silently converted before the cleaning rules get to log it.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def csv_files(source):
    """The CSV files making up one dataset: a single .csv file, or every .csv in a folder."""
    source = Path(source)
    if source.is_file():
        return [source]
    files = sorted(source.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No .csv files found in {source.resolve()}")
    return files


def discover_datasets(data_root):
    """Split a data folder into separate datasets, each analysed on its own.

    - every .csv directly in data_root is one dataset, named after the file
    - every subfolder holding .csv files is one dataset of linked tables
      (e.g. orders.csv + customers.csv), named after the folder

    Returns [(name, path)] sorted by name.
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data folder not found: {data_root.resolve()}")
    datasets = [(path.stem, path) for path in data_root.glob("*.csv")]
    datasets += [(path.name, path) for path in data_root.iterdir()
                 if path.is_dir() and not path.name.startswith(".") and any(path.glob("*.csv"))]
    names = [name for name, _ in datasets]
    clashes = sorted({name for name in names if names.count(name) > 1})
    if clashes:
        raise ValueError(f"A file and a folder share the name(s) {clashes} in {data_root}; rename one.")
    if not datasets:
        raise FileNotFoundError(f"No .csv files or folders of .csv files found in {data_root.resolve()}")
    return sorted(datasets)


def load_raw_tables(source):
    """Return {table_name: DataFrame of strings} for one dataset (a .csv file or a folder of them)."""
    return {path.stem: pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
            for path in csv_files(source)}


def file_hashes(source):
    """SHA-256 of every input file, so a run can be tied to exact inputs."""
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in csv_files(source)}


def to_jsonable(value):
    """Convert numpy / pandas values into plain JSON types (NaN becomes null)."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return round(number, 10)   # stable text output across runs
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(data), f, indent=2, sort_keys=False)
        f.write("\n")
