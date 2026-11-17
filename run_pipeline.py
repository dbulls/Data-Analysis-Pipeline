"""Analyse every dataset in a data folder, each into its own output folder.

    python run_pipeline.py                         # all datasets in ./data -> ./output/<dataset>/
    python run_pipeline.py --only sales            # just one dataset
    python run_pipeline.py --config my_config.json

What counts as a dataset:
    data/sales.csv                   -> output/sales/
    data/shop/orders.csv             \\
    data/shop/customers.csv          /  -> output/shop/   (linked tables analysed together)

output/index.html links to every analysis's report.
"""

import argparse
import sys
import warnings
from pathlib import Path

from autoanalysis.config import Config
from autoanalysis.loading import discover_datasets
from autoanalysis.pipeline import run_pipeline
from autoanalysis.report import write_index
from autoanalysis.wording import statement


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="folder of datasets (default: ./data)")
    parser.add_argument("--output", default="output", help="folder for results (default: ./output)")
    parser.add_argument("--only", action="append", metavar="NAME",
                        help="analyse only this dataset (repeat for several); others' results are kept")
    parser.add_argument("--config", help="optional JSON file overriding values in autoanalysis/config.py")
    args = parser.parse_args()

    # scipy warns when a test sees nearly identical values; such tests are
    # already caught by the effect-size and robustness checks.
    warnings.filterwarnings("ignore", message="Precision loss occurred")

    datasets = discover_datasets(args.data)
    if args.only:
        unknown = sorted(set(args.only) - {name for name, _ in datasets})
        if unknown:
            sys.exit(f"Unknown dataset(s) {unknown}. Found: {[name for name, _ in datasets]}")
        datasets = [(name, path) for name, path in datasets if name in args.only]

    output_root = Path(args.output)
    for name, path in datasets:
        print(f"\n=== {name}  ({path})")
        # A fresh config per dataset, so nothing set during one run leaks into the next.
        config = Config.from_file(args.config) if args.config else Config()
        run = run_pipeline(path, output_root / name, config)

        print(f"Tested {len(run['candidates'])} patterns: {len(run['accepted'])} passed, "
              f"{len(run['rejected'])} rejected, {len(run['skipped'])} not testable.")
        for finding in [f for f in run["accepted"] if f["in_top"]]:
            print(f"  {finding['rank']:>3}. [{finding['confidence']:<6}] {statement(finding)}")
        print(f"Report: {output_root / name / '06_report.html'}")

    write_index(output_root)
    print(f"\nAll analyses: {output_root / 'index.html'}")


if __name__ == "__main__":
    main()
