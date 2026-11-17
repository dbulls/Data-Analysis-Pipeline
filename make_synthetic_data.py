"""Create a small synthetic dataset with known, planted patterns.

Used by the tests and as a demo. THIS IS INVENTED DATA: any "finding" on it
only shows that the pipeline recovers what was planted.

    python make_synthetic_data.py --out sample_data/shop

The two tables are linked, so they go together in one folder (one dataset).

Planted on purpose (the pipeline should FIND these):
    * amount per order rises ~35% over three years
    * amount differs by customer segment (Enterprise > SMB > Consumer); segment lives in customers.csv
    * longer delivery_days goes with lower satisfaction
    * a few customers place most of the orders
    * delivery_days drops suddenly from September 2023
    * quantity is higher every November and December
    * discount_pct is unusually high for the whole of March 2023
    * the number of orders per month grows

Planted traps (the pipeline should REJECT these):
    * noise_a, noise_b, channel, region, age, returned: pure noise, related to nothing
    * one absurd order amount (250,000) in August 2022: a one-record "spike"
    * total quantity per month rises only because there are more orders

Planted mess (the cleaning rules should handle and log these):
    stray whitespace, mixed letter case, two date formats, "N/A" tokens,
    unparseable numbers, exact duplicate rows, missing satisfaction scores.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BREAK_DATE = pd.Timestamp("2023-09-01")
ANOMALY_MONTH = "2023-03"
SPIKE_ORDER_DATE = pd.Timestamp("2022-08-15")


def make_tables(seed=7, n_customers=400, n_orders=6000):
    rng = np.random.default_rng(seed)

    # ---- customers --------------------------------------------------------
    customers = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(1, n_customers + 1)],
        "segment": rng.choice(["Consumer", "SMB", "Enterprise"], n_customers, p=[0.6, 0.3, 0.1]),
        "region": rng.choice(["North", "South", "East", "West"], n_customers),
        "age": rng.integers(18, 76, n_customers),
    })

    # ---- orders -----------------------------------------------------------
    days = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    progress = np.linspace(0, 1, len(days))                 # 0 at the start, 1 at the end
    day_weights = 1 + progress                              # order volume doubles
    day_index = np.sort(rng.choice(len(days), n_orders, p=day_weights / day_weights.sum()))
    order_date = days[day_index]
    t = progress[day_index]

    popularity = rng.pareto(1.1, n_customers) + 0.05        # a few very active customers
    customer_index = rng.choice(n_customers, n_orders, p=popularity / popularity.sum())
    segment = customers["segment"].to_numpy()[customer_index]
    segment_multiplier = pd.Series(segment).map({"Consumer": 1.0, "SMB": 1.3, "Enterprise": 2.0}).to_numpy()

    amount = rng.lognormal(4.0, 0.5, n_orders) * segment_multiplier * (1 + 0.35 * t)
    holiday = pd.Series(order_date).dt.month.isin([11, 12]).to_numpy()
    quantity = 1 + rng.poisson(np.where(holiday, 5.0, 3.0))
    delivery_days = np.clip(rng.normal(np.where(order_date < BREAK_DATE, 6.0, 4.5), 1.5), 1, None)
    satisfaction = np.clip(8 - 0.5 * delivery_days + rng.normal(0, 1.2, n_orders), 1, 10)
    in_anomaly_month = pd.Series(order_date).dt.strftime("%Y-%m").eq(ANOMALY_MONTH).to_numpy()
    discount_pct = np.clip(rng.normal(10, 3, n_orders) + np.where(in_anomaly_month, 6, 0), 0, None)

    orders = pd.DataFrame({
        "order_id": np.arange(100001, 100001 + n_orders),
        "customer_id": customers["customer_id"].to_numpy()[customer_index],
        "order_date": order_date.strftime("%Y-%m-%d"),
        "amount": amount.round(2),
        "quantity": quantity,
        "delivery_days": delivery_days.round(1),
        "satisfaction": satisfaction.round(1),
        "discount_pct": discount_pct.round(1),
        "channel": rng.choice(["Web", "Phone", "Store"], n_orders),
        "returned": rng.choice(["yes", "no"], n_orders, p=[0.08, 0.92]),
        "noise_a": rng.normal(50, 10, n_orders).round(2),
        "noise_b": rng.normal(0, 1, n_orders).round(3),
    })

    # Trap: one absurd order.
    spike_row = int(np.searchsorted(order_date, SPIKE_ORDER_DATE))
    orders.loc[spike_row, "amount"] = 250000.00

    # ---- mess -------------------------------------------------------------
    orders = orders.astype(object)
    customers = customers.astype(object)

    def pick(n):
        return rng.choice(n_orders, n, replace=False)

    rows = pick(240)                                        # second date format
    orders.loc[rows, "order_date"] = pd.to_datetime(orders.loc[rows, "order_date"]).dt.strftime("%m/%d/%Y")
    rows = pick(150)
    orders.loc[rows, "channel"] = "  " + orders.loc[rows, "channel"].astype(str) + " "
    rows = pick(100)
    orders.loc[rows, "channel"] = orders.loc[rows, "channel"].astype(str).str.lower()
    orders.loc[pick(700), "satisfaction"] = ""              # genuinely missing
    orders.loc[pick(120), "noise_a"] = "N/A"
    orders.loc[pick(3), "quantity"] = "abc"
    customers.loc[rng.choice(n_customers, 15, replace=False), "segment"] = "consumer "

    duplicates = orders.iloc[pick(5)]
    orders = pd.concat([orders, duplicates], ignore_index=True)
    return {"customers": customers, "orders": orders}


def write_tables(out_dir, seed=7):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in make_tables(seed).items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write the synthetic demo dataset.")
    parser.add_argument("--out", default="sample_data/shop")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(f"Wrote synthetic data to {write_tables(args.out, args.seed)}/")
