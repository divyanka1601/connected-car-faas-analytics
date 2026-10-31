"""
export_for_powerbi.py
=====================
Writes Power BI-ready CSV files into ./exports

  Star-schema tables (clean, typed, de-duplicated by the DuckDB staging layer):
    dim_users.csv          one row per vehicle owner
    dim_features.csv       OTA feature price list
    dim_date.csv           daily calendar table (Power BI needs a proper date table)
    fact_subscriptions.csv one row per user x feature subscription (+ monthly_price)

  SQL result tables (pre-computed by analytics_engine.py, all countries / models):
    mart_mrr_waterfall.csv, mart_nrr_trend.csv, mart_conversion_by_model.csv,
    mart_conversion_trend.csv, mart_funnel.csv, mart_cohort_matrix.csv,
    mart_churn_velocity.csv, mart_seasonal_churn.csv, mart_churn_reasons.csv,
    mart_feature_adoption.csv, mart_feature_segments.csv

Run:  python export_for_powerbi.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analytics_engine import AnalyticsEngine, ensure_data

EXPORT_DIR = Path(__file__).resolve().parent / "exports"


def write_csv(df: pd.DataFrame, name: str) -> None:
    """Write one DataFrame to exports/<name>.csv (ISO dates, skip empty frames)."""
    if df is None or df.empty:
        print(f"  SKIPPED {name}: no rows")
        return
    path = EXPORT_DIR / f"{name}.csv"
    df.to_csv(path, index=False, date_format="%Y-%m-%d")
    print(f"  {name + '.csv':<32} {len(df):>7,} rows")


def build_dim_date(engine: AnalyticsEngine) -> pd.DataFrame:
    """Daily calendar from the first subscription month up to the 'as of' date."""
    first = engine.con.execute("SELECT MIN(start_date) FROM stg_subs").fetchone()[0]
    start = pd.Timestamp(first).replace(day=1)
    dates = pd.date_range(start, pd.Timestamp(engine.as_of), freq="D")
    return pd.DataFrame({
        "date": dates,
        "year": dates.year,
        "month_number": dates.month,
        "month_name": dates.strftime("%b"),
        "month_start": dates.to_period("M").to_timestamp(),
        "year_month": dates.strftime("%Y-%m"),
        "quarter": ["Q" + str(q) for q in dates.quarter],
    })


def main() -> None:
    ensure_data()                                   # generate CSVs if missing
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    engine = AnalyticsEngine()
    print(f"Exporting to {EXPORT_DIR}  (data as of {engine.as_of})\n")

    # --- Star schema ---------------------------------------------------------
    print("Star-schema tables:")
    write_csv(engine.con.execute("SELECT * FROM stg_users ORDER BY user_id").df(), "dim_users")
    write_csv(engine.con.execute("SELECT * FROM stg_features ORDER BY feature_id").df(), "dim_features")
    write_csv(build_dim_date(engine), "dim_date")
    write_csv(engine.con.execute("""
        SELECT s.*, f.monthly_price
        FROM stg_subs s
        JOIN stg_features f ON f.feature_id = s.feature_id
        ORDER BY s.sub_id
    """).df(), "fact_subscriptions")

    # --- SQL marts (all countries and vehicle models) --------------------------
    print("\nSQL result tables:")
    snap = engine.snapshot()
    for name, frame in snap.items():
        if isinstance(frame, pd.DataFrame) and name != "segment_users":
            write_csv(frame, f"mart_{name}")

    print("\nDone. In Power BI: Get Data -> Text/CSV -> pick the files in /exports.")


if __name__ == "__main__":
    main()
    