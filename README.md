# Connected Car "Feature-as-a-Service" Digital Monetization Engine

**Live demo:** https://connected-car-faas-analytics.onrender.com (free tier, so the first load after idle takes about a minute)

An analytics project that simulates how an automaker monetizes over-the-air software upgrades such as Highway Autopilot, Heated Seats and Performance Boost. It answers the questions a subscription-revenue team asks every month: how much recurring revenue are we making, who converts from trial, who churns and why, and which features earn their place in the catalogue.

> All data is synthetic, simulated through 30 Oct 2026. No real customers or brands.

## Business problem

Selling features by subscription turns a one-time car sale into recurring revenue, but only if customers convert from trial and stay. This project turns raw billing exports into the metrics leadership uses to decide pricing, bundling and retention spend:

| Question | Metric |
|---|---|
| How fast is recurring revenue growing? | MRR split into new, expansion and churned; ARR |
| Do existing customers grow or shrink? | Net and gross revenue retention (NRR, GRR) |
| Which customers convert from trial? | Trial-to-paid conversion by vehicle model |
| When and why do customers leave? | Cohort retention matrix, seasonal churn index, churn reasons |
| Which features earn their keep? | Feature Adoption Index (adoption vs revenue) |

## Key findings (from the generated dataset)

- Current MRR is about $33.3K (ARR about $400K) from 1,562 paying subscriptions.
- 42.7% of trials convert to paid. Volt GT drivers convert best (53%); Metro Compact worst (35%).
- Heated-feature churn peaks June to August at 1.7 to 1.9 times a normal month: customers cancel heated seats when summer arrives. A seasonal pause or discount would likely reduce this churn.
- Highway Autopilot has only 6% adoption but earns 33% of lifetime revenue: a premium niche, not a volume product.

## Architecture

```
mock_data_generator.py  ->  data/*.csv  ->  analytics_engine.py (DuckDB SQL)  ->  app.py (Streamlit)
                                                      |
                                                      +->  export_for_powerbi.py  ->  exports/*.csv
```

- **No database server.** DuckDB runs analytical SQL directly against CSV files in memory, so the app deploys on Render's free tier.
- **Three-layer SQL pipeline:** raw CSV (read as text, nothing silently mis-typed), cleaned staging tables, then one SQL query per business question.
- **Data cleaning in SQL:** mixed `YYYY-MM-DD` and `DD/MM/YYYY` dates, inconsistent country and status spellings, duplicate rows, orphan foreign keys and impossible date ranges. The sidebar shows raw vs cleaned row counts.
- **Self-checking:** the MRR waterfall includes a reconciliation column (opening + new + expansion - churned - closing) that must equal zero.

## SQL concepts used

- CTE chains (`WITH ...`) for readable multi-step logic
- Window functions: `LAG`, `ROW_NUMBER`, `RANK`, `PERCENT_RANK`, `FIRST_VALUE`, moving averages with `ROWS BETWEEN ...`
- Conditional aggregation with `FILTER (WHERE ...)`
- A calendar table built with `generate_series`, and a reusable SQL macro for date parsing
- Cohort analysis with right-censoring handled (only observable months are counted)

## Run locally

```bash
git clone https://github.com/divyanka1601/connected-car-faas-analytics
cd connected-car-faas-analytics
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

The CSVs are included in `data/`. To regenerate them, run `python mock_data_generator.py`.

## Deploy to Render (free tier)

1. Push this repo to your GitHub account.
2. On render.com choose **New + > Web Service** and connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Instance type: **Free**.

## Power BI version

`python export_for_powerbi.py` writes a star schema (`dim_users`, `dim_features`, `dim_date`, `fact_subscriptions`) and the SQL result tables into `exports/`. Load them in Power BI Desktop via Get Data > Text/CSV.

## Tech stack

Python, Pandas, NumPy, DuckDB, Streamlit, Plotly, Power BI (DAX)