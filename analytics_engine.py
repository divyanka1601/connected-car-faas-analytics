"""
analytics_engine.py
===================
Connected Car "Feature-as-a-Service" (FaaS) Digital Monetization Engine
-----------------------------------------------------------------------
A DuckDB-powered analytics layer that runs *heavy analytical SQL directly against
flat CSV files* - no database server, so the whole app fits on Render's free tier.

Architecture (a tiny "medallion" pipeline)
------------------------------------------
    RAW   CSV files (messy)            data/*.csv
      |   read_csv(all_varchar=true)   -> nothing is silently mis-typed
      v
    STAGING  stg_users / stg_features / stg_subs      (cleaned, typed, de-duplicated)
      |   + engine_params (as-of date) + month_spine (calendar table)
      v
    SCOPE    scoped_users / scoped_subs   (TEMP tables = sidebar filters applied once)
      v
    MARTS    one explicit SQL query per business question (see QUERIES below)

Key business definitions (documented assumptions)
-------------------------------------------------
* Every subscription starts as a free TRIAL of TRIAL_DAYS days (start_date = trial start).
* A subscription CONVERTED if it reached paid status: status = 'active', or it was
  cancelled AFTER the trial ended (end_date > trial_end_date).
* Paid period: paid_start_date = trial_end_date ; paid_end_date = end_date (if cancelled).
* MRR is a month-end snapshot of monthly_price for all paid subscriptions.
* NEW MRR       = first paid subscription of a customer (or a win-back after full churn).
  EXPANSION MRR = a paid subscription started while the customer already pays for another.
  CHURNED MRR   = paid subscriptions cancelled in the month.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
REQUIRED_FILES = ("users_dim.csv", "features_dim.csv", "subscription_fact.csv")
TRIAL_DAYS = 14          # must match the generator's trial length
CONVERSION_WINDOW = 30   # "converted within 30 days of trial expiry"


# =========================================================================== #
# 1. STAGING SQL  (raw CSV -> clean typed tables)
# =========================================================================== #

# A reusable macro: parses BOTH 'YYYY-MM-DD' and 'DD/MM/YYYY' text, returns NULL for junk.
# Regex guards avoid DuckDB guessing a day/month order on ambiguous strings.
SQL_MACRO_PARSE_DATE = r"""
CREATE OR REPLACE MACRO parse_date_str(x) AS (
    CASE
        WHEN regexp_matches(TRIM(x), '^\d{4}-\d{2}-\d{2}')
            THEN TRY_CAST(substr(TRIM(x), 1, 10) AS DATE)
        WHEN regexp_matches(TRIM(x), '^\d{2}/\d{2}/\d{4}$')
            THEN CAST(try_strptime(TRIM(x), '%d/%m/%Y') AS DATE)
        ELSE NULL
    END
);
"""

SQL_STG_USERS = r"""
CREATE OR REPLACE TABLE stg_users AS
WITH raw AS (
    SELECT
        UPPER(TRIM(user_id))                          AS user_id,
        parse_date_str(signup_date)                   AS signup_date,
        UPPER(TRIM(country))                          AS country_raw,
        NULLIF(TRIM(vehicle_model), '')               AS vehicle_model,
        LOWER(TRIM(tier))                             AS tier_raw
    FROM read_csv('__USERS_CSV__', header = true, all_varchar = true)
),
deduped AS (
    -- Duplicate users: keep the earliest signup record per user_id
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY signup_date NULLS LAST) AS rn
    FROM raw
    WHERE user_id IS NOT NULL AND user_id <> ''
)
SELECT
    user_id,
    signup_date,
    -- Standardise messy country spellings into one canonical label
    CASE
        WHEN country_raw IN ('US', 'USA', 'U.S.', 'UNITED STATES')          THEN 'United States'
        WHEN country_raw IN ('DE', 'DEUTSCHLAND', 'GERMANY')                THEN 'Germany'
        WHEN country_raw IN ('UK', 'GB', 'GREAT BRITAIN', 'UNITED KINGDOM') THEN 'United Kingdom'
        WHEN country_raw IN ('JP', 'JAPAN')                                 THEN 'Japan'
        WHEN country_raw IN ('CA', 'CANADA')                                THEN 'Canada'
        WHEN country_raw IN ('FR', 'FRANCE')                                THEN 'France'
        WHEN country_raw IN ('AU', 'AUSTRALIA')                             THEN 'Australia'
        WHEN country_raw IN ('NO', 'NORWAY')                                THEN 'Norway'
        WHEN country_raw IS NULL OR country_raw = ''                        THEN 'Unknown'
        ELSE 'Other'
    END AS country,
    COALESCE(vehicle_model, 'Unknown') AS vehicle_model,
    CASE tier_raw
        WHEN 'standard'    THEN 'Standard'
        WHEN 'premium'     THEN 'Premium'
        WHEN 'performance' THEN 'Performance'
        ELSE 'Unknown'
    END AS tier
FROM deduped
WHERE rn = 1;
"""

SQL_STG_FEATURES = r"""
CREATE OR REPLACE TABLE stg_features AS
WITH raw AS (
    SELECT
        UPPER(TRIM(feature_id))                         AS feature_id,
        TRIM(feature_name)                              AS feature_name,
        COALESCE(TRY_CAST(monthly_price AS DOUBLE), 0)  AS monthly_price,
        COALESCE(TRY_CAST(upfront_purchase_price AS DOUBLE), 0) AS upfront_purchase_price
    FROM read_csv('__FEATURES_CSV__', header = true, all_varchar = true)
),
deduped AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY feature_id ORDER BY feature_name) AS rn
    FROM raw
    WHERE feature_id IS NOT NULL AND feature_id <> ''
)
SELECT feature_id, feature_name, monthly_price, upfront_purchase_price
FROM deduped
WHERE rn = 1;
"""

SQL_STG_SUBS = r"""
CREATE OR REPLACE TABLE stg_subs AS
WITH raw AS (
    SELECT
        TRIM(sub_id)                                    AS sub_id,
        UPPER(TRIM(user_id))                            AS user_id,
        UPPER(TRIM(feature_id))                         AS feature_id,
        parse_date_str(start_date)                      AS start_date,
        parse_date_str(end_date)                        AS end_date,
        -- 'Canceled', ' CANCELLED ', 'cancelled' -> 'cancelled'
        CASE WHEN LOWER(TRIM(status)) IN ('canceled', 'cancelled') THEN 'cancelled'
             ELSE LOWER(TRIM(status)) END               AS status,
        -- blank / N/A style placeholders become real NULLs
        CASE WHEN churn_reason IS NULL
                  OR LOWER(TRIM(churn_reason)) IN ('', 'n/a', 'na', 'null', 'none', 'nan')
             THEN NULL ELSE TRIM(churn_reason) END      AS churn_reason
    FROM read_csv('__SUBS_CSV__', header = true, all_varchar = true)
),
deduped AS (
    -- Exact duplicate exports: keep one row per sub_id
    SELECT *, ROW_NUMBER() OVER (PARTITION BY sub_id ORDER BY start_date NULLS LAST) AS rn
    FROM raw
    WHERE sub_id IS NOT NULL AND sub_id <> ''
),
valid AS (
    -- Referential integrity + logical validity checks
    SELECT d.sub_id, d.user_id, d.feature_id, d.start_date,
           CASE WHEN d.status IN ('active', 'trial') THEN NULL ELSE d.end_date END AS end_date,
           d.status, d.churn_reason
    FROM deduped d
    JOIN stg_users    u ON u.user_id    = d.user_id        -- drops orphan user_ids
    JOIN stg_features f ON f.feature_id = d.feature_id     -- drops orphan feature_ids
    WHERE d.rn = 1
      AND d.start_date IS NOT NULL
      AND d.status IN ('trial', 'active', 'cancelled')
      AND NOT (d.status = 'cancelled' AND d.end_date IS NULL)
      AND (d.end_date IS NULL OR d.end_date >= d.start_date)   -- drops end < start
),
with_trial AS (
    SELECT *, start_date + __TRIAL_DAYS__ AS trial_end_date FROM valid
),
flagged AS (
    SELECT *,
           CASE WHEN status = 'active'
                  OR (status = 'cancelled' AND end_date > trial_end_date)
                THEN 1 ELSE 0 END AS is_converted
    FROM with_trial
)
SELECT
    sub_id, user_id, feature_id, start_date, end_date, status,
    CASE WHEN status = 'cancelled' THEN COALESCE(churn_reason, 'Unspecified') END AS churn_reason,
    trial_end_date,
    is_converted,
    CASE WHEN is_converted = 1 THEN trial_end_date END                         AS paid_start_date,
    CASE WHEN is_converted = 1 AND status = 'cancelled' THEN end_date END      AS paid_end_date
FROM flagged;
"""

# Global parameters derived from the data itself ("today" = latest observed date)
SQL_ENGINE_PARAMS = r"""
CREATE OR REPLACE TABLE engine_params AS
SELECT
    CAST(COALESCE(GREATEST(MAX(start_date), COALESCE(MAX(end_date), MAX(start_date))),
                  CURRENT_DATE) AS DATE)       AS as_of,
    __TRIAL_DAYS__                             AS trial_days,
    __CONVERSION_WINDOW__                      AS conversion_window_days
FROM stg_subs;
"""

# Calendar table: one row per month, with a month-end capped at the as-of date.
SQL_MONTH_SPINE = r"""
CREATE OR REPLACE TABLE month_spine AS
SELECT
    CAST(t.m AS DATE)                                                      AS month_start,
    LEAST(last_day(CAST(t.m AS DATE)), DATE '__AS_OF__')                   AS month_end,
    (last_day(CAST(t.m AS DATE)) <= DATE '__AS_OF__')                      AS is_full_month
FROM generate_series(DATE '__FIRST_MONTH__', DATE '__AS_OF__', INTERVAL 1 MONTH) AS t(m);
"""

# Row-count reconciliation raw -> staged (surfaced in the dashboard sidebar)
SQL_DATA_QUALITY = r"""
SELECT 'Users'         AS dataset,
       (SELECT COUNT(*) FROM read_csv('__USERS_CSV__',    header = true, all_varchar = true)) AS raw_rows,
       (SELECT COUNT(*) FROM stg_users)                                                       AS clean_rows
UNION ALL
SELECT 'Features',
       (SELECT COUNT(*) FROM read_csv('__FEATURES_CSV__', header = true, all_varchar = true)),
       (SELECT COUNT(*) FROM stg_features)
UNION ALL
SELECT 'Subscriptions',
       (SELECT COUNT(*) FROM read_csv('__SUBS_CSV__',     header = true, all_varchar = true)),
       (SELECT COUNT(*) FROM stg_subs);
"""

# =========================================================================== #
# 2. SCOPE SQL  (sidebar filters applied once per request into TEMP tables)
# =========================================================================== #
SQL_SCOPED_USERS = r"""
CREATE OR REPLACE TEMP TABLE scoped_users AS
SELECT u.*
FROM stg_users u
WHERE u.country       IN (SELECT country       FROM f_countries)
  AND u.vehicle_model IN (SELECT vehicle_model FROM f_models);
"""

SQL_SCOPED_SUBS = r"""
CREATE OR REPLACE TEMP TABLE scoped_subs AS
SELECT
    s.*,
    u.country, u.vehicle_model, u.tier, u.signup_date,
    f.feature_name, f.monthly_price, f.upfront_purchase_price,
    -- Flag the seasonal product family used in the seasonality analysis
    CASE WHEN f.feature_name ILIKE '%heated%' THEN 'Heated (seasonal)'
         ELSE 'Year-round' END AS seasonal_group
FROM stg_subs s
JOIN scoped_users u ON u.user_id    = s.user_id
JOIN stg_features f ON f.feature_id = s.feature_id;
"""

# =========================================================================== #
# 3. MART SQL  (one explicit query per business question)
# =========================================================================== #

# --- 3.1  MRR waterfall: new / expansion / churned, with window functions ----
SQL_MRR_WATERFALL = r"""
WITH paid_subs AS (
    -- Every converted subscription, with per-customer ordering via window functions
    SELECT
        s.sub_id, s.user_id, s.feature_id, s.monthly_price,
        s.paid_start_date, s.paid_end_date,
        ROW_NUMBER() OVER (PARTITION BY s.user_id
                           ORDER BY s.paid_start_date, s.sub_id)            AS user_sub_seq,
        -- Latest paid_end among this customer's EARLIER subscriptions
        -- (open-ended subs count as "9999-12-31" = still paying)
        MAX(COALESCE(s.paid_end_date, DATE '9999-12-31'))
            OVER (PARTITION BY s.user_id
                  ORDER BY s.paid_start_date, s.sub_id
                  ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)         AS prior_latest_end
    FROM scoped_subs s
    WHERE s.is_converted = 1
),
classified AS (
    -- Expansion = added a feature while an earlier paid subscription was still live
    SELECT *,
           CASE WHEN user_sub_seq > 1 AND prior_latest_end > paid_start_date
                THEN 'expansion' ELSE 'new' END AS mrr_type
    FROM paid_subs
),
monthly AS (
    SELECT
        ms.month_start, ms.month_end, ms.is_full_month,
        COALESCE(SUM(c.monthly_price) FILTER (
            WHERE c.mrr_type = 'new'
              AND c.paid_start_date BETWEEN ms.month_start AND ms.month_end), 0)    AS new_mrr,
        COALESCE(SUM(c.monthly_price) FILTER (
            WHERE c.mrr_type = 'expansion'
              AND c.paid_start_date BETWEEN ms.month_start AND ms.month_end), 0)    AS expansion_mrr,
        COALESCE(SUM(c.monthly_price) FILTER (
            WHERE c.paid_end_date BETWEEN ms.month_start AND ms.month_end), 0)      AS churned_mrr,
        COALESCE(SUM(c.monthly_price) FILTER (
            WHERE c.paid_start_date <= ms.month_end
              AND (c.paid_end_date IS NULL OR c.paid_end_date > ms.month_end)), 0)  AS ending_mrr,
        COUNT(*) FILTER (
            WHERE c.paid_start_date <= ms.month_end
              AND (c.paid_end_date IS NULL OR c.paid_end_date > ms.month_end))      AS ending_paid_subs,
        COUNT(*) FILTER (
            WHERE c.paid_end_date BETWEEN ms.month_start AND ms.month_end)          AS churned_subs
    FROM month_spine ms
    LEFT JOIN classified c ON c.paid_start_date <= ms.month_end
    GROUP BY ms.month_start, ms.month_end, ms.is_full_month
),
with_lag AS (
    -- LAG gives last month's closing MRR = this month's opening MRR
    SELECT *,
           COALESCE(LAG(ending_mrr) OVER (ORDER BY month_start), 0) AS beginning_mrr
    FROM monthly
)
SELECT
    month_start, month_end, is_full_month,
    ROUND(beginning_mrr, 2)                                    AS beginning_mrr,
    ROUND(new_mrr, 2)                                          AS new_mrr,
    ROUND(expansion_mrr, 2)                                    AS expansion_mrr,
    ROUND(churned_mrr, 2)                                      AS churned_mrr,
    ROUND(ending_mrr, 2)                                       AS ending_mrr,
    ROUND(new_mrr + expansion_mrr - churned_mrr, 2)            AS net_new_mrr,
    ROUND(ending_mrr * 12, 2)                                  AS arr,
    ending_paid_subs,
    churned_subs,
    ROUND(100.0 * (ending_mrr - beginning_mrr) / NULLIF(beginning_mrr, 0), 2) AS mom_growth_pct,
    ROUND(100.0 * churned_mrr / NULLIF(beginning_mrr, 0), 2)                  AS revenue_churn_pct,
    -- Integrity check: opening + new + expansion - churn must equal closing
    ROUND(beginning_mrr + new_mrr + expansion_mrr - churned_mrr - ending_mrr, 2) AS reconciliation_gap
FROM with_lag
ORDER BY month_start;
"""

# --- 3.2  Net / Gross revenue retention (trailing 12 months, per customer) ----
SQL_NRR_TREND = r"""
WITH paid_subs AS (
    SELECT user_id, monthly_price, paid_start_date, paid_end_date
    FROM scoped_subs
    WHERE is_converted = 1
),
paying_users AS (SELECT DISTINCT user_id FROM paid_subs),
user_month AS (
    -- Customer x month grid of month-end MRR (zeros kept so LAG(12) lines up)
    SELECT pu.user_id, ms.month_start, ms.is_full_month,
           COALESCE(SUM(p.monthly_price), 0) AS user_mrr
    FROM paying_users pu
    CROSS JOIN month_spine ms
    LEFT JOIN paid_subs p
           ON p.user_id = pu.user_id
          AND p.paid_start_date <= ms.month_end
          AND (p.paid_end_date IS NULL OR p.paid_end_date > ms.month_end)
    GROUP BY pu.user_id, ms.month_start, ms.is_full_month
),
lagged AS (
    SELECT *,
           LAG(user_mrr, 12) OVER (PARTITION BY user_id ORDER BY month_start) AS mrr_12m_ago
    FROM user_month
)
SELECT
    month_start, is_full_month,
    -- NRR: today's revenue from customers who paid 12 months ago / their revenue then
    ROUND(100.0 * SUM(CASE WHEN mrr_12m_ago > 0 THEN user_mrr END)
          / NULLIF(SUM(CASE WHEN mrr_12m_ago > 0 THEN mrr_12m_ago END), 0), 2)            AS nrr_pct,
    -- GRR: same, but expansion is capped (can only lose revenue)
    ROUND(100.0 * SUM(CASE WHEN mrr_12m_ago > 0 THEN LEAST(user_mrr, mrr_12m_ago) END)
          / NULLIF(SUM(CASE WHEN mrr_12m_ago > 0 THEN mrr_12m_ago END), 0), 2)            AS grr_pct,
    COUNT(CASE WHEN mrr_12m_ago > 0 THEN 1 END)                                           AS base_customers
FROM lagged
GROUP BY month_start, is_full_month
HAVING COUNT(CASE WHEN mrr_12m_ago > 0 THEN 1 END) > 0
ORDER BY month_start;
"""

# --- 3.3  Trial-to-paid conversion by vehicle model ---------------------------
SQL_CONVERSION_BY_MODEL = r"""
WITH matured AS (
    -- Only trials whose 30-day post-expiry window has fully elapsed (no censoring bias)
    SELECT s.vehicle_model, s.user_id, s.trial_end_date, s.is_converted, s.paid_start_date
    FROM scoped_subs s
    CROSS JOIN engine_params p
    WHERE s.status IN ('active', 'cancelled')
      AND s.trial_end_date + p.conversion_window_days <= p.as_of
),
flagged AS (
    SELECT m.*,
           CASE WHEN m.is_converted = 1
                 AND m.paid_start_date <= m.trial_end_date + (SELECT conversion_window_days FROM engine_params)
                THEN 1 ELSE 0 END AS converted_in_window
    FROM matured m
),
by_model AS (
    SELECT vehicle_model,
           COUNT(*)                  AS expired_trials,
           SUM(converted_in_window)  AS conversions,
           COUNT(DISTINCT user_id)   AS trialing_users
    FROM flagged
    GROUP BY vehicle_model
)
SELECT
    vehicle_model, expired_trials, conversions, trialing_users,
    ROUND(100.0 * conversions / NULLIF(expired_trials, 0), 2)                         AS conversion_rate_pct,
    ROUND(100.0 * SUM(conversions) OVER () / NULLIF(SUM(expired_trials) OVER (), 0), 2) AS overall_rate_pct,
    ROUND(100.0 * conversions / NULLIF(expired_trials, 0)
          - 100.0 * SUM(conversions) OVER () / NULLIF(SUM(expired_trials) OVER (), 0), 2) AS vs_overall_pp,
    RANK() OVER (ORDER BY 1.0 * conversions / NULLIF(expired_trials, 0) DESC NULLS LAST) AS conversion_rank
FROM by_model
ORDER BY conversion_rank;
"""

# --- 3.4  Conversion trend by trial-start month (LAG + moving average) --------
SQL_CONVERSION_TREND = r"""
WITH matured AS (
    SELECT date_trunc('month', s.start_date) AS trial_month,
           s.is_converted
    FROM scoped_subs s
    CROSS JOIN engine_params p
    WHERE s.status IN ('active', 'cancelled')
      AND s.trial_end_date + p.conversion_window_days <= p.as_of
),
monthly AS (
    SELECT trial_month,
           COUNT(*)          AS expired_trials,
           SUM(is_converted) AS conversions,
           100.0 * SUM(is_converted) / NULLIF(COUNT(*), 0) AS conv_rate
    FROM matured
    GROUP BY trial_month
)
SELECT
    CAST(trial_month AS DATE)                                                    AS trial_month,
    expired_trials, conversions,
    ROUND(conv_rate, 2)                                                          AS conversion_rate_pct,
    ROUND(conv_rate - LAG(conv_rate) OVER (ORDER BY trial_month), 2)             AS mom_change_pp,
    ROUND(AVG(conv_rate) OVER (ORDER BY trial_month
                               ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2)     AS conversion_rate_3m_avg
FROM monthly
ORDER BY trial_month;
"""

# --- 3.5  Subscription funnel -------------------------------------------------
SQL_FUNNEL = r"""
WITH stages AS (
    SELECT 1 AS stage_order, 'Trials started' AS stage, COUNT(*) AS subs
    FROM scoped_subs
    UNION ALL
    SELECT 2, 'Trial window closed', COUNT(*)
    FROM scoped_subs s CROSS JOIN engine_params p
    WHERE s.trial_end_date <= p.as_of
    UNION ALL
    SELECT 3, 'Converted to paid', COUNT(*)
    FROM scoped_subs WHERE is_converted = 1
    UNION ALL
    SELECT 4, 'Paid for 90+ days', COUNT(*)
    FROM scoped_subs s CROSS JOIN engine_params p
    WHERE s.is_converted = 1
      AND COALESCE(s.paid_end_date, p.as_of) >= s.paid_start_date + 90
    UNION ALL
    SELECT 5, 'Still paying today', COUNT(*)
    FROM scoped_subs WHERE status = 'active'
)
SELECT
    stage_order, stage, subs,
    ROUND(100.0 * subs / NULLIF(FIRST_VALUE(subs) OVER (ORDER BY stage_order), 0), 1)  AS pct_of_started,
    ROUND(100.0 * subs / NULLIF(LAG(subs) OVER (ORDER BY stage_order), 0), 1)          AS pct_of_prev_stage
FROM stages
ORDER BY stage_order;
"""

# --- 3.6  Customer churn cohort matrix (retention by months since conversion) -
SQL_COHORT_MATRIX = r"""
WITH paid AS (
    SELECT
        s.sub_id,
        date_trunc('month', s.paid_start_date) AS cohort_month,
        -- Months the subscription actually stayed paid (or has been paying until today)
        CAST(FLOOR(date_diff('day', s.paid_start_date,
                             COALESCE(s.paid_end_date, p.as_of)) / 30.4375) AS INTEGER) AS tenure_m,
        -- Months we could possibly have observed for this subscription
        CAST(FLOOR(date_diff('day', s.paid_start_date, p.as_of) / 30.4375) AS INTEGER)  AS observable_m
    FROM scoped_subs s
    CROSS JOIN engine_params p
    WHERE s.is_converted = 1
),
periods AS (SELECT k FROM generate_series(0, 24) AS t(k)),
matrix AS (
    SELECT
        p.cohort_month,
        pe.k AS month_offset,
        COUNT(*)                                                          AS cohort_size,
        COUNT(*) FILTER (WHERE p.observable_m >= pe.k)                    AS observed_subs,
        COUNT(*) FILTER (WHERE p.observable_m >= pe.k
                           AND p.tenure_m     >= pe.k)                    AS retained_subs
    FROM paid p
    CROSS JOIN periods pe
    GROUP BY p.cohort_month, pe.k
),
rates AS (
    SELECT *,
           ROUND(100.0 * retained_subs / NULLIF(observed_subs, 0), 1) AS retention_pct
    FROM matrix
    WHERE observed_subs > 0
)
SELECT
    CAST(cohort_month AS DATE) AS cohort_month,
    month_offset, cohort_size, observed_subs, retained_subs, retention_pct,
    -- Cancellation velocity: retention points lost versus the previous month of life
    ROUND(LAG(retention_pct) OVER (PARTITION BY cohort_month ORDER BY month_offset)
          - retention_pct, 1) AS period_drop_pp
FROM rates
ORDER BY cohort_month, month_offset;
"""

# --- 3.7  Monthly cancellation velocity (moving average, MoM change, rank) ----
SQL_CHURN_VELOCITY = r"""
WITH paid AS (
    SELECT monthly_price, paid_start_date, paid_end_date
    FROM scoped_subs
    WHERE is_converted = 1
),
monthly AS (
    SELECT
        ms.month_start, ms.is_full_month,
        COUNT(*) FILTER (WHERE p.paid_start_date < ms.month_start
                           AND (p.paid_end_date IS NULL OR p.paid_end_date >= ms.month_start)) AS opening_subs,
        COUNT(*) FILTER (WHERE p.paid_end_date BETWEEN ms.month_start AND ms.month_end)       AS cancellations,
        COALESCE(SUM(p.monthly_price) FILTER (
            WHERE p.paid_end_date BETWEEN ms.month_start AND ms.month_end), 0)                AS churned_mrr
    FROM month_spine ms
    LEFT JOIN paid p ON p.paid_start_date <= ms.month_end
    GROUP BY ms.month_start, ms.is_full_month
)
SELECT
    month_start, is_full_month, opening_subs, cancellations,
    ROUND(churned_mrr, 2)                                                           AS churned_mrr,
    ROUND(100.0 * cancellations / NULLIF(opening_subs, 0), 2)                       AS churn_rate_pct,
    ROUND(AVG(100.0 * cancellations / NULLIF(opening_subs, 0)) OVER (
          ORDER BY month_start ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2)        AS churn_rate_3m_avg,
    cancellations - LAG(cancellations) OVER (ORDER BY month_start)                  AS cancellations_mom_change,
    RANK() OVER (ORDER BY 1.0 * cancellations / NULLIF(opening_subs, 0) DESC NULLS LAST) AS churn_severity_rank
FROM monthly
WHERE opening_subs > 0
ORDER BY month_start;
"""

# --- 3.8  Seasonal churn index (e.g. heated seats cancelled in summer) --------
SQL_SEASONAL_CHURN = r"""
WITH paid AS (
    SELECT seasonal_group, paid_start_date, paid_end_date
    FROM scoped_subs
    WHERE is_converted = 1
),
exposure AS (
    -- For each full month and product family: opening base vs cancellations
    SELECT
        ms.month_start, p.seasonal_group,
        COUNT(*) FILTER (WHERE p.paid_start_date < ms.month_start
                           AND (p.paid_end_date IS NULL OR p.paid_end_date >= ms.month_start)) AS opening_subs,
        COUNT(*) FILTER (WHERE p.paid_end_date BETWEEN ms.month_start AND ms.month_end)       AS cancellations
    FROM month_spine ms
    JOIN paid p ON p.paid_start_date <= ms.month_end
    WHERE ms.is_full_month
    GROUP BY ms.month_start, p.seasonal_group
),
by_calendar_month AS (
    -- Collapse years: January 2025 and January 2026 both feed "January"
    SELECT seasonal_group,
           month(month_start)   AS calendar_month,
           SUM(cancellations)   AS cancellations,
           SUM(opening_subs)    AS opening_sub_months
    FROM exposure
    GROUP BY seasonal_group, month(month_start)
),
rates AS (
    SELECT *,
           1.0 * cancellations / NULLIF(opening_sub_months, 0) AS monthly_rate,
           1.0 * SUM(cancellations)       OVER (PARTITION BY seasonal_group)
               / NULLIF(SUM(opening_sub_months) OVER (PARTITION BY seasonal_group), 0) AS group_avg_rate
    FROM by_calendar_month
)
SELECT
    seasonal_group, calendar_month,
    strftime(make_date(2000, calendar_month, 1), '%b')               AS month_label,
    cancellations, opening_sub_months,
    ROUND(100.0 * monthly_rate, 2)                                   AS churn_rate_pct,
    -- 100 = the family's average month; 300 = churn running 3x normal
    ROUND(100.0 * monthly_rate / NULLIF(group_avg_rate, 0), 1)       AS seasonal_index,
    CASE WHEN monthly_rate > 1.25 * group_avg_rate THEN 'Peak churn'
         WHEN monthly_rate < 0.75 * group_avg_rate THEN 'Low churn'
         ELSE 'Normal' END                                           AS season_flag
FROM rates
ORDER BY seasonal_group, calendar_month;
"""

# --- 3.9  Churn reasons --------------------------------------------------------
SQL_CHURN_REASONS = r"""
WITH cancelled AS (
    SELECT
        CASE WHEN is_converted = 1 THEN 'Post-trial (paid)' ELSE 'During trial' END AS phase,
        churn_reason, monthly_price, is_converted
    FROM scoped_subs
    WHERE status = 'cancelled'
),
agg AS (
    SELECT phase, churn_reason,
           COUNT(*) AS cancellations,
           SUM(CASE WHEN is_converted = 1 THEN monthly_price ELSE 0 END) AS lost_mrr
    FROM cancelled
    GROUP BY phase, churn_reason
)
SELECT
    phase, churn_reason, cancellations, ROUND(lost_mrr, 2) AS lost_mrr,
    ROUND(100.0 * cancellations / NULLIF(SUM(cancellations) OVER (), 0), 1)                AS pct_of_all,
    ROUND(100.0 * cancellations / NULLIF(SUM(cancellations) OVER (PARTITION BY phase), 0), 1) AS pct_of_phase,
    ROW_NUMBER() OVER (PARTITION BY phase ORDER BY cancellations DESC, churn_reason)       AS reason_rank
FROM agg
ORDER BY phase, reason_rank;
"""

# --- 3.10 Feature Adoption Index ----------------------------------------------
SQL_FEATURE_ADOPTION = r"""
WITH population AS (SELECT COUNT(*) AS total_users FROM scoped_users),
feature_stats AS (
    SELECT
        s.feature_id, s.feature_name, s.monthly_price, s.upfront_purchase_price,
        COUNT(*)                                                       AS trials_started,
        SUM(s.is_converted)                                            AS paid_conversions,
        COUNT(DISTINCT CASE WHEN s.is_converted = 1 THEN s.user_id END) AS adopters,
        COUNT(*) FILTER (WHERE s.status = 'active')                    AS active_paid_subs,
        COALESCE(SUM(s.monthly_price) FILTER (WHERE s.status = 'active'), 0) AS current_mrr,
        -- Realised recurring revenue to date = price x months actually paid
        COALESCE(SUM(CASE WHEN s.is_converted = 1
              THEN s.monthly_price
                   * date_diff('day', s.paid_start_date, COALESCE(s.paid_end_date, p.as_of)) / 30.4375
              END), 0)                                                 AS lifetime_revenue
    FROM scoped_subs s
    CROSS JOIN engine_params p
    GROUP BY s.feature_id, s.feature_name, s.monthly_price, s.upfront_purchase_price
),
scored AS (
    SELECT
        fs.*, pop.total_users,
        ROUND(100.0 * adopters / NULLIF(pop.total_users, 0), 2)           AS adoption_rate_pct,
        ROUND(100.0 * paid_conversions / NULLIF(trials_started, 0), 2)    AS trial_conversion_pct,
        ROUND(lifetime_revenue / NULLIF(adopters, 0), 2)                  AS revenue_per_adopter,
        RANK() OVER (ORDER BY adopters DESC)                              AS adoption_rank,
        RANK() OVER (ORDER BY lifetime_revenue DESC)                      AS revenue_rank,
        PERCENT_RANK() OVER (ORDER BY adopters)                           AS adoption_pctile,
        PERCENT_RANK() OVER (ORDER BY lifetime_revenue)                   AS revenue_pctile,
        ROUND(100.0 * lifetime_revenue / NULLIF(SUM(lifetime_revenue) OVER (), 0), 1) AS revenue_share_pct,
        AVG(adopters)         OVER () AS avg_adopters,
        AVG(lifetime_revenue) OVER () AS avg_revenue
    FROM feature_stats fs
    CROSS JOIN population pop
)
SELECT
    feature_id, feature_name, monthly_price, upfront_purchase_price,
    trials_started, paid_conversions, adopters, active_paid_subs,
    ROUND(current_mrr, 2)        AS current_mrr,
    ROUND(lifetime_revenue, 2)   AS lifetime_revenue,
    adoption_rate_pct, trial_conversion_pct, revenue_per_adopter, revenue_share_pct,
    adoption_rank, revenue_rank,
    -- Composite 0-100 index: half adoption frequency, half revenue contribution
    ROUND(100.0 * (0.5 * adoption_pctile + 0.5 * revenue_pctile), 1) AS adoption_index,
    adoption_rank - revenue_rank                                     AS rank_gap,   -- >0 = earns more than its popularity suggests
    CASE WHEN adopters >= avg_adopters AND lifetime_revenue >= avg_revenue THEN 'Star'
         WHEN adopters >= avg_adopters AND lifetime_revenue <  avg_revenue THEN 'Volume driver'
         WHEN adopters <  avg_adopters AND lifetime_revenue >= avg_revenue THEN 'Premium niche'
         ELSE 'Long tail' END                                        AS quadrant
FROM scored
ORDER BY adoption_index DESC, feature_name;
"""

# --- 3.11 Feature x customer-segment grain (tier x vehicle model) -------------
SQL_FEATURE_SEGMENTS = r"""
SELECT
    s.tier, s.vehicle_model, s.feature_name,
    COUNT(*)                                                         AS trials_started,
    SUM(s.is_converted)                                              AS paid_conversions,
    COUNT(DISTINCT CASE WHEN s.is_converted = 1 THEN s.user_id END)  AS adopters,
    ROUND(COALESCE(SUM(s.monthly_price) FILTER (WHERE s.status = 'active'), 0), 2) AS current_mrr,
    ROUND(COALESCE(SUM(CASE WHEN s.is_converted = 1
          THEN s.monthly_price
               * date_diff('day', s.paid_start_date, COALESCE(s.paid_end_date, p.as_of)) / 30.4375
          END), 0), 2)                                               AS lifetime_revenue
FROM scoped_subs s
CROSS JOIN engine_params p
GROUP BY s.tier, s.vehicle_model, s.feature_name
ORDER BY s.tier, s.vehicle_model, current_mrr DESC;
"""

SQL_SEGMENT_USERS = r"""
SELECT tier, vehicle_model, COUNT(*) AS segment_users
FROM scoped_users
GROUP BY tier, vehicle_model
ORDER BY tier, vehicle_model;
"""

# Registry: result name -> SQL (keeps snapshot() tidy and every query explicit above)
QUERIES: dict[str, str] = {
    "mrr_waterfall": SQL_MRR_WATERFALL,
    "nrr_trend": SQL_NRR_TREND,
    "conversion_by_model": SQL_CONVERSION_BY_MODEL,
    "conversion_trend": SQL_CONVERSION_TREND,
    "funnel": SQL_FUNNEL,
    "cohort_matrix": SQL_COHORT_MATRIX,
    "churn_velocity": SQL_CHURN_VELOCITY,
    "seasonal_churn": SQL_SEASONAL_CHURN,
    "churn_reasons": SQL_CHURN_REASONS,
    "feature_adoption": SQL_FEATURE_ADOPTION,
    "feature_segments": SQL_FEATURE_SEGMENTS,
    "segment_users": SQL_SEGMENT_USERS,
}


# =========================================================================== #
# 4. Python orchestration
# =========================================================================== #
def ensure_data(data_dir: Path | str = DATA_DIR) -> None:
    """Generate the CSVs on first boot if they are not in the repo (handy on Render)."""
    data_dir = Path(data_dir)
    if all((data_dir / f).exists() for f in REQUIRED_FILES):
        return
    from mock_data_generator import generate_all  # local import: only needed on first run
    generate_all(data_dir)


def _last_full_months(df: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    """Return the last n *complete* months (the current month is partial)."""
    if df is None or df.empty or "is_full_month" not in df.columns:
        return pd.DataFrame()
    return df[df["is_full_month"].astype(bool)].tail(n)


def _safe_float(x, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


def compute_kpis(snap: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Boil the SQL marts down to executive headline numbers (defensive on empties)."""
    wf, nrr = snap.get("mrr_waterfall"), snap.get("nrr_trend")
    conv = snap.get("conversion_by_model")
    kpis = {k: float("nan") for k in
            ("mrr", "arr", "mom_growth_pct", "revenue_churn_pct", "nrr_pct", "grr_pct",
             "paying_subs", "trial_conversion_pct")}

    if wf is not None and not wf.empty:
        last = wf.iloc[-1]                                   # incl. partial month = "right now"
        kpis["mrr"] = _safe_float(last["ending_mrr"])
        kpis["arr"] = _safe_float(last["arr"])
        kpis["paying_subs"] = _safe_float(last["ending_paid_subs"])
        full = _last_full_months(wf, 1)
        if not full.empty:
            kpis["mom_growth_pct"] = _safe_float(full.iloc[-1]["mom_growth_pct"])
        # Stable churn KPI: average of the last 3 complete months
        recent = _last_full_months(wf, 3)["revenue_churn_pct"].dropna()
        if not recent.empty:
            kpis["revenue_churn_pct"] = float(recent.mean())
    if nrr is not None and not nrr.empty:
        full = _last_full_months(nrr, 1)
        if not full.empty:
            kpis["nrr_pct"] = _safe_float(full.iloc[-1]["nrr_pct"])
            kpis["grr_pct"] = _safe_float(full.iloc[-1]["grr_pct"])
    if conv is not None and not conv.empty:
        kpis["trial_conversion_pct"] = _safe_float(conv.iloc[0]["overall_rate_pct"])
    return kpis


class AnalyticsEngine:
    """
    Owns one in-memory DuckDB database. Staging tables are built once at start-up;
    each dashboard request opens a lightweight cursor, applies the sidebar filters
    into TEMP tables and runs every mart query.
    """

    def __init__(self, data_dir: Path | str = DATA_DIR, memory_limit: str = "256MB"):
        self.data_dir = Path(data_dir)
        missing = [f for f in REQUIRED_FILES if not (self.data_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing data files in {self.data_dir}: {missing}. "
                "Run `python mock_data_generator.py` first."
            )
        self.con = duckdb.connect(database=":memory:")
        # Keep DuckDB frugal so the whole app stays under Render's 512 MB limit
        self.con.execute(f"SET memory_limit = '{memory_limit}'")
        self.con.execute("SET threads TO 2")
        self.as_of: date = date.today()
        self._build_staging()

    # ---- helpers ------------------------------------------------------------
    def _sql(self, template: str) -> str:
        """Fill the __PLACEHOLDERS__ used by the staging SQL (paths, constants)."""
        def q(name: str) -> str:  # escape single quotes in file paths
            return str(self.data_dir / name).replace("'", "''")
        return (template
                .replace("__USERS_CSV__", q("users_dim.csv"))
                .replace("__FEATURES_CSV__", q("features_dim.csv"))
                .replace("__SUBS_CSV__", q("subscription_fact.csv"))
                .replace("__TRIAL_DAYS__", str(int(TRIAL_DAYS)))
                .replace("__CONVERSION_WINDOW__", str(int(CONVERSION_WINDOW))))

    def _build_staging(self) -> None:
        con = self.con
        con.execute(SQL_MACRO_PARSE_DATE)
        for sql in (SQL_STG_USERS, SQL_STG_FEATURES, SQL_STG_SUBS, SQL_ENGINE_PARAMS):
            con.execute(self._sql(sql))

        n_subs = con.execute("SELECT COUNT(*) FROM stg_subs").fetchone()[0]
        if not n_subs:
            raise ValueError("No valid subscription rows after cleaning - check the CSV files.")

        self.as_of = con.execute("SELECT as_of FROM engine_params").fetchone()[0]
        first_month = con.execute(
            "SELECT CAST(date_trunc('month', MIN(start_date)) AS DATE) FROM stg_subs"
        ).fetchone()[0]
        con.execute(self._sql(SQL_MONTH_SPINE)
                    .replace("__AS_OF__", self.as_of.isoformat())
                    .replace("__FIRST_MONTH__", first_month.isoformat()))

    # ---- public API ---------------------------------------------------------
    def filter_options(self) -> tuple[list[str], list[str]]:
        """Distinct countries and vehicle models for the sidebar."""
        countries = [r[0] for r in self.con.execute(
            "SELECT DISTINCT country FROM stg_users ORDER BY 1").fetchall()]
        models = [r[0] for r in self.con.execute(
            "SELECT DISTINCT vehicle_model FROM stg_users ORDER BY 1").fetchall()]
        return countries, models

    def data_quality(self) -> pd.DataFrame:
        """Raw vs cleaned row counts, plus how many rows were rejected."""
        df = self.con.execute(self._sql(SQL_DATA_QUALITY)).df()
        df["rows_removed"] = df["raw_rows"] - df["clean_rows"]
        return df

    def empty_snapshot(self) -> dict[str, pd.DataFrame]:
        out = {name: pd.DataFrame() for name in QUERIES}
        out["kpis"] = compute_kpis(out)
        return out

    def snapshot(self, countries: list[str] | None = None,
                 models: list[str] | None = None) -> dict:
        """Run every mart query for the chosen filter scope. None = no filtering."""
        all_countries, all_models = self.filter_options()
        countries = all_countries if countries is None else [c for c in countries if c in all_countries]
        models = all_models if models is None else [m for m in models if m in all_models]
        if not countries or not models:           # nothing selected -> empty (never crash)
            return self.empty_snapshot()

        cur = self.con.cursor()                   # own connection => own TEMP tables
        try:
            cur.register("f_countries", pd.DataFrame({"country": countries}))
            cur.register("f_models", pd.DataFrame({"vehicle_model": models}))
            cur.execute(SQL_SCOPED_USERS)
            cur.execute(SQL_SCOPED_SUBS)
            if cur.execute("SELECT COUNT(*) FROM scoped_subs").fetchone()[0] == 0:
                return self.empty_snapshot()
            snap = {name: cur.execute(sql).df() for name, sql in QUERIES.items()}
        finally:
            cur.close()

        # Tidy date columns for the UI layer
        for name, col in (("mrr_waterfall", "month_start"), ("nrr_trend", "month_start"),
                          ("churn_velocity", "month_start"), ("cohort_matrix", "cohort_month"),
                          ("conversion_trend", "trial_month")):
            if col in snap[name].columns:
                snap[name][col] = pd.to_datetime(snap[name][col], errors="coerce")

        # Trim leading months before any revenue existed (cleaner charts)
        wf = snap["mrr_waterfall"]
        if not wf.empty and (wf["ending_mrr"] > 0).any():
            first_rev = wf.loc[wf["ending_mrr"] > 0, "month_start"].min()
            snap["mrr_waterfall"] = wf[wf["month_start"] >= first_rev].reset_index(drop=True)
        snap["kpis"] = compute_kpis(snap)
        return snap


# =========================================================================== #
# 5. CLI smoke test:  python analytics_engine.py
# =========================================================================== #
def main() -> None:
    ensure_data()
    engine = AnalyticsEngine()
    print(f"Data as of {engine.as_of}")
    print("\nData quality (raw vs clean):")
    print(engine.data_quality().to_string(index=False))

    snap = engine.snapshot()
    print("\nHeadline KPIs:")
    for k, v in snap["kpis"].items():
        print(f"  {k:<22} {v:,.2f}")
    gap = snap["mrr_waterfall"]["reconciliation_gap"].abs().max()
    print(f"\nMRR reconciliation max gap: {gap}  (0.0 = waterfall ties out)")

    pd.set_option("display.width", 180, "display.max_columns", 30)
    for name in ("conversion_by_model", "funnel", "feature_adoption", "seasonal_churn"):
        print(f"\n--- {name} ---")
        print(snap[name].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
    