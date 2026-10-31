"""
mock_data_generator.py
======================
Connected Car "Feature-as-a-Service" (FaaS) Digital Monetization Engine
-----------------------------------------------------------------------
Generates a realistic, *deliberately slightly messy* relational dataset that
mimics what an automaker's over-the-air (OTA) subscription platform would
export from its billing system:

    users_dim          one row per vehicle owner
    features_dim       the OTA feature catalogue (price list)
    subscription_fact  one row per user x feature subscription

Business behaviours baked into the simulation (so the analytics have a story):
  * Every subscription begins as a free 14-day TRIAL.
  * Trial-to-paid conversion depends on the feature, the customer tier and the
    vehicle model (performance buyers convert performance features more often).
  * Heated features (seats / steering wheel) are started mostly in autumn/winter
    and cancelled disproportionately in spring/summer -> SEASONAL CHURN.
  * Signups grow over time (a scaling product), so MRR trends upward.

Data-quality problems injected on purpose (the engine must clean them):
  * Mixed date formats (ISO 'YYYY-MM-DD' and 'DD/MM/YYYY')
  * Inconsistent country spellings / casing / whitespace
  * Inconsistent status casing ('ACTIVE', ' cancelled ', 'Canceled')
  * Missing churn reasons, missing vehicle models, missing tiers
  * Duplicate rows, orphan foreign keys, end_date earlier than start_date

Run:  python mock_data_generator.py            (writes CSVs to ./data)
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_DIR = Path(__file__).resolve().parent / "data"

SEED = 42                       # reproducible portfolio data
N_USERS = 3500                  # ~3.5k users -> ~7.5k subscriptions
SIM_START = date(2024, 1, 1)    # first possible signup
AS_OF = date(2026, 10, 30)      # "today" inside the simulation
TRIAL_DAYS = 14                 # free-trial length (documented assumption)

COUNTRIES = {  # country -> sampling weight
    "United States": 0.32, "Germany": 0.16, "United Kingdom": 0.12,
    "Japan": 0.09, "Canada": 0.08, "France": 0.08, "Australia": 0.08,
    "Norway": 0.07,
}
# Cold-climate markets lean heavily on heated features.
HEAT_FACTOR = {
    "Norway": 1.5, "Canada": 1.4, "Germany": 1.1, "United Kingdom": 1.0,
    "United States": 1.0, "Japan": 0.9, "France": 0.9, "Australia": 0.5,
}
COUNTRY_ALIASES = {  # messy spellings injected into ~8% of users
    "United States": ["USA", "usa", "U.S.", "united states", " United States "],
    "Germany": ["germany", "Deutschland", "DE", " Germany"],
    "United Kingdom": ["UK", "uk", "Great Britain", "united kingdom "],
    "Japan": ["japan", "JP"],
    "Canada": ["canada", "CA", " Canada "],
    "France": ["france", "FR"],
    "Australia": ["australia", "AU"],
    "Norway": ["norway", " Norway", "NO"],
}

# Fictional vehicle line-up (avoids real brand/trademark names).
VEHICLE_MODELS = {  # model -> (sampling weight, {tier: probability})
    "Metro Compact": (0.22, {"Standard": 0.70, "Premium": 0.27, "Performance": 0.03}),
    "Aero Sedan":    (0.27, {"Standard": 0.35, "Premium": 0.50, "Performance": 0.15}),
    "Aero SUV":      (0.24, {"Standard": 0.30, "Premium": 0.50, "Performance": 0.20}),
    "Volt GT":       (0.11, {"Standard": 0.05, "Premium": 0.30, "Performance": 0.65}),
    "Terra Pickup":  (0.16, {"Standard": 0.40, "Premium": 0.45, "Performance": 0.15}),
}
MODEL_CONV_ADJ = {  # small conversion uplift / drag by model
    "Metro Compact": -0.03, "Aero Sedan": 0.00, "Aero SUV": 0.02,
    "Volt GT": 0.06, "Terra Pickup": -0.01,
}
TIER_CONV_ADJ = {"Standard": -0.05, "Premium": 0.03, "Performance": 0.06}
TIER_CHURN_MULT = {"Standard": 1.30, "Premium": 1.00, "Performance": 0.80}

# Feature catalogue + behavioural parameters.
FEATURES = [
    # id,   name,                    monthly, upfront, popularity, base_conv, base_monthly_churn
    ("F01", "Highway Autopilot",       99.0, 6000.0, 0.35, 0.30, 0.050),
    ("F02", "Heated Seats",            15.0,  450.0, 0.75, 0.45, 0.030),
    ("F03", "Heated Steering Wheel",    9.0,  300.0, 0.45, 0.40, 0.035),
    ("F04", "Performance Boost",       49.0, 2000.0, 0.20, 0.38, 0.040),
    ("F05", "Premium Connectivity",    10.0,  250.0, 0.80, 0.55, 0.035),
    ("F06", "Track Mode Telemetry",    29.0, 1200.0, 0.12, 0.32, 0.060),
    ("F07", "Remote Park Assist",      19.0,  700.0, 0.40, 0.35, 0.040),
    ("F08", "Dashcam Cloud Backup",    12.0,  400.0, 0.50, 0.40, 0.045),
]
HEATED = {"F02", "F03"}
PERF_FEATURES = {"F04", "F06"}

# How strongly each tier gravitates to each feature (multiplies popularity).
TIER_AFFINITY = {
    "Standard":    {"F01": 0.5, "F04": 0.10, "F06": 0.05, "F07": 0.7},
    "Premium":     {"F01": 1.3, "F07": 1.3, "F05": 1.1},
    "Performance": {"F01": 1.5, "F04": 3.0, "F06": 4.0, "F07": 1.1},
}

# Calendar seasonality (month -> multiplier) for heated features.
HEATED_START_WEIGHT = {10: 1.0, 11: 1.0, 12: 1.0, 1: 1.0, 2: 1.0, 3: 0.7,
                       4: 0.45, 5: 0.30, 6: 0.25, 7: 0.25, 8: 0.30, 9: 0.60}
HEATED_CHURN_MULT = {4: 2.0, 5: 3.0, 6: 4.0, 7: 4.0, 8: 3.0, 9: 1.2,
                     10: 0.7, 11: 0.5, 12: 0.4, 1: 0.4, 2: 0.5, 3: 1.0}

TRIAL_REASONS = (["Did not use during trial", "Price too high", "Unclear value",
                  "Technical issues", "Other"], [0.40, 0.30, 0.15, 0.10, 0.05])
PAID_REASONS = (["Price too high", "Not using feature enough", "Sold / traded in vehicle",
                 "Technical issues", "Switched to upfront purchase",
                 "Missing value / feature gaps", "Competitor / aftermarket option"],
                [0.26, 0.24, 0.10, 0.10, 0.08, 0.14, 0.08])


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _normalise(weights: dict | list) -> np.ndarray:
    """Turn arbitrary positive weights into a probability vector."""
    arr = np.asarray(list(weights.values()) if isinstance(weights, dict) else weights, dtype=float)
    total = arr.sum()
    if total <= 0:  # defensive: never divide by zero
        return np.full(len(arr), 1.0 / len(arr))
    return arr / total


def _fmt_date(rng: np.random.Generator, d, alt_rate: float = 0.03) -> str | None:
    """Format a date as ISO text; ~3% of the time use messy DD/MM/YYYY."""
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return None
    return d.strftime("%d/%m/%Y") if rng.random() < alt_rate else d.isoformat()


# --------------------------------------------------------------------------- #
# Table builders (clean versions first, mess injected afterwards)
# --------------------------------------------------------------------------- #
def build_features_dim() -> pd.DataFrame:
    """Feature catalogue / price list."""
    return pd.DataFrame(
        [{"feature_id": f[0], "feature_name": f[1], "monthly_price": f[2],
          "upfront_purchase_price": f[3]} for f in FEATURES]
    )


def build_users_dim(rng: np.random.Generator, n_users: int) -> pd.DataFrame:
    """Vehicle owners. Signups ramp up over time (growth product)."""
    n_days = (AS_OF - SIM_START).days - 2
    day_weights = 1.0 + 3.0 * (np.arange(n_days) / max(n_days - 1, 1))  # linear growth ramp
    day_idx = rng.choice(n_days, size=n_users, p=_normalise(day_weights))
    signup = [SIM_START + timedelta(days=int(i)) for i in day_idx]

    countries = rng.choice(list(COUNTRIES), size=n_users, p=_normalise(COUNTRIES))
    model_names = list(VEHICLE_MODELS)
    models = rng.choice(model_names, size=n_users,
                        p=_normalise([VEHICLE_MODELS[m][0] for m in model_names]))
    tiers = []
    for m in models:
        probs = VEHICLE_MODELS[m][1]
        tiers.append(rng.choice(list(probs), p=_normalise(probs)))

    users = pd.DataFrame({
        "user_id": [f"U{i:05d}" for i in range(1, n_users + 1)],
        "signup_date": signup,
        "country": countries,
        "vehicle_model": models,
        "tier": tiers,
    })
    return users.sort_values("signup_date").reset_index(drop=True)


def _pick_start(rng: np.random.Generator, signup: date, fid: str) -> date | None:
    """Trial start = signup + exponential lag, re-rolled to respect seasonality."""
    candidate = signup
    for _ in range(6):
        candidate = signup + timedelta(days=int(rng.exponential(30)))
        weight = HEATED_START_WEIGHT[candidate.month] if fid in HEATED else 1.0
        if rng.random() < weight:
            break
    return candidate if candidate <= AS_OF else None


def build_subscription_fact(rng: np.random.Generator, users: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate each trial -> (cancel during trial | convert -> monthly churn hazard).
    Output carries start_date (= trial start), end_date, status and churn_reason.
    """
    feat_ids = [f[0] for f in FEATURES]
    feat = {f[0]: f for f in FEATURES}
    rows: list[dict] = []
    sub_counter = 1

    for u in users.itertuples(index=False):
        # Feature propensity for this user (popularity x tier affinity x climate)
        w = np.array([
            feat[fid][4]
            * TIER_AFFINITY.get(u.tier, {}).get(fid, 1.0)
            * (HEAT_FACTOR.get(u.country, 1.0) if fid in HEATED else 1.0)
            for fid in feat_ids
        ])
        n_trials = min(1 + int(rng.poisson(1.1)), len(feat_ids))
        chosen = rng.choice(feat_ids, size=n_trials, replace=False, p=_normalise(w))

        for fid in chosen:
            start = _pick_start(rng, u.signup_date, fid)
            if start is None:
                continue  # trial would start in the future -> skip
            _, _, _, _, _, base_conv, base_churn = feat[fid]
            trial_end = start + timedelta(days=TRIAL_DAYS)
            rec = {"sub_id": f"S{sub_counter:06d}", "user_id": u.user_id,
                   "feature_id": fid, "start_date": start,
                   "end_date": None, "status": "trial", "churn_reason": None}
            sub_counter += 1

            if trial_end > AS_OF:                       # still inside the free trial
                rows.append(rec)
                continue

            # --- Trial decision ------------------------------------------------
            p_conv = base_conv + TIER_CONV_ADJ[u.tier] + MODEL_CONV_ADJ[u.vehicle_model]
            if fid in PERF_FEATURES and u.tier == "Performance":
                p_conv += 0.08
            p_conv = float(np.clip(p_conv, 0.05, 0.90))

            if rng.random() >= p_conv:                  # cancelled during the trial
                rec.update(status="cancelled",
                           end_date=start + timedelta(days=int(rng.integers(3, TRIAL_DAYS))),
                           churn_reason=rng.choice(TRIAL_REASONS[0], p=TRIAL_REASONS[1]))
                rows.append(rec)
                continue

            # --- Paid life: monthly churn hazard with seasonality ---------------
            cursor = trial_end
            rec["status"] = "active"
            while cursor <= AS_OF:
                hazard = base_churn * TIER_CHURN_MULT[u.tier]
                if fid in HEATED:
                    hazard *= HEATED_CHURN_MULT[cursor.month]
                if rng.random() < min(hazard, 0.9):
                    cancel_date = cursor + timedelta(days=int(rng.integers(1, 31)))
                    if cancel_date <= AS_OF:
                        rec.update(status="cancelled", end_date=cancel_date)
                        if fid in HEATED and 4 <= cancel_date.month <= 8 and rng.random() < 0.55:
                            rec["churn_reason"] = "Seasonal - not needed"
                        else:
                            rec["churn_reason"] = rng.choice(PAID_REASONS[0], p=_normalise(PAID_REASONS[1]))
                    break
                cursor += timedelta(days=30)
            rows.append(rec)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Mess injection - what real exports look like
# --------------------------------------------------------------------------- #
def make_users_messy(rng: np.random.Generator, users: pd.DataFrame) -> pd.DataFrame:
    df = users.copy()
    n = len(df)

    # Country spelling / casing / whitespace
    alias_mask = rng.random(n) < 0.08
    df["country"] = [
        rng.choice(COUNTRY_ALIASES[c]) if m else c for c, m in zip(df["country"], alias_mask)
    ]
    # Tier casing + a few missing values
    df["tier"] = [
        (t.lower() if r < 0.03 else t.upper() if r < 0.05 else t)
        for t, r in zip(df["tier"], rng.random(n))
    ]
    df.loc[rng.random(n) < 0.01, "tier"] = None
    # Missing vehicle model
    df.loc[rng.random(n) < 0.01, "vehicle_model"] = None
    # Mixed date formats (stored as text)
    df["signup_date"] = [_fmt_date(rng, d) for d in df["signup_date"]]
    # Duplicate rows (~0.4%)
    dupes = df.sample(frac=0.004, random_state=int(rng.integers(0, 10_000)))
    return pd.concat([df, dupes], ignore_index=True)


def make_subs_messy(rng: np.random.Generator, subs: pd.DataFrame) -> pd.DataFrame:
    df = subs.copy()
    n = len(df)

    # Bad end dates: cancelled rows whose end_date precedes start_date (~0.3%)
    cancelled_idx = df.index[df["status"] == "cancelled"]
    bad_idx = rng.choice(cancelled_idx, size=max(1, int(0.003 * len(cancelled_idx))), replace=False)
    df.loc[bad_idx, "end_date"] = [df.loc[i, "start_date"] - timedelta(days=int(rng.integers(1, 10)))
                                   for i in bad_idx]

    # Missing churn reasons on ~3% of cancellations
    miss_idx = rng.choice(cancelled_idx, size=int(0.03 * len(cancelled_idx)), replace=False)
    df.loc[miss_idx, "churn_reason"] = rng.choice([None, "", "N/A", "n/a"], size=len(miss_idx))

    # Status casing / whitespace / spelling variants
    def mess_status(s: str, r: float) -> str:
        if r < 0.02:
            return s.upper()
        if r < 0.04:
            return f" {s} "
        if r < 0.06:
            return "Canceled" if s == "cancelled" else s.title()
        return s
    df["status"] = [mess_status(s, r) for s, r in zip(df["status"], rng.random(n))]

    # Orphan foreign keys (~0.4%) and stray whitespace on user_id (~1%)
    orphan = rng.random(n) < 0.004
    df.loc[orphan, "user_id"] = [f"U9{int(x):04d}" for x in rng.integers(0, 9999, orphan.sum())]
    pad = rng.random(n) < 0.01
    df.loc[pad, "user_id"] = df.loc[pad, "user_id"] + " "

    # Mixed date formats (stored as text; missing end_date stays blank)
    df["start_date"] = [_fmt_date(rng, d) for d in df["start_date"]]
    df["end_date"] = [_fmt_date(rng, d) if d is not None and not pd.isna(d) else None
                      for d in df["end_date"]]

    # Duplicate rows (~0.8%) and shuffle like a real unordered export
    dupes = df.sample(frac=0.008, random_state=int(rng.integers(0, 10_000)))
    return pd.concat([df, dupes], ignore_index=True).sample(frac=1, random_state=7).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def generate_all(data_dir: Path | str = DATA_DIR, seed: int = SEED,
                 n_users: int = N_USERS) -> dict[str, pd.DataFrame]:
    """Generate the three tables, write CSVs, and return the DataFrames."""
    if n_users < 50:
        raise ValueError("n_users must be >= 50 to produce a meaningful dataset")
    rng = np.random.default_rng(seed)
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = build_features_dim()
    users_clean = build_users_dim(rng, n_users)
    subs_clean = build_subscription_fact(rng, users_clean)

    users = make_users_messy(rng, users_clean)
    subs = make_subs_messy(rng, subs_clean)

    tables = {"users_dim": users, "features_dim": features, "subscription_fact": subs}
    for name, frame in tables.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FaaS mock data CSVs.")
    parser.add_argument("--out", default=str(DATA_DIR), help="output folder (default ./data)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--users", type=int, default=N_USERS)
    args = parser.parse_args()

    tables = generate_all(args.out, args.seed, args.users)
    print(f"Wrote CSVs to {Path(args.out).resolve()}")
    for name, frame in tables.items():
        print(f"  {name:<18} {len(frame):>6,} rows")
    status = tables["subscription_fact"]["status"].str.strip().str.lower().value_counts()
    print("Status mix (approx):", status.to_dict())


if __name__ == "__main__":
    main()
    