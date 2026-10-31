"""
app.py  -  Streamlit executive dashboard
========================================
Connected Car "Feature-as-a-Service" (FaaS) Digital Monetization Engine

This file is deliberately thin: ALL business logic lives in SQL inside
analytics_engine.py. The UI only (1) collects sidebar filters, (2) asks the engine
for a "snapshot" of pre-computed result tables, and (3) draws them.

    CSV files --> DuckDB staging + SQL marts (analytics_engine.py) --> this UI

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analytics_engine import AnalyticsEngine, ensure_data

# --------------------------------------------------------------------------- #
# Page setup + visual identity (dark, high-contrast)
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="FaaS Digital Monetization Engine",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

TEAL, BLUE, AMBER, RED, PURPLE, GREEN = (
    "#2DD4BF", "#60A5FA", "#FBBF24", "#F87171", "#A78BFA", "#4ADE80")
PALETTE = [TEAL, BLUE, AMBER, PURPLE, RED, GREEN, "#F472B6", "#94A3B8"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
      [data-testid="stMetric"] {
          background: #161B22; border: 1px solid #30363D;
          padding: 14px 16px; border-radius: 12px;
      }
      [data-testid="stMetricLabel"] {color: #8B949E;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      .subtle {color: #8B949E; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached data access (keeps the app fast and inside Render's free RAM limit)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Building the DuckDB analytics layer...")
def get_engine() -> AnalyticsEngine:
    """One engine per server process. Generates the CSVs first if they are missing."""
    ensure_data()
    return AnalyticsEngine()


@st.cache_data(show_spinner="Running analytical SQL...")
def get_snapshot(countries: tuple, models: tuple) -> dict:
    """All SQL marts for one filter selection. Cached so re-clicks are instant."""
    return get_engine().snapshot(list(countries), list(models))


# --------------------------------------------------------------------------- #
# Small UI helpers (all defensive against empty / NaN values)
# --------------------------------------------------------------------------- #
def has(df, *cols) -> bool:
    """True only if df is a non-empty DataFrame containing every listed column."""
    return isinstance(df, pd.DataFrame) and not df.empty and all(c in df.columns for c in cols)


def money(v, decimals: int = 0) -> str:
    return "n/a" if v is None or pd.isna(v) else f"${v:,.{decimals}f}"


def pct(v, decimals: int = 1) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:,.{decimals}f}%"


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply the shared dark look to every chart."""
    fig.update_layout(
        template="plotly_dark", height=height, colorway=PALETTE,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def show(fig: go.Figure, key: str) -> None:
    """Render a Plotly figure full-width. Tries the newer Streamlit width API first."""
    try:
        st.plotly_chart(fig, width="stretch", theme=None, key=key)
    except Exception:  # older Streamlit versions
        st.plotly_chart(fig, use_container_width=True, theme=None, key=key)


def show_df(df: pd.DataFrame, **kwargs) -> None:
    """Render a DataFrame full-width (same version-compatibility trick)."""
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def empty_note(what: str) -> None:
    st.info(f"No data to show for {what} with the current filters.")


# --------------------------------------------------------------------------- #
# Header + sidebar filters
# --------------------------------------------------------------------------- #
engine = get_engine()
all_countries, all_models = engine.filter_options()

st.title("Connected Car FaaS: Digital Monetization Engine")
st.markdown(
    f'<span class="subtle">Over-the-air feature subscriptions: trials, MRR, churn and '
    f'adoption &nbsp;|&nbsp; data as of <b>{engine.as_of:%d %b %Y}</b> &nbsp;|&nbsp; '
    f'every number below is produced by SQL running in DuckDB</span>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Filters")
    sel_countries = st.multiselect("Country", all_countries, default=all_countries)
    sel_models = st.multiselect("Vehicle model", all_models, default=all_models)
    st.caption("Filters re-run the SQL on the selected customers only.")
    with st.expander("Data quality: raw vs cleaned rows"):
        show_df(engine.data_quality())
    with st.expander("Business definitions"):
        st.markdown(
            "- **Trial**: free for 14 days, then paid.\n"
            "- **Converted**: still active, or cancelled after the trial ended.\n"
            "- **MRR**: month-end sum of monthly price of paid subscriptions.\n"
            "- **Expansion MRR**: a feature added while the customer already pays for another.\n"
            "- **NRR**: revenue today from customers who paid 12 months ago, divided by what they paid then.\n"
            "- **Trial to paid**: only trials whose 30-day window has fully elapsed."
        )

if not sel_countries or not sel_models:
    st.warning("Select at least one country and one vehicle model in the sidebar.")
    st.stop()

snap = get_snapshot(tuple(sel_countries), tuple(sel_models))
kpis = snap.get("kpis", {})
if not has(snap.get("mrr_waterfall")):
    st.warning("No subscriptions match these filters. Try selecting more countries or models.")
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "Executive Revenue Control",
    "Subscription Funnel & Churn Matrix",
    "Feature Performance & Segments",
])

# =========================================================================== #
# TAB 1  -  Executive Revenue Control
# =========================================================================== #
with tab1:
    mom = kpis.get("mom_growth_pct")
    row1, row2 = st.columns(3), st.columns(3)
    c1, c2, c3 = row1
    c4, c5, c6 = row2
    c1.metric("ARR", money(kpis.get("arr")), help="Annual recurring revenue = current MRR x 12.")
    c2.metric("MRR", money(kpis.get("mrr")),
              delta=None if mom is None or pd.isna(mom) else f"{mom:+.1f}% MoM",
              help="Monthly recurring revenue today. Delta is growth in the last complete month.")
    c3.metric("Revenue churn", pct(kpis.get("revenue_churn_pct")),
              help="Share of opening MRR lost per month (average of the last 3 complete months).")
    c4.metric("NRR (12-month)", pct(kpis.get("nrr_pct")),
              help="Net revenue retention. Above 100% means existing customers grow revenue.")
    c5.metric("Paying subscriptions", "n/a" if pd.isna(kpis.get("paying_subs", float("nan")))
              else f"{kpis['paying_subs']:,.0f}")
    c6.metric("Trial to paid (30d)", pct(kpis.get("trial_conversion_pct")),
              help="Share of matured trials that converted to a paid plan.")

    wf = snap["mrr_waterfall"]
    st.subheader("MRR growth components")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=wf["month_start"], y=wf["new_mrr"], name="New", marker_color=GREEN)
    fig.add_bar(x=wf["month_start"], y=wf["expansion_mrr"], name="Expansion", marker_color=BLUE)
    fig.add_bar(x=wf["month_start"], y=-wf["churned_mrr"], name="Churned", marker_color=RED)
    fig.add_trace(go.Scatter(x=wf["month_start"], y=wf["ending_mrr"], name="Ending MRR",
                             mode="lines+markers", line=dict(color=TEAL, width=3)),
                  secondary_y=True)
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title_text="Monthly movement ($)", secondary_y=False)
    fig.update_yaxes(title_text="Ending MRR ($)", secondary_y=True, showgrid=False)
    show(style(fig, 430), "mrr_components")
    st.caption(f"The latest month is partial (through {engine.as_of:%d %b}), so its bars are smaller.")

    left, right = st.columns(2)
    with left:
        st.subheader("Retention: NRR vs GRR")
        nrr = snap.get("nrr_trend")
        if has(nrr, "month_start", "nrr_pct", "grr_pct"):
            nrr = nrr[nrr["is_full_month"].astype(bool)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nrr["month_start"], y=nrr["nrr_pct"], name="NRR",
                                     mode="lines+markers", line=dict(color=TEAL, width=3)))
            fig.add_trace(go.Scatter(x=nrr["month_start"], y=nrr["grr_pct"], name="GRR",
                                     mode="lines+markers", line=dict(color=AMBER, width=2, dash="dot")))
            fig.add_hline(y=100, line_dash="dot", line_color="#8B949E")
            fig.update_yaxes(title_text="% of revenue retained (trailing 12 months)")
            show(style(fig), "nrr_trend")
        else:
            empty_note("retention")
    with right:
        st.subheader("Net new MRR")
        fig = go.Figure(go.Bar(
            x=wf["month_start"], y=wf["net_new_mrr"],
            marker_color=[GREEN if v >= 0 else RED for v in wf["net_new_mrr"]], name="Net new MRR"))
        fig.update_yaxes(title_text="New + expansion - churned ($)")
        show(style(fig), "net_new")

    with st.expander("MRR waterfall table (with reconciliation check)"):
        gap = wf["reconciliation_gap"].abs().max()
        st.caption(f"Integrity check: opening + new + expansion - churned = closing. "
                   f"Largest gap in any month: **{gap}** (0.0 means it ties out).")
        show_df(wf.drop(columns=["month_end"], errors="ignore"))

# =========================================================================== #
# TAB 2  -  Subscription Funnel & Churn Matrix
# =========================================================================== #
with tab2:
    top_l, top_r = st.columns(2)
    with top_l:
        st.subheader("Subscription funnel")
        funnel = snap.get("funnel")
        if has(funnel, "stage", "subs"):
            fig = go.Figure(go.Funnel(
                y=funnel["stage"], x=funnel["subs"], textinfo="value+percent initial",
                marker=dict(color=[BLUE, "#38BDF8", TEAL, GREEN, AMBER][:len(funnel)])))
            show(style(fig, 400), "funnel")
        else:
            empty_note("the funnel")
    with top_r:
        st.subheader("Trial-to-paid conversion by vehicle model")
        conv = snap.get("conversion_by_model")
        if has(conv, "vehicle_model", "conversion_rate_pct"):
            fig = px.bar(conv, x="vehicle_model", y="conversion_rate_pct")
            fig.update_traces(marker_color=TEAL, texttemplate="%{y:.1f}%", textposition="outside")
            fig.add_hline(y=float(conv["overall_rate_pct"].iloc[0]), line_dash="dot",
                          line_color=AMBER, annotation_text="overall", annotation_position="top left")
            fig.update_yaxes(title_text="% of trials converting within 30 days")
            fig.update_xaxes(title_text=None)
            show(style(fig, 400), "conv_model")
        else:
            empty_note("conversion")

    st.subheader("Customer churn cohort matrix")
    st.caption("Each row is the month customers started paying. Each column shows the % still "
               "paying N months later. Green = retained, red = churned.")
    cm = snap.get("cohort_matrix")
    if has(cm, "cohort_month", "month_offset", "retention_pct"):
        cm = cm[cm["cohort_size"] >= 5] if (cm["cohort_size"] >= 5).any() else cm  # hide tiny cohorts
        pivot = cm.pivot(index="cohort_month", columns="month_offset", values="retention_pct").sort_index()
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=[f"M{int(c)}" for c in pivot.columns],
            y=list(pd.to_datetime(pivot.index).strftime("%Y-%m")),
            colorscale="RdYlGn", zmin=0, zmax=100, hoverongaps=False,
            texttemplate="%{z:.0f}", textfont=dict(size=9),
            colorbar=dict(title="Retained %")))
        fig.update_yaxes(autorange="reversed", type="category")
        fig.update_xaxes(side="top")
        show(style(fig, max(420, 22 * len(pivot) + 120)), "cohort")
    else:
        empty_note("the cohort matrix")

    mid_l, mid_r = st.columns(2)
    with mid_l:
        st.subheader("Why customers cancel")
        reasons = snap.get("churn_reasons")
        if has(reasons, "churn_reason", "cancellations", "phase"):
            fig = px.bar(reasons, x="cancellations", y="churn_reason", color="phase",
                         orientation="h", color_discrete_map={
                             "During trial": AMBER, "Post-trial (paid)": RED})
            fig.update_yaxes(categoryorder="total ascending", title_text=None)
            fig.update_xaxes(title_text="Cancellations")
            show(style(fig, 440), "reasons")
        else:
            empty_note("churn reasons")
    with mid_r:
        st.subheader("Seasonal churn index")
        st.caption("100 = the product family's average month. Heated features spike in summer.")
        sc = snap.get("seasonal_churn")
        if has(sc, "month_label", "seasonal_index", "seasonal_group"):
            fig = px.bar(sc, x="month_label", y="seasonal_index", color="seasonal_group",
                         barmode="group", category_orders={"month_label": MONTHS},
                         color_discrete_map={"Heated (seasonal)": AMBER, "Year-round": TEAL})
            fig.add_hline(y=100, line_dash="dot", line_color="#8B949E",
                          annotation_text="average month", annotation_position="top left")
            fig.update_xaxes(title_text=None)
            fig.update_yaxes(title_text="Churn rate vs own average (index)")
            show(style(fig, 440), "seasonal")
        else:
            empty_note("seasonality")

    st.subheader("Cancellation velocity")
    cv = snap.get("churn_velocity")
    if has(cv, "month_start", "churn_rate_pct"):
        cv = cv[cv["is_full_month"].astype(bool)]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cv["month_start"], y=cv["churn_rate_pct"], name="Monthly churn %",
                             marker_color=RED, opacity=0.65))
        fig.add_trace(go.Scatter(x=cv["month_start"], y=cv["churn_rate_3m_avg"], name="3-month average",
                                 mode="lines", line=dict(color=TEAL, width=3)))
        fig.update_yaxes(title_text="% of opening paid subscriptions cancelled")
        show(style(fig), "velocity")
    else:
        empty_note("cancellation velocity")

# =========================================================================== #
# TAB 3  -  Feature Performance & Segment Breakdown
# =========================================================================== #
with tab3:
    fa = snap.get("feature_adoption")
    segs = snap.get("feature_segments")
    seg_users = snap.get("segment_users")

    st.subheader("Feature Adoption Index")
    st.caption("Index (0-100) blends how many customers adopt a feature with how much revenue it earns. "
               "Rank gap > 0 means it earns more than its popularity suggests.")
    if has(fa, "feature_name", "adoption_index"):
        view = fa[["feature_name", "quadrant", "adoption_index", "adoption_rate_pct",
                   "trial_conversion_pct", "adopters", "current_mrr", "lifetime_revenue",
                   "revenue_share_pct", "rank_gap"]].copy()
        show_df(view, column_config={
            "feature_name": "Feature", "quadrant": "Quadrant",
            "adoption_index": st.column_config.ProgressColumn(
                "Adoption index", min_value=0, max_value=100, format="%.1f"),
            "adoption_rate_pct": st.column_config.NumberColumn("Adoption %", format="%.1f"),
            "trial_conversion_pct": st.column_config.NumberColumn("Trial to paid %", format="%.1f"),
            "adopters": st.column_config.NumberColumn("Adopters", format="%d"),
            "current_mrr": st.column_config.NumberColumn("Current MRR", format="$%.0f"),
            "lifetime_revenue": st.column_config.NumberColumn("Lifetime revenue", format="$%.0f"),
            "revenue_share_pct": st.column_config.NumberColumn("Revenue share %", format="%.1f"),
            "rank_gap": st.column_config.NumberColumn("Rank gap", format="%d"),
        })
    else:
        empty_note("feature adoption")

    st.subheader("Adoption vs revenue by customer tier")
    if has(segs, "tier", "feature_name") and has(seg_users, "tier", "segment_users"):
        by_tier = (segs.groupby(["tier", "feature_name"], as_index=False)
                   [["trials_started", "adopters", "current_mrr", "lifetime_revenue"]].sum())
        users_per_tier = seg_users.groupby("tier")["segment_users"].sum()
        by_tier["users"] = by_tier["tier"].map(users_per_tier)
        by_tier["adoption_rate_pct"] = (100 * by_tier["adopters"]
                                        / by_tier["users"].where(by_tier["users"] > 0))

        tier_options = ["All tiers"] + [t for t in ["Standard", "Premium", "Performance", "Unknown"]
                                        if t in set(by_tier["tier"])]
        pick = st.radio("Customer tier", tier_options, horizontal=True)
        if pick == "All tiers":
            scatter_df = by_tier.groupby("feature_name", as_index=False)[
                ["adopters", "current_mrr", "lifetime_revenue"]].sum()
        else:
            scatter_df = by_tier[by_tier["tier"] == pick][
                ["feature_name", "adopters", "current_mrr", "lifetime_revenue"]]

        if not scatter_df.empty:
            scatter_df = scatter_df.assign(bubble=scatter_df["current_mrr"].clip(lower=1))
            fig = px.scatter(scatter_df, x="adopters", y="lifetime_revenue", size="bubble",
                             color="feature_name", text="feature_name", size_max=40,
                             hover_data={"bubble": False, "current_mrr": ":$,.0f",
                                         "lifetime_revenue": ":$,.0f"})
            fig.update_traces(textposition="top center", cliponaxis=False)
            fig.update_layout(showlegend=False)
            # Pad the axes so bubbles and labels never touch the edge
            fig.update_xaxes(title_text="Paying adopters",
                             range=[0, scatter_df["adopters"].max() * 1.25])
            fig.update_yaxes(title_text="Lifetime revenue ($)",
                             range=[0, scatter_df["lifetime_revenue"].max() * 1.25])
            show(style(fig, 480), "feature_scatter")
            st.caption("Bubble size = current MRR. Top-right = popular and lucrative; "
                       "top-left = a premium niche.")
        else:
            empty_note("this tier")

        st.subheader("Adoption rate by tier")
        st.caption("Share of each tier's customers who pay for the feature.")
        heat = by_tier.pivot(index="feature_name", columns="tier", values="adoption_rate_pct")
        heat = heat[[t for t in ["Standard", "Premium", "Performance", "Unknown"]
                     if t in heat.columns]]
        fig = go.Figure(go.Heatmap(
            z=heat.values, x=list(heat.columns), y=list(heat.index),
            colorscale="Teal", hoverongaps=False,
            texttemplate="%{z:.1f}%", textfont=dict(size=13),
            colorbar=dict(title="Adoption %")))
        fig.update_yaxes(autorange="reversed", automargin=True)
        fig.update_xaxes(side="top")
        show(style(fig, 420), "tier_heat")
    else:
        empty_note("segment breakdown")

    with st.expander("Feature x vehicle model x tier detail"):
        if has(segs):
            show_df(segs)
        else:
            empty_note("segment detail")

st.markdown("---")
st.caption("Portfolio project: synthetic data, no real customers. "
           "Stack: Python, DuckDB (SQL window functions and CTEs), Streamlit, Plotly.")