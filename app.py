"""
Insurance Premium Sales Dashboard
----------------------------------
Interactive Streamlit app for exploring the synthetic premium/policy sales
dataset and running quick forecasts on demand.

Run with:  streamlit run app.py
"""

import io
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Premium Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "synthetic_premium_sales_data.csv"


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    return df


df_raw = load_data(DATA_PATH)

# ------------------------------------------------------------------
# Session state defaults (so buttons can reset things)
# ------------------------------------------------------------------
if "show_anomalies_only" not in st.session_state:
    st.session_state.show_anomalies_only = False
if "chart_type" not in st.session_state:
    st.session_state.chart_type = "Line"
if "forecast_model" not in st.session_state:
    st.session_state.forecast_model = None
if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None

# ------------------------------------------------------------------
# Sidebar — filters & controls
# ------------------------------------------------------------------
st.sidebar.title("📊 Filters")

min_date, max_date = df_raw["month"].min(), df_raw["month"].max()
date_range = st.sidebar.slider(
    "Date range",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="MMM YYYY",
)

all_products = sorted(df_raw["product_line"].unique())
all_channels = sorted(df_raw["channel"].unique())
all_regions = sorted(df_raw["region"].unique())

selected_products = st.sidebar.multiselect("Product line", all_products, default=all_products)
selected_channels = st.sidebar.multiselect("Channel", all_channels, default=all_channels)
selected_regions = st.sidebar.multiselect("Region", all_regions, default=all_regions)

st.sidebar.markdown("---")
col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("🔄 Reset filters", use_container_width=True):
        st.session_state.clear()
        st.rerun()
with col_b:
    apply_clicked = st.button("✅ Apply", use_container_width=True, type="primary")

st.sidebar.markdown("---")
st.session_state.show_anomalies_only = st.sidebar.toggle(
    "Show anomaly months only", value=st.session_state.show_anomalies_only
)

# ------------------------------------------------------------------
# Apply filters
# ------------------------------------------------------------------
mask = (
    (df_raw["month"] >= pd.Timestamp(date_range[0]))
    & (df_raw["month"] <= pd.Timestamp(date_range[1]))
    & (df_raw["product_line"].isin(selected_products))
    & (df_raw["channel"].isin(selected_channels))
    & (df_raw["region"].isin(selected_regions))
)
df = df_raw[mask].copy()

if st.session_state.show_anomalies_only:
    anomaly_months = df[df["notes"].astype(str).str.strip() != ""]["month"].unique()
    df = df[df["month"].isin(anomaly_months)]

if df.empty:
    st.warning("No data matches the current filters. Adjust filters in the sidebar.")
    st.stop()

# ------------------------------------------------------------------
# Header + KPI row
# ------------------------------------------------------------------
st.title("🏢 Insurance Premium Sales Dashboard")
st.caption("Synthetic data — explore trends, seasonality, and run quick forecasts.")

total_gwp = df["gross_written_premium"].sum()
total_policies = df["policy_count"].sum()
avg_ticket = df["average_ticket_size"].mean()
total_renewal = df["renewal_premium"].sum()

monthly_totals_all = df.groupby("month")["gross_written_premium"].sum().sort_index()
if len(monthly_totals_all) >= 13:
    yoy = (monthly_totals_all.iloc[-1] / monthly_totals_all.iloc[-13] - 1) * 100
    yoy_label = f"{yoy:+.1f}%"
else:
    yoy_label = "n/a (need 13+ months)"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total GWP", f"₹{total_gwp/1e7:,.2f} Cr")
k2.metric("New Policies", f"{total_policies:,.0f}")
k3.metric("Avg Ticket Size", f"₹{avg_ticket:,.0f}")
k4.metric("Renewal Premium", f"₹{total_renewal/1e7:,.2f} Cr")
k5.metric("YoY Growth (latest mo.)", yoy_label)

st.markdown("---")

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab_overview, tab_breakdown, tab_seasonality, tab_anomalies, tab_forecast, tab_data = st.tabs(
    ["📈 Overview", "🧩 Breakdown", "🗓️ Seasonality", "⚠️ Anomalies", "🔮 Forecast", "📋 Raw Data"]
)

# ---------------- Overview ----------------
with tab_overview:
    st.subheader("Monthly Gross Written Premium — Trend")

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
    with btn_col1:
        if st.button("📉 Line chart"):
            st.session_state.chart_type = "Line"
    with btn_col2:
        if st.button("📊 Bar chart"):
            st.session_state.chart_type = "Bar"

    monthly = df.groupby("month", as_index=False)["gross_written_premium"].sum()
    anomaly_months = df[df["notes"].astype(str).str.strip() != ""]["month"].unique()

    if st.session_state.chart_type == "Line":
        fig = px.line(monthly, x="month", y="gross_written_premium", markers=True)
    else:
        fig = px.bar(monthly, x="month", y="gross_written_premium")

    for m in anomaly_months:
        fig.add_vline(x=m, line_dash="dash", line_color="red", opacity=0.4)

    fig.update_layout(yaxis_title="Gross Written Premium (₹)", xaxis_title="Month", height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Policy Count vs. Premium — is average ticket size drifting?")
    scatter_df = df.groupby("month", as_index=False).agg(
        policy_count=("policy_count", "sum"), gwp=("gross_written_premium", "sum")
    )
    scatter_df["avg_ticket"] = scatter_df["gwp"] / scatter_df["policy_count"]
    fig2 = px.scatter(
        scatter_df, x="policy_count", y="gwp", size="avg_ticket", color="avg_ticket",
        hover_data=["month"], color_continuous_scale="Viridis",
    )
    fig2.update_layout(height=420)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- Breakdown ----------------
with tab_breakdown:
    st.subheader("Breakdown by dimension")
    dim_choice = st.radio(
        "Break down by:", ["product_line", "channel", "region"], horizontal=True
    )

    c1, c2 = st.columns(2)
    with c1:
        show_stacked = st.button("📚 Stacked area view")
    with c2:
        show_pie = st.button("🥧 Share (pie) view")

    grouped = df.groupby(["month", dim_choice], as_index=False)["gross_written_premium"].sum()

    if show_pie:
        pie_data = df.groupby(dim_choice, as_index=False)["gross_written_premium"].sum()
        fig = px.pie(pie_data, names=dim_choice, values="gross_written_premium", hole=0.4)
        fig.update_layout(height=480)
    elif show_stacked:
        fig = px.area(grouped, x="month", y="gross_written_premium", color=dim_choice)
        fig.update_layout(height=480)
    else:
        fig = px.line(grouped, x="month", y="gross_written_premium", color=dim_choice, markers=True)
        fig.update_layout(height=480)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Total GWP by {dim_choice}")
    totals = (
        df.groupby(dim_choice, as_index=False)["gross_written_premium"]
        .sum()
        .sort_values("gross_written_premium", ascending=True)
    )
    fig3 = px.bar(totals, x="gross_written_premium", y=dim_choice, orientation="h")
    fig3.update_layout(height=350, xaxis_title="Gross Written Premium (₹)")
    st.plotly_chart(fig3, use_container_width=True)

# ---------------- Seasonality ----------------
with tab_seasonality:
    st.subheader("Seasonality patterns")

    season_df = df.copy()
    season_df["month_num"] = season_df["month"].dt.month
    season_df["month_name"] = season_df["month"].dt.strftime("%b")

    heat = df.pivot_table(
        index="product_line", columns=df["month"].dt.month, values="gross_written_premium", aggfunc="sum"
    )
    heat = heat.div(heat.sum(axis=1), axis=0)
    fig = px.imshow(
        heat, labels=dict(x="Month", y="Product Line", color="Share of GWP"),
        color_continuous_scale="YlOrRd", aspect="auto",
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribution of monthly GWP by calendar month")
    monthly_by_year = (
        season_df.groupby(["month_num", "month_name"], as_index=False)["gross_written_premium"]
        .sum()
        .sort_values("month_num")
    )
    fig2 = px.box(
        season_df.groupby(["month_num", "month_name", season_df["month"].dt.year], as_index=False)[
            "gross_written_premium"
        ].sum(),
        x="month_name", y="gross_written_premium",
        category_orders={"month_name": monthly_by_year["month_name"].tolist()},
    )
    fig2.update_layout(height=420, xaxis_title="Month", yaxis_title="GWP (₹)")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- Anomalies ----------------
with tab_anomalies:
    st.subheader("Flagged anomaly months")
    anomaly_rows = (
        df[df["notes"].astype(str).str.strip() != ""]
        [["month", "product_line", "region", "notes"]]
        .drop_duplicates()
        .sort_values("month")
    )
    if anomaly_rows.empty:
        st.info("No anomalies in the current filtered view.")
    else:
        st.dataframe(anomaly_rows, use_container_width=True, hide_index=True)

        for _, row in anomaly_rows.drop_duplicates(subset="month").iterrows():
            with st.expander(f"📅 {row['month'].strftime('%b %Y')} — {row['notes']}"):
                sub = df[df["month"] == row["month"]].groupby("product_line", as_index=False)[
                    "gross_written_premium"
                ].sum()
                fig = px.bar(sub, x="product_line", y="gross_written_premium")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

# ---------------- Forecast ----------------
with tab_forecast:
    st.subheader("Quick forecast")
    st.caption("Runs on the company-level aggregated series for the current filter selection.")

    f1, f2, f3 = st.columns(3)
    with f1:
        run_sarimax = st.button("▶️ Run SARIMAX", use_container_width=True)
    with f2:
        run_prophet = st.button("▶️ Run Prophet", use_container_width=True)
    with f3:
        horizon = st.number_input("Forecast horizon (months)", min_value=1, max_value=24, value=6)

    monthly_series = df.groupby("month")["gross_written_premium"].sum().asfreq(
        "MS"
    ).sort_index()

    def plot_forecast(history, forecast_index, forecast_mean, lower=None, upper=None, title=""):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history.index, y=history.values, name="History", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=forecast_index, y=forecast_mean, name="Forecast",
                                  mode="lines+markers", line=dict(dash="dash")))
        if lower is not None and upper is not None:
            fig.add_trace(go.Scatter(
                x=list(forecast_index) + list(forecast_index)[::-1],
                y=list(upper) + list(lower)[::-1],
                fill="toself", fillcolor="rgba(0,100,255,0.15)",
                line=dict(color="rgba(255,255,255,0)"), name="Confidence interval", showlegend=True,
            ))
        fig.update_layout(title=title, height=450, xaxis_title="Month", yaxis_title="Gross Written Premium (₹)")
        return fig

    if run_sarimax:
        if len(monthly_series) < 24:
            st.warning("Need at least 24 months of history in the current filter for a stable SARIMAX fit.")
        else:
            with st.spinner("Fitting SARIMAX..."):
                from statsmodels.tsa.statespace.sarimax import SARIMAX

                model = SARIMAX(
                    monthly_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False)
                fc = model.get_forecast(steps=int(horizon))
                mean = fc.predicted_mean
                ci = fc.conf_int(alpha=0.2)
                st.session_state.forecast_result = dict(
                    model="SARIMAX", history=monthly_series, index=mean.index,
                    mean=mean.values, lower=ci.iloc[:, 0].values, upper=ci.iloc[:, 1].values,
                )

    if run_prophet:
        if len(monthly_series) < 12:
            st.warning("Need at least 12 months of history in the current filter to fit Prophet.")
        else:
            with st.spinner("Fitting Prophet..."):
                from prophet import Prophet

                p_df = monthly_series.reset_index()
                p_df.columns = ["ds", "y"]
                m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                m.fit(p_df)
                future = m.make_future_dataframe(periods=int(horizon), freq="MS")
                fc_full = m.predict(future)
                tail = fc_full.tail(int(horizon))
                st.session_state.forecast_result = dict(
                    model="Prophet", history=monthly_series, index=pd.to_datetime(tail["ds"]),
                    mean=tail["yhat"].values, lower=tail["yhat_lower"].values, upper=tail["yhat_upper"].values,
                )

    if st.session_state.forecast_result:
        r = st.session_state.forecast_result
        fig = plot_forecast(r["history"], r["index"], r["mean"], r["lower"], r["upper"],
                             title=f"{r['model']} — {horizon}-month forecast")
        st.plotly_chart(fig, use_container_width=True)

        forecast_table = pd.DataFrame({
            "month": pd.to_datetime(r["index"]).strftime("%Y-%m"),
            "forecast": np.round(r["mean"], 0),
            "lower": np.round(r["lower"], 0),
            "upper": np.round(r["upper"], 0),
        })
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)

        csv_buf = io.StringIO()
        forecast_table.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download forecast CSV", csv_buf.getvalue(),
            file_name=f"{r['model'].lower()}_forecast.csv", mime="text/csv",
        )
    else:
        st.info("Click **Run SARIMAX** or **Run Prophet** above to generate a forecast.")

# ---------------- Raw Data ----------------
with tab_data:
    st.subheader("Filtered raw data")
    st.dataframe(df.sort_values("month"), use_container_width=True, hide_index=True)

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download filtered data as CSV", csv_buf.getvalue(),
        file_name="filtered_premium_sales_data.csv", mime="text/csv",
    )

st.markdown("---")
st.caption("Data is synthetic — generated for demo/prototyping purposes only.")
