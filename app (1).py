"""
Insurance Premium Sales Dashboard — Enhanced Edition
------------------------------------------------------
Interactive Streamlit app: filters, drill-downs, live forecasting/prediction,
model comparison, and a styled front end with a company logo.

Swap LOGO_PATH below for your real company logo file (png/svg/jpg).

Run with:  streamlit run app.py
"""

import io
import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Config — swap these for your real assets
# ------------------------------------------------------------------
DATA_PATH = "synthetic_premium_sales_data.csv"
LOGO_PATH = "company_logo.png"          # <- replace with your real logo file
COMPANY_NAME = "YourCo General Insurance"
ACCENT = "#0A3D62"       # primary brand color — change to match your palette
ACCENT_2 = "#F6B93B"     # secondary accent

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title=f"{COMPANY_NAME} — Premium Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS — make the front end more interesting
# ------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .main {{
        background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 40%);
    }}
    .kpi-card {{
        background: white;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-left: 5px solid {ACCENT};
        transition: transform 0.15s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.10);
    }}
    .kpi-label {{
        font-size: 0.80rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }}
    .kpi-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {ACCENT};
    }}
    .kpi-delta-pos {{ color: #16a34a; font-weight: 600; }}
    .kpi-delta-neg {{ color: #dc2626; font-weight: 600; }}
    .hero-banner {{
        background: linear-gradient(120deg, {ACCENT} 0%, #145a8a 100%);
        border-radius: 16px;
        padding: 26px 30px;
        color: white;
        margin-bottom: 18px;
    }}
    .hero-banner h1 {{
        margin: 0;
        font-size: 1.8rem;
    }}
    .hero-banner p {{
        margin: 4px 0 0 0;
        opacity: 0.9;
    }}
    .prediction-box {{
        background: linear-gradient(135deg, {ACCENT_2}22, {ACCENT}11);
        border: 1px solid {ACCENT_2}55;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        background-color: #eef2f7;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {ACCENT} !important;
        color: white !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    return df


df_raw = load_data(DATA_PATH)

# ------------------------------------------------------------------
# Session state defaults
# ------------------------------------------------------------------
defaults = {
    "chart_type": "Line",
    "show_anomalies_only": False,
    "show_moving_avg": False,
    "forecast_result": None,
    "backtest_results": None,
    "uploaded_logo": None,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ------------------------------------------------------------------
# Sidebar — logo + filters + controls
# ------------------------------------------------------------------
with st.sidebar:
    logo_col1, logo_col2 = st.columns([1, 2])
    with logo_col1:
        if st.session_state.uploaded_logo is not None:
            st.image(st.session_state.uploaded_logo, width=70)
        elif os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=70)
        else:
            st.markdown(
                f"<div style='width:60px;height:60px;border-radius:50%;"
                f"background:{ACCENT};color:white;display:flex;align-items:center;"
                f"justify-content:center;font-weight:bold;font-size:1.1rem;'>IL</div>",
                unsafe_allow_html=True,
            )
    with logo_col2:
        st.markdown(f"**{COMPANY_NAME}**")
        st.caption("Premium Sales Analytics")

    with st.expander("🖼️ Replace logo"):
        uploaded = st.file_uploader("Upload your company logo", type=["png", "jpg", "jpeg", "svg"])
        if uploaded is not None:
            st.session_state.uploaded_logo = uploaded
            st.rerun()

    st.markdown("---")
    st.subheader("🔎 Filters")

    min_date, max_date = df_raw["month"].min(), df_raw["month"].max()
    date_range = st.slider(
        "Date range", min_value=min_date.to_pydatetime(), max_value=max_date.to_pydatetime(),
        value=(min_date.to_pydatetime(), max_date.to_pydatetime()), format="MMM YYYY",
    )

    all_products = sorted(df_raw["product_line"].unique())
    all_channels = sorted(df_raw["channel"].unique())
    all_regions = sorted(df_raw["region"].unique())

    selected_products = st.multiselect("Product line", all_products, default=all_products)
    selected_channels = st.multiselect("Channel", all_channels, default=all_channels)
    selected_regions = st.multiselect("Region", all_regions, default=all_regions)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Reset", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()
    with c2:
        st.button("✅ Apply", use_container_width=True, type="primary")

    st.session_state.show_anomalies_only = st.toggle(
        "⚠️ Anomaly months only", value=st.session_state.show_anomalies_only
    )
    st.session_state.show_moving_avg = st.toggle(
        "〰️ Show 3-month moving average", value=st.session_state.show_moving_avg
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
# Hero banner + KPIs
# ------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-banner">
        <h1>📊 {COMPANY_NAME} — Premium Sales Dashboard</h1>
        <p>Live exploration of policy sales, seasonality, anomalies, and forward-looking forecasts.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

total_gwp = df["gross_written_premium"].sum()
total_policies = df["policy_count"].sum()
avg_ticket = df["average_ticket_size"].mean()
total_renewal = df["renewal_premium"].sum()

monthly_totals_all = df.groupby("month")["gross_written_premium"].sum().sort_index()
if len(monthly_totals_all) >= 13:
    yoy = (monthly_totals_all.iloc[-1] / monthly_totals_all.iloc[-13] - 1) * 100
    yoy_class = "kpi-delta-pos" if yoy >= 0 else "kpi-delta-neg"
    yoy_label = f"<span class='{yoy_class}'>{yoy:+.1f}%</span>"
else:
    yoy_label = "n/a"

kpis = [
    ("💰 Total GWP", f"₹{total_gwp/1e7:,.2f} Cr"),
    ("📝 New Policies", f"{total_policies:,.0f}"),
    ("🎟️ Avg Ticket Size", f"₹{avg_ticket:,.0f}"),
    ("🔁 Renewal Premium", f"₹{total_renewal/1e7:,.2f} Cr"),
    ("📈 YoY Growth", yoy_label),
]
cols = st.columns(5)
for col, (label, value) in zip(cols, kpis):
    col.markdown(
        f"""<div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )

st.write("")

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab_overview, tab_breakdown, tab_seasonality, tab_anomalies, tab_predict, tab_compare, tab_data = st.tabs(
    ["📈 Overview", "🧩 Breakdown", "🗓️ Seasonality", "⚠️ Anomalies",
     "🔮 Predict", "🏁 Model Comparison", "📋 Raw Data"]
)

# ---------------- Overview ----------------
with tab_overview:
    st.subheader("Monthly Gross Written Premium — Trend")

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("📉 Line", use_container_width=True):
            st.session_state.chart_type = "Line"
    with b2:
        if st.button("📊 Bar", use_container_width=True):
            st.session_state.chart_type = "Bar"
    with b3:
        if st.button("🌄 Area", use_container_width=True):
            st.session_state.chart_type = "Area"
    with b4:
        if st.button("🎈 Celebrate best month", use_container_width=True):
            st.balloons()

    monthly = df.groupby("month", as_index=False)["gross_written_premium"].sum().sort_values("month")
    monthly["moving_avg"] = monthly["gross_written_premium"].rolling(3).mean()
    anomaly_months = df[df["notes"].astype(str).str.strip() != ""]["month"].unique()

    if st.session_state.chart_type == "Line":
        fig = px.line(monthly, x="month", y="gross_written_premium", markers=True,
                       color_discrete_sequence=[ACCENT])
    elif st.session_state.chart_type == "Bar":
        fig = px.bar(monthly, x="month", y="gross_written_premium",
                      color_discrete_sequence=[ACCENT])
    else:
        fig = px.area(monthly, x="month", y="gross_written_premium",
                       color_discrete_sequence=[ACCENT])

    if st.session_state.show_moving_avg:
        fig.add_trace(go.Scatter(
            x=monthly["month"], y=monthly["moving_avg"], name="3-mo moving avg",
            line=dict(color=ACCENT_2, dash="dot", width=3),
        ))

    for m in anomaly_months:
        fig.add_vline(x=m, line_dash="dash", line_color="red", opacity=0.4)

    fig.update_layout(yaxis_title="Gross Written Premium (₹)", xaxis_title="Month", height=450,
                       plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Policy Count vs. Premium (bubble = avg ticket size)")
    scatter_df = df.groupby("month", as_index=False).agg(
        policy_count=("policy_count", "sum"), gwp=("gross_written_premium", "sum")
    )
    scatter_df["avg_ticket"] = scatter_df["gwp"] / scatter_df["policy_count"]
    fig2 = px.scatter(
        scatter_df, x="policy_count", y="gwp", size="avg_ticket", color="avg_ticket",
        hover_data=["month"], color_continuous_scale="Blues",
    )
    fig2.update_layout(height=420, plot_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- Breakdown ----------------
with tab_breakdown:
    st.subheader("Breakdown by dimension")
    dim_choice = st.radio("Break down by:", ["product_line", "channel", "region"], horizontal=True)

    v1, v2, v3 = st.columns(3)
    with v1:
        view_stacked = st.button("📚 Stacked area", use_container_width=True)
    with v2:
        view_pie = st.button("🥧 Share (pie)", use_container_width=True)
    with v3:
        view_line = st.button("〰️ Lines", use_container_width=True)

    grouped = df.groupby(["month", dim_choice], as_index=False)["gross_written_premium"].sum()

    if view_pie:
        pie_data = df.groupby(dim_choice, as_index=False)["gross_written_premium"].sum()
        fig = px.pie(pie_data, names=dim_choice, values="gross_written_premium", hole=0.45,
                      color_discrete_sequence=px.colors.sequential.Blues_r)
    elif view_line:
        fig = px.line(grouped, x="month", y="gross_written_premium", color=dim_choice, markers=True)
    else:
        fig = px.area(grouped, x="month", y="gross_written_premium", color=dim_choice)

    fig.update_layout(height=480, plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Total GWP by {dim_choice}")
    totals = df.groupby(dim_choice, as_index=False)["gross_written_premium"].sum().sort_values(
        "gross_written_premium"
    )
    fig3 = px.bar(totals, x="gross_written_premium", y=dim_choice, orientation="h",
                   color="gross_written_premium", color_continuous_scale="Blues")
    fig3.update_layout(height=350, xaxis_title="Gross Written Premium (₹)", plot_bgcolor="white")
    st.plotly_chart(fig3, use_container_width=True)

# ---------------- Seasonality ----------------
with tab_seasonality:
    st.subheader("Seasonality patterns")
    heat = df.pivot_table(index="product_line", columns=df["month"].dt.month,
                           values="gross_written_premium", aggfunc="sum")
    heat = heat.div(heat.sum(axis=1), axis=0)
    fig = px.imshow(heat, labels=dict(x="Month", y="Product Line", color="Share of GWP"),
                      color_continuous_scale="YlGnBu", aspect="auto")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribution of monthly GWP by calendar month")
    season_df = df.copy()
    season_df["month_num"] = season_df["month"].dt.month
    season_df["month_name"] = season_df["month"].dt.strftime("%b")
    order = season_df.drop_duplicates("month_num").sort_values("month_num")["month_name"].tolist()
    box_data = season_df.groupby(["month_num", "month_name", season_df["month"].dt.year],
                                  as_index=False)["gross_written_premium"].sum()
    fig2 = px.box(box_data, x="month_name", y="gross_written_premium",
                   category_orders={"month_name": order}, color_discrete_sequence=[ACCENT])
    fig2.update_layout(height=420, xaxis_title="Month", yaxis_title="GWP (₹)", plot_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- Anomalies ----------------
with tab_anomalies:
    st.subheader("Flagged anomaly months")
    anomaly_rows = (
        df[df["notes"].astype(str).str.strip() != ""]
        [["month", "product_line", "region", "notes"]].drop_duplicates().sort_values("month")
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
                fig = px.bar(sub, x="product_line", y="gross_written_premium",
                              color_discrete_sequence=[ACCENT_2])
                fig.update_layout(height=300, plot_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)

# ---------------- Predict ----------------
with tab_predict:
    st.subheader("🔮 Live prediction")
    st.caption("Runs on the company-level aggregated series for the current filter selection.")

    f1, f2, f3 = st.columns(3)
    with f1:
        run_sarimax = st.button("▶️ Run SARIMAX", use_container_width=True)
    with f2:
        run_prophet = st.button("▶️ Run Prophet", use_container_width=True)
    with f3:
        horizon = st.number_input("Forecast horizon (months)", min_value=1, max_value=24, value=6)

    monthly_series = df.groupby("month")["gross_written_premium"].sum().asfreq("MS").sort_index()

    def plot_forecast(history, forecast_index, forecast_mean, lower=None, upper=None, title=""):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history.index, y=history.values, name="History",
                                  mode="lines+markers", line=dict(color=ACCENT)))
        fig.add_trace(go.Scatter(x=forecast_index, y=forecast_mean, name="Forecast",
                                  mode="lines+markers", line=dict(dash="dash", color=ACCENT_2)))
        if lower is not None and upper is not None:
            fig.add_trace(go.Scatter(
                x=list(forecast_index) + list(forecast_index)[::-1],
                y=list(upper) + list(lower)[::-1],
                fill="toself", fillcolor="rgba(246,185,59,0.18)",
                line=dict(color="rgba(255,255,255,0)"), name="Confidence interval", showlegend=True,
            ))
        fig.update_layout(title=title, height=450, xaxis_title="Month",
                           yaxis_title="Gross Written Premium (₹)", plot_bgcolor="white")
        return fig

    if run_sarimax:
        if len(monthly_series) < 24:
            st.warning("Need at least 24 months of history in the current filter for a stable SARIMAX fit.")
        else:
            with st.spinner("Fitting SARIMAX..."):
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                model = SARIMAX(monthly_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                                 enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                fc = model.get_forecast(steps=int(horizon))
                mean = fc.predicted_mean
                ci = fc.conf_int(alpha=0.2)
                st.session_state.forecast_result = dict(
                    model="SARIMAX", history=monthly_series, index=mean.index,
                    mean=mean.values, lower=ci.iloc[:, 0].values, upper=ci.iloc[:, 1].values,
                )
            st.success("SARIMAX forecast ready!")

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
            st.success("Prophet forecast ready!")

    if st.session_state.forecast_result:
        r = st.session_state.forecast_result

        next_val = r["mean"][0]
        last_actual = r["history"].values[-1]
        delta_pct = (next_val / last_actual - 1) * 100
        arrow = "▲" if delta_pct >= 0 else "▼"
        delta_class = "kpi-delta-pos" if delta_pct >= 0 else "kpi-delta-neg"

        pc1, pc2, pc3 = st.columns([1, 1, 2])
        with pc1:
            st.markdown(
                f"""<div class="prediction-box">
                        <div class="kpi-label">Next month prediction</div>
                        <div class="kpi-value">₹{next_val/1e5:,.1f} L</div>
                        <div class="{delta_class}">{arrow} {abs(delta_pct):.1f}% vs last actual</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        with pc2:
            st.markdown(
                f"""<div class="prediction-box">
                        <div class="kpi-label">Model used</div>
                        <div class="kpi-value">{r['model']}</div>
                        <div>{horizon}-month horizon</div>
                    </div>""",
                unsafe_allow_html=True,
            )

        fig = plot_forecast(r["history"], r["index"], r["mean"], r["lower"], r["upper"],
                             title=f"{r['model']} — {horizon}-month forecast")
        st.plotly_chart(fig, use_container_width=True)

        forecast_table = pd.DataFrame({
            "month": pd.DatetimeIndex(r["index"]).strftime("%Y-%m"),
            "forecast": np.round(r["mean"], 0),
            "lower": np.round(r["lower"], 0),
            "upper": np.round(r["upper"], 0),
        })
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)

        csv_buf = io.StringIO()
        forecast_table.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download forecast CSV", csv_buf.getvalue(),
                            file_name=f"{r['model'].lower()}_forecast.csv", mime="text/csv")
    else:
        st.info("Click **Run SARIMAX** or **Run Prophet** above to generate a live prediction.")

# ---------------- Model Comparison ----------------
with tab_compare:
    st.subheader("🏁 Backtest: SARIMAX vs Prophet")
    st.caption("Holds out the last few months as a test set and scores each model — click to run.")

    test_months = st.slider("Holdout months for backtest", min_value=3, max_value=9, value=6)
    run_backtest = st.button("🏁 Run Backtest Comparison", type="primary")

    if run_backtest:
        series = df.groupby("month")["gross_written_premium"].sum().asfreq("MS").sort_index()
        if len(series) < test_months + 12:
            st.warning("Not enough history in the current filter for this backtest window.")
        else:
            split = series.index[-test_months]
            train, test = series[series.index < split], series[series.index >= split]

            with st.spinner("Backtesting both models..."):
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                from prophet import Prophet
                from sklearn.metrics import mean_absolute_error, mean_squared_error

                sarimax_fit = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                sarimax_pred = sarimax_fit.get_forecast(steps=len(test)).predicted_mean
                sarimax_pred.index = test.index

                p_train = train.reset_index(); p_train.columns = ["ds", "y"]
                m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                m.fit(p_train)
                future = m.make_future_dataframe(periods=len(test), freq="MS")
                prophet_pred = m.predict(future).set_index("ds")["yhat"].loc[test.index]

                def scores(actual, pred, name):
                    mae = mean_absolute_error(actual, pred)
                    rmse = np.sqrt(mean_squared_error(actual, pred))
                    mape = np.mean(np.abs((actual - pred) / actual)) * 100
                    return {"Model": name, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape}

                results = pd.DataFrame([
                    scores(test.values, sarimax_pred.values, "SARIMAX"),
                    scores(test.values, prophet_pred.values, "Prophet"),
                ]).set_index("Model")

                st.session_state.backtest_results = dict(
                    results=results, test=test, sarimax_pred=sarimax_pred, prophet_pred=prophet_pred,
                )
            st.success("Backtest complete!")
            st.balloons()

    if st.session_state.backtest_results:
        br = st.session_state.backtest_results
        winner = br["results"]["MAPE (%)"].idxmin()

        st.markdown(f"### 🏆 Winner: **{winner}** (lowest MAPE)")
        st.dataframe(br["results"].style.highlight_min(subset=["MAPE (%)"], color="#c6f6d5"),
                     use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=br["results"].index, y=br["results"]["MAPE (%)"],
                              marker_color=[ACCENT if m == winner else "#c7cdd6" for m in br["results"].index]))
        fig.update_layout(title="MAPE Comparison (lower is better)", height=350, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=br["test"].index, y=br["test"].values, name="Actual",
                                   mode="lines+markers", line=dict(color="black", width=3)))
        fig2.add_trace(go.Scatter(x=br["sarimax_pred"].index, y=br["sarimax_pred"].values, name="SARIMAX",
                                   mode="lines+markers", line=dict(dash="dash", color=ACCENT)))
        fig2.add_trace(go.Scatter(x=br["prophet_pred"].index, y=br["prophet_pred"].values, name="Prophet",
                                   mode="lines+markers", line=dict(dash="dash", color=ACCENT_2)))
        fig2.update_layout(title="Actual vs. Predicted — Holdout Window", height=420, plot_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Click **Run Backtest Comparison** above to score both models on held-out months.")

# ---------------- Raw Data ----------------
with tab_data:
    st.subheader("Filtered raw data")
    st.dataframe(df.sort_values("month"), use_container_width=True, hide_index=True)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download filtered data as CSV", csv_buf.getvalue(),
                        file_name="filtered_premium_sales_data.csv", mime="text/csv")

st.markdown("---")
st.caption(f"© {COMPANY_NAME} · Data is synthetic — generated for demo/prototyping purposes only.")
