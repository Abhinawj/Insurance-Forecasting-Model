# 📊 Premium Sales Dashboard

An interactive Streamlit dashboard for exploring insurance premium/policy sales data —
with filters, drill-downs, seasonality analysis, anomaly detection, and live forecasting
(SARIMAX & Prophet) with model comparison.

> **Note:** This repo ships with a **synthetic** dataset generated to resemble Indian
> general-insurance sales patterns (seasonality, growth trends, and a few flagged
> anomalies). Swap in your own CSV to point the dashboard at real data — see
> [Using your own data](#using-your-own-data) below.

---

## ✨ Features

- **Filters** — date range, product line, channel, region (sidebar)
- **Overview** — monthly GWP trend (line/bar/area toggle), moving average overlay,
  policy-count-vs-premium bubble chart
- **Breakdown** — slice by product line / channel / region as stacked area, pie, or line views
- **Seasonality** — heatmap and box plot of monthly patterns
- **Anomalies** — flagged months with per-anomaly drill-down charts
- **Predict** — run SARIMAX or Prophet live, with a "next month" prediction card,
  confidence intervals, and CSV export
- **Model Comparison** — one-click backtest scoring SARIMAX vs. Prophet (MAE / RMSE / MAPE)
  on a held-out window, with a winner call-out
- **Company branding** — logo in the sidebar (upload your own on the fly, or swap the file)

---

## 📁 Repo structure

```
.
├── app.py                              # Streamlit app
├── company_logo.png                    # Placeholder logo — replace with your real one
├── synthetic_premium_sales_data.csv    # Sample/synthetic dataset
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## 🚀 Running locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## ☁️ Deploying on Streamlit Community Cloud

1. Push this repo to GitHub (public, or private with Streamlit's GitHub OAuth scoped to it).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with the same GitHub account.
3. Click **Create app**, then select:
   - **Repository:** `<your-username>/<your-repo>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**. First build takes a few minutes (Prophet compiles a Stan model on install).
5. Any future push to `main` auto-redeploys the app.

> **Prophet build issues on Cloud?** If the deploy fails on `prophet`, try pinning versions
> in `requirements.txt` (e.g. `prophet==1.1.5`, `cmdstanpy==1.2.0`). If it still fails, the
> SARIMAX model alone still works — you can temporarily remove the Prophet button/import
> and redeploy.

---

## 🖼️ Using your own logo

- **Quick swap:** replace `company_logo.png` in the repo with your real logo (same filename), or
- **Change the path:** edit `LOGO_PATH` at the top of `app.py`, or
- **Live upload:** use the **"Replace logo"** expander in the sidebar when the app is running
  (this only persists for your session, not committed to the repo)

---

## 📄 Using your own data

Replace `synthetic_premium_sales_data.csv` with your real export, keeping these columns
(or update `DATA_PATH` / column references in `app.py` to match yours):

| Column | Description |
|---|---|
| `month` | `YYYY-MM` |
| `product_line` | e.g. Motor (Car), Health (Retail), Travel |
| `channel` | e.g. Direct/Online, Agent, Broker |
| `region` | e.g. North, South, East, West |
| `policy_count` | New policies sold |
| `gross_written_premium` | Total premium (₹) |
| `renewal_count` | Renewed policies |
| `renewal_premium` | Renewal premium (₹) |
| `average_ticket_size` | Avg premium per policy |
| `notes` | Optional — flags anomaly months |

---

## ⚠️ Disclaimer

Forecasts and KPIs in this dashboard are for demo/prototyping purposes. When pointed at
real data, re-validate model assumptions (seasonality, holdout length, SARIMAX orders)
before using outputs for business decisions.
