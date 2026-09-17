# Sales Forecasting & Gen AI Recommendation System

An end-to-end project that predicts future sales, calculates the gap to a target revenue, identifies where a business is underperforming, and uses Gen AI to recommend concrete actions — built on a SQL + ML/DL + Gen AI + Power BI stack.

## What it does

1. **Predicts** next month's/year's revenue using a trained ML model.
2. **Compares** that prediction against a user-defined target (e.g. ₹4 Cr).
3. **Explains** where the business is lagging — by region and category — using a Random Forest driver model interpreted with SHAP.
4. **Recommends** concrete actions using a Gen AI layer (Gemini 2.5 Flash), fed only with pre-computed numbers — never letting the LLM invent figures.
5. **Visualizes** everything in an interactive Streamlit app and a Power BI dashboard.

## Tech stack

| Layer | Tools |
|---|---|
| Data storage | MySQL |
| Data cleaning & features | Python (pandas, numpy) |
| Forecasting | Linear Regression, Ridge, Lasso, Random Forest, XGBoost, SARIMA/ARIMA |
| Driver analysis | Random Forest + SHAP |
| Gen AI | Gemini 2.5 Flash (structured-facts-to-prompt pattern, not vector RAG) |
| Web app | Streamlit |
| BI dashboard | Power BI |

## Repository structure

```
sales-forecast-genai-project/
├── sql/            # Schema + the 17-question SQL analysis bank
├── notebooks/       # Feature engineering, model training, driver model, gap engine
├── models/           # Saved .pkl models (forecast + driver)
├── genai/            # Gen AI recommendation layer notebook
├── powerbi/          # .pbix dashboard + custom theme JSON
├── webapp/            # Streamlit app (forecast, gap analysis, year-wise analysis, AI recs)
└── data/               # Sample/synthetic data for testing
```

## Pipeline overview

```
SQL (clean data + 17 analysis queries)
        │
        ▼
Feature Engineering (lag, rolling, discount, customer features)
        │
        ▼
Forecasting Models  ──► Model Comparison (MAPE/RMSE)
        │
        ▼
Driver Model (Random Forest + SHAP) ──► "why are we lagging"
        │
        ▼
Target-Gap Engine (deterministic Python — target vs forecast)
        │
        ▼
Gen AI Layer (Gemini) ──► narrates the gap, recommends actions
        │
        ▼
Streamlit App  +  Power BI Dashboard
```

## Model comparison results

| Model | MAPE (%) | RMSE |
|---|---|---|
| **Linear Regression** | **2.37** | **344,677** |
| Ridge (Tuned) | 2.81 | 398,594 |
| Lasso (Tuned) | 4.11 | 546,184 |
| Random Forest (Tuned) | 5.37 | 665,991 |
| Baseline (Naive Seasonal) | 17.84 | 2,748,209 |
| XGBoost (Tuned) | 22.63 | 3,003,295 |

Plain Linear Regression outperformed every regularized/ensemble model — on this dataset's size (~24 usable monthly rows after cleaning), the underlying trend was simple enough that added model complexity didn't help, and in some cases hurt generalization.

## Setup

```bash
git clone <your-repo-url>
cd sales-forecast-genai-project
pip install -r requirements.txt
```

1. Run `sql/schema.sql` against your MySQL instance to create the tables.
2. Load your cleaned data into the tables.
3. Run the notebooks in `notebooks/` in order (Phase 2 → 5).
4. Run the Gen AI notebook in `genai/` (needs a Gemini API key).
5. Launch the web app:
   ```bash
   streamlit run webapp/streamlit_app.py
   ```
6. Open `powerbi/dashboard.pbix` in Power BI Desktop, point it at your MySQL database, and import `PowerBI_Theme.json` for the color theme (View → Themes → Browse for themes).

## Power BI dashboard

- ✅ Overview (KPIs, trend)
- ✅ Forecast vs Actual (model backtest accuracy)
- ✅ Target Gap Analysis (region/category shortfall heatmap)
- ✅ Customer Segments (RFM, churn-risk)

## Known limitations

- Recursive forecasting assumes exogenous features (discount, order count, active customers) stay near their recent average — a simplification, not a true forecast of those drivers.
- Region/category breakdown for future years falls back to the latest actual year's pattern, since no genuine future region-level data exists.
- Small dataset size (36 months) limits how much complex models like XGBoost/LSTM can be trusted over simpler ones.

## Author

Built by Aryan, transitioning from a Data Analytics background into Machine Learning.
