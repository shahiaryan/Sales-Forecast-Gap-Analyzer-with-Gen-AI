"""
Sales Forecast + Gap Analysis with Gen AI —
-------------------------------------------------------------
"""

import json
import pickle
import pandas as pd
import numpy as np
import shap
import streamlit as st
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from google import genai

st.set_page_config(page_title="Sales Forecast & Gap Analysis with Gen AI", layout="wide")

# ------------------------------------------------------------------
# SIDEBAR — Connection, Model, API Key, Target
# ------------------------------------------------------------------
st.sidebar.header("1. Model Files")
model_path = st.sidebar.text_input("Forecast model .pkl path", value="sales_forecast_model_LR.pkl")
driver_model_path = st.sidebar.text_input("Driver model .pkl path (Random Forest)", value="driver_model_RF.pkl")

st.sidebar.header("2. Database Connection")
db_user = st.sidebar.text_input("MySQL Username", value="root")
db_password = st.sidebar.text_input("MySQL Password", type="password")
db_host = st.sidebar.text_input("Host", value="localhost")
db_port = st.sidebar.number_input("Port", value=3306)
db_name = st.sidebar.text_input("Database Name", value="ai_sales_analysis")

st.sidebar.header("3. Gemini API")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")

st.sidebar.header("4. Your Target")
target_amount = st.sidebar.number_input("Target Revenue", value=40000000, step=1000000)
target_year = st.sidebar.number_input("Target Year", value=2026, step=1)

run_button = st.sidebar.button("Run Full Analysis")

st.sidebar.header("5. Year-Wise Analysis")
analysis_year = st.sidebar.number_input("Analysis Year", value=2025, step=1)
year_button = st.sidebar.button("Generate Year Summary")


# ------------------------------------------------------------------
# LOAD SAVED MODEL (.pkl) — apna already-trained model directly use karo
# ------------------------------------------------------------------
FEATURE_COLUMNS = [
    'month', 'quarter',
    'revenue_lag_1', 'revenue_lag_3', 'revenue_lag_12',
    'rolling_mean_3', 'rolling_mean_6', 'rolling_std_3',
    'mom_growth_pct', 'yoy_growth_pct',
    'avg_discount', 'order_count', 'avg_order_value',
    'active_customers', 'repeat_customer_rate',
    'top_category_share', 'is_festive_month'
]


@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model, FEATURE_COLUMNS


@st.cache_resource
def load_driver_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
@st.cache_resource
def get_engine(user, password, host, port, name):
    encoded_password = quote_plus(password)
    connection_string = f"mysql+pymysql://{user}:{encoded_password}@{host}:{port}/{name}"
    return create_engine(connection_string)


@st.cache_data
def load_data(user, password, host, port, name):
    engine = get_engine(user, password, host, port, name)

    with engine.connect() as conn:
        sales = pd.read_sql("SELECT * FROM sales_transactions", conn)
        products = pd.read_sql("SELECT product_id, category FROM products", conn)
        feature_table = pd.read_sql("SELECT * FROM feature_table", conn)

    sales['txn_date'] = pd.to_datetime(sales['txn_date'])
    sales['year'] = sales['txn_date'].dt.year
    sales['month'] = sales['txn_date'].dt.month
    sales = sales.merge(products, on='product_id', how='left')

    feature_table = feature_table.sort_values(['year', 'month']).reset_index(drop=True)

    return sales, feature_table


# ------------------------------------------------------------------
# RECURSIVE FORECASTING (it uses its own model predictions to generate future features)
# ------------------------------------------------------------------
def recursive_forecast(model, feature_columns, history_df, months_to_forecast):
    """
    At each step: lag/rolling revenue features are generated using the model's previous predictions.
    For columns like discount/order_count/customers, the average of the last 3 months is used
    (assumption: operational patterns will remain roughly the same — this is a simplification
    and in real business scenarios, these should also be forecast separately).
    """
    history = history_df.copy().reset_index(drop=True)
    predictions = []

    for _ in range(months_to_forecast):
        last_month = history['month'].iloc[-1]
        last_year = history['year'].iloc[-1]
        next_month = int(last_month % 12) + 1
        next_year = int(last_year + 1) if next_month == 1 else int(last_year)

        revenue_lag_1 = history['total_revenue'].iloc[-1]
        revenue_lag_3 = history['total_revenue'].iloc[-3] if len(history) >= 3 else revenue_lag_1
        revenue_lag_12 = history['total_revenue'].iloc[-12] if len(history) >= 12 else revenue_lag_1

        rolling_mean_3 = history['total_revenue'].iloc[-3:].mean()
        rolling_mean_6 = history['total_revenue'].iloc[-6:].mean() if len(history) >= 6 else history['total_revenue'].mean()
        rolling_std_3 = history['total_revenue'].iloc[-3:].std()
        rolling_std_3 = 0 if pd.isna(rolling_std_3) else rolling_std_3

        prev_revenue = history['total_revenue'].iloc[-2] if len(history) >= 2 else revenue_lag_1
        mom_growth_pct = (revenue_lag_1 - prev_revenue) / prev_revenue * 100 if prev_revenue else 0
        yoy_growth_pct = (revenue_lag_1 - revenue_lag_12) / revenue_lag_12 * 100 if revenue_lag_12 else 0

        quarter = ((next_month - 1) // 3) + 1
        is_festive_month = 1 if next_month in [10, 11, 12] else 0

        # Exogenous features: last 3 months average carry-forward assumption
        avg_discount = history['avg_discount'].iloc[-3:].mean()
        order_count = history['order_count'].iloc[-3:].mean()
        avg_order_value = history['avg_order_value'].iloc[-3:].mean()
        active_customers = history['active_customers'].iloc[-3:].mean()
        repeat_customer_rate = history['repeat_customer_rate'].iloc[-3:].mean()
        top_category_share = history['top_category_share'].iloc[-3:].mean()

        row = pd.DataFrame([{
            'month': next_month, 'quarter': quarter,
            'revenue_lag_1': revenue_lag_1, 'revenue_lag_3': revenue_lag_3, 'revenue_lag_12': revenue_lag_12,
            'rolling_mean_3': rolling_mean_3, 'rolling_mean_6': rolling_mean_6, 'rolling_std_3': rolling_std_3,
            'mom_growth_pct': mom_growth_pct, 'yoy_growth_pct': yoy_growth_pct,
            'avg_discount': avg_discount, 'order_count': order_count, 'avg_order_value': avg_order_value,
            'active_customers': active_customers, 'repeat_customer_rate': repeat_customer_rate,
            'top_category_share': top_category_share, 'is_festive_month': is_festive_month
        }])

        pred_revenue = model.predict(row[feature_columns])[0]
        predictions.append({'year': next_year, 'month': next_month, 'predicted_revenue': pred_revenue})

        new_row = row.copy()
        new_row['year'] = next_year
        new_row['total_revenue'] = pred_revenue
        history = pd.concat([history, new_row], ignore_index=True)

    return pd.DataFrame(predictions)


# ------------------------------------------------------------------
# GAP ANALYSIS
# ------------------------------------------------------------------
def compute_gap(sales, forecast_df, target_amount, target_year):
    actual_so_far = sales[sales['year'] == target_year]['revenue'].sum()
    forecast_this_year = forecast_df[forecast_df['year'] == target_year]['predicted_revenue'].sum()

    projected_total = actual_so_far + forecast_this_year
    gap_amount = target_amount - projected_total
    gap_pct = (gap_amount / target_amount) * 100

    years_with_data = sales['year'].unique()
    breakdown_year = target_year if (target_year in years_with_data and actual_so_far > 0) else sales['year'].max()

    region_df = (
        sales[sales['year'] == breakdown_year]
        .groupby('region', as_index=False)['revenue'].sum()
        .rename(columns={'revenue': 'actual_revenue'})
    )
    region_df['revenue_share'] = region_df['actual_revenue'] / region_df['actual_revenue'].sum()
    region_df['pro_rata_target'] = region_df['revenue_share'] * target_amount
    region_df['shortfall'] = region_df['actual_revenue'] - region_df['pro_rata_target']
    region_df = region_df.sort_values('shortfall')

    category_df = (
        sales[sales['year'] == breakdown_year]
        .groupby('category', as_index=False)['revenue'].sum()
        .rename(columns={'revenue': 'actual_revenue'})
    )
    category_df['revenue_share'] = category_df['actual_revenue'] / category_df['actual_revenue'].sum()
    category_df['pro_rata_target'] = category_df['revenue_share'] * target_amount
    category_df['shortfall'] = category_df['actual_revenue'] - category_df['pro_rata_target']
    category_df = category_df.sort_values('shortfall')

    summary = {
        "target_amount": target_amount,
        "target_year": int(target_year),
        "actual_so_far": round(actual_so_far, 2),
        "forecasted_remaining": round(forecast_this_year, 2),
        "projected_total": round(projected_total, 2),
        "gap_amount": round(gap_amount, 2),
        "gap_pct": round(gap_pct, 2),
        "region_category_breakdown_based_on_year": int(breakdown_year),
        "top_underperforming_regions": region_df.head(2)[['region', 'shortfall']].to_dict('records'),
        "top_underperforming_categories": category_df.head(2)[['category', 'shortfall']].to_dict('records'),
    }
    return summary, region_df, category_df, breakdown_year


# ------------------------------------------------------------------
# DRIVER MODEL — build slice table (month x region x category)
# ------------------------------------------------------------------
def build_slice_table(sales):
    slices = (
        sales.groupby(['year', 'month', 'region', 'category'], as_index=False)
        .agg(revenue=('revenue', 'sum'), avg_discount=('discount', 'mean'),
             order_count=('transaction_id', 'count'), active_customers=('customer_id', 'nunique'))
    )
    slices = slices.sort_values(['region', 'category', 'year', 'month']).reset_index(drop=True)

    slices['prior_month_revenue'] = slices.groupby(['region', 'category'])['revenue'].shift(1)
    slices['yoy_revenue'] = slices.groupby(['region', 'category'])['revenue'].shift(12)
    slices['yoy_growth_pct'] = ((slices['revenue'] - slices['yoy_revenue']) / slices['yoy_revenue'] * 100).round(2)
    slices['is_festive_month'] = slices['month'].isin([10, 11, 12]).astype(int)

    sales_sorted = sales.sort_values('txn_date')
    first_purchase = sales_sorted.groupby('customer_id')['txn_date'].min().rename('first_purchase_date')
    swf = sales.merge(first_purchase, on='customer_id', how='left')
    swf['is_repeat'] = (swf['txn_date'] > swf['first_purchase_date']).astype(int)
    rep = swf.groupby(['year', 'month', 'region', 'category'], as_index=False).agg(repeat_customer_rate=('is_repeat', 'mean'))
    rep['repeat_customer_rate'] = (rep['repeat_customer_rate'] * 100).round(2)
    slices = slices.merge(rep, on=['year', 'month', 'region', 'category'], how='left')

    return slices


# ------------------------------------------------------------------
# DRIVER MODEL — live SHAP insights (overall + region/category specific)
# ------------------------------------------------------------------
def get_driver_insights(driver_model, sales, top_region, top_category):
    slices = build_slice_table(sales)
    slices_clean = slices.dropna().reset_index(drop=True)
    slices_encoded = pd.get_dummies(slices_clean, columns=['region', 'category'])

    # model ne jin columns pe train kiya tha, unhi se align karo (naye/missing columns 0 se fill)
    model_features = list(driver_model.feature_names_in_)
    for col in model_features:
        if col not in slices_encoded.columns:
            slices_encoded[col] = 0
    X_driver = slices_encoded[model_features]

    explainer = shap.TreeExplainer(driver_model)
    shap_values = explainer.shap_values(X_driver)

    overall_importance = pd.DataFrame({
        'feature': model_features,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    top_overall_drivers = overall_importance.head(3)['feature'].tolist()

    # top underperforming region+category specific slice (latest available month)
    match = slices_clean[
        (slices_clean['region'] == top_region) & (slices_clean['category'] == top_category)
    ]
    region_specific_drivers = None
    if not match.empty:
        latest_idx = match.sort_values(['year', 'month']).index[-1]
        row_position = slices_clean.index.get_loc(latest_idx)
        row_shap = shap_values[row_position]
        row_df = pd.DataFrame({'feature': model_features, 'shap_value': row_shap}).sort_values('shap_value')
        region_specific_drivers = row_df.head(2).to_dict('records')   # sabse negative contributors

    return top_overall_drivers, region_specific_drivers


# ------------------------------------------------------------------
# GEN AI LAYER — structured facts retrieve -> prompt -> Gemini
# ------------------------------------------------------------------
def build_prompt(summary):
    return f"""You are a sales performance analyst. Use ONLY the numbers given below —
do not recalculate or invent any numbers, only interpret and explain them.Numbers are in INR (₹) and percentages (%).

DATA:
{json.dumps(summary, indent=2)}

Write a short business report with these 4 sections:
1. A 2-3 sentence summary of the gap to target.
2. The top underperforming regions/categories and a plausible business reason.
3. Three concrete, specific actions ranked by expected impact.
4. Which customer segments or areas to prioritize and why.

Keep it concise and business-friendly, no more than 250 words."""


def call_gemini(api_key, prompt):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


def get_ai_recommendation(api_key, summary):
    return call_gemini(api_key, build_prompt(summary))


# ------------------------------------------------------------------
# YEAR-WISE ANALYSIS — historical summary for a chosen year
# ------------------------------------------------------------------
def build_year_summary(sales, year):
    year_sales = sales[sales['year'] == year]
    prev_year_sales = sales[sales['year'] == year - 1]

    total_revenue = year_sales['revenue'].sum()
    prev_revenue = prev_year_sales['revenue'].sum()
    yoy_growth_pct = ((total_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else None

    monthly_df = (
        year_sales.groupby('month', as_index=False)['revenue'].sum()
        .sort_values('month')
    )

    region_df = (
        year_sales.groupby('region', as_index=False)['revenue'].sum()
        .sort_values('revenue', ascending=False)
    )

    category_df = (
        year_sales.groupby('category', as_index=False)['revenue'].sum()
        .sort_values('revenue', ascending=False)
    )

    summary = {
        "year": int(year),
        "total_revenue": round(total_revenue, 2),
        "previous_year_revenue": round(prev_revenue, 2) if prev_revenue > 0 else None,
        "yoy_growth_pct": round(yoy_growth_pct, 2) if yoy_growth_pct is not None else None,
        "order_count": int(year_sales['transaction_id'].nunique()),
        "active_customers": int(year_sales['customer_id'].nunique()),
        "top_region": region_df.iloc[0]['region'] if not region_df.empty else None,
        "bottom_region": region_df.iloc[-1]['region'] if not region_df.empty else None,
        "top_category": category_df.iloc[0]['category'] if not category_df.empty else None,
        "bottom_category": category_df.iloc[-1]['category'] if not category_df.empty else None,
    }

    return summary, monthly_df, region_df, category_df


def build_year_prompt(summary):
    return f"""You are a sales performance analyst. Use ONLY the numbers given below —
do not recalculate or invent any numbers, only interpret and explain them.numbers are in INR (₹) and percentages (%).

DATA:
{json.dumps(summary, indent=2)}

Write a short year-in-review summary with these 3 sections:
1. A 2-3 sentence overview of how the year performed (mention YoY growth if available).
2. Which region and category performed best, and which lagged.
3. Two takeaways or things worth watching for next year.

Keep it concise and business-friendly, no more than 200 words."""


def get_year_ai_summary(api_key, summary):
    return call_gemini(api_key, build_year_prompt(summary))


# ------------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------------
st.title("Sales Forecast, Gap Analysis & AI Recommendation")

tab1, tab2 = st.tabs(["Target vs Forecast", "Year-Wise Analysis"])

with tab1:
    st.caption("Model prediction -> Target vs Forecast gap -> Gen AI recommendation.")

    if run_button:
        if not db_password:
            st.error("Enter MySQL password in the sidebar.")
        else:
            try:
                with st.spinner("Model and data are loading ..."):
                    model, feature_columns = load_model(model_path)
                    engine = get_engine(db_user, db_password, db_host, db_port, db_name)
                    sales, feature_table = load_data(db_user, db_password, db_host, db_port, db_name)

                with st.spinner("Future revenue prediction going on..."):
                    last_year = feature_table['year'].iloc[-1]
                    last_month = feature_table['month'].iloc[-1]
                    months_needed = (target_year - last_year) * 12 + (12 - last_month)
                    months_to_forecast = max(months_needed, 12)  
                    forecast_df = recursive_forecast(model, feature_columns, feature_table, months_to_forecast=months_to_forecast)

                with st.spinner("Calculating gap..."):
                    summary, region_df, category_df, breakdown_year = compute_gap(sales, forecast_df, target_amount, target_year)

                with st.spinner("SHAP insights are being generated from the driver model..."):
                    try:
                        driver_model = load_driver_model(driver_model_path)
                        top_region = region_df.iloc[0]['region']
                        top_category = category_df.iloc[0]['category']
                        top_overall_drivers, region_specific_drivers = get_driver_insights(
                            driver_model, sales, top_region, top_category
                        )
                        summary["key_revenue_drivers"] = top_overall_drivers
                        summary["region_specific_negative_drivers"] = region_specific_drivers
                        driver_available = True
                    except FileNotFoundError:
                        st.warning(f"'{driver_model_path}' Not found — SHAP insights are getting skipped, rest analysis will continue.")
                        driver_available = False

                col1, col2, col3 = st.columns(3)
                col1.metric("Target", f"₹{summary['target_amount']:,.0f}")
                col2.metric("Projected Revenue", f"₹{summary['projected_total']:,.0f}")
                col3.metric("Gap", f"₹{summary['gap_amount']:,.0f}", f"{summary['gap_pct']:.1f}%")

                st.subheader("Next 12 Months — Model Prediction")
                chart_df = forecast_df.copy()
                chart_df['period'] = chart_df['year'].astype(str) + "-" + chart_df['month'].astype(str).str.zfill(2)
                st.line_chart(chart_df.set_index('period')['predicted_revenue'])
                st.dataframe(forecast_df)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Region-Wise Shortfall")
                    st.caption(f"Based on {breakdown_year} actual data (Future-year projections are available only for overall revenue, not for the region-wise breakdown).")
                    st.dataframe(region_df[['region', 'actual_revenue', 'shortfall']])
                with col_b:
                    st.subheader("Category-Wise Shortfall")
                    st.caption(f"Based on {breakdown_year} actual data.")
                    st.dataframe(category_df[['category', 'actual_revenue', 'shortfall']])

                if driver_available:
                    st.subheader("Driver Model Insights (SHAP)")
                    st.write(f"**Overall top revenue drivers:** {', '.join(top_overall_drivers)}")
                    if region_specific_drivers:
                        st.write(f"**Most negative contributors for {top_region} / {top_category}:**")
                        st.dataframe(pd.DataFrame(region_specific_drivers))
                    else:
                        st.write(f"'{top_region}' / '{top_category}' No matching slice found for SHAP analysis.")

                st.subheader("AI Recommendation")
                if not gemini_api_key:
                    st.warning("Enter Gemini API key in the sidebar.")
                else:
                    with st.spinner("AI recommendation are being generated..."):
                        try:
                            ai_text = get_ai_recommendation(gemini_api_key, summary)
                            st.write(ai_text)

                            record = pd.DataFrame([{
                                "target_amount": summary["target_amount"],
                                "period_year": summary["target_year"],
                                "gap_amount": summary["gap_amount"],
                                "gap_pct": summary["gap_pct"],
                                "summary_text": ai_text,
                                "top_actions": "",
                                "target_segments": ""
                            }])
                            with engine.connect() as conn:
                                record.to_sql("ai_recommendations", conn, if_exists="append", index=False)
                                conn.commit()
                            st.success("Recommendation are successfully saved to MySQL (ai_recommendations table).")
                        except Exception as e:
                            st.error(f"Gemini API call failed: {e}")

                with st.expander("Raw gap_summary (features send to AI)"):
                    st.json(summary)

            except FileNotFoundError:
                st.error(f"'{model_path}' Not found. Please check the path or place your trained model in this folder.")
    else:
        st.info("Enter details in the sidebar and click 'Run Full Analysis.")


with tab2:
    st.caption("Take a look at the complete summary of any historical year — revenue, growth, region/category breakdown, AI overview.")

    if year_button:
        if not db_password:
            st.error("Enter MySQL password in the sidebar   .")
        else:
            with st.spinner("Data is being loaded..."):
                sales, feature_table = load_data(db_user, db_password, db_host, db_port, db_name)

            if analysis_year not in sales['year'].unique():
                st.error(f"{analysis_year} year data is not available. Available years: {sorted(sales['year'].unique())}")
            else:
                summary, monthly_df, region_df, category_df = build_year_summary(sales, analysis_year)

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Revenue", f"₹{summary['total_revenue']:,.0f}")
                if summary['yoy_growth_pct'] is not None:
                    col2.metric("YoY Growth", f"{summary['yoy_growth_pct']:.1f}%")
                else:
                    col2.metric("YoY Growth", "N/A (No previous year data)")
                col3.metric("Active Customers", f"{summary['active_customers']:,}")

                st.subheader(f"{analysis_year} — Monthly Revenue Trend")
                chart_df = monthly_df.copy()
                chart_df['month_label'] = chart_df['month'].astype(str).str.zfill(2)
                st.line_chart(chart_df.set_index('month_label')['revenue'])

                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Region-Wise Revenue")
                    st.dataframe(region_df)
                with col_b:
                    st.subheader("Category-Wise Revenue")
                    st.dataframe(category_df)

                st.subheader("AI Year-in-Review Summary")
                if not gemini_api_key:
                    st.warning("Enter Gemini API key in the sidebar.")
                else:
                    with st.spinner("AI summary is being generated..."):
                        try:
                            ai_text = get_year_ai_summary(gemini_api_key, summary)
                            st.write(ai_text)
                        except Exception as e:
                            st.error(f"Gemini API call failed: {e}")

                with st.expander("Raw year_summary"):
                    st.json(summary)
    else:
        st.info("Enter a year in the sidebar and click 'Generate Year Summary'.")
