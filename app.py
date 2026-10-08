import os
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RetailPulse Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .metric-card {
        background: #ffffff;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e6e6e6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .small-note {
        color: #666666;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 RetailPulse")
st.subheader(
    "Customer Behavior Analytics & Revenue Intelligence Platform"
)

st.markdown(
    """
    RetailPulse provides an end-to-end view of customer behavior,
    segmentation, churn risk, revenue forecasting, and retention.
    """
)


# ============================================================
# HELPER FUNCTION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def project_path(relative_path):
    return os.path.join(BASE_DIR, relative_path)


def load_csv(relative_path):
    path = project_path(relative_path)

    if os.path.exists(path):
        return pd.read_csv(path)

    return pd.DataFrame()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_all_data():

    cleaned = load_csv("data/cleaned_transactions.csv")
    rfm = load_csv("data/rfm_customer_segments.csv")
    segmentation = load_csv("data/customer_segmentation.csv")
    churn = load_csv("data/churn_features.csv")
    forecast = load_csv("data/revenue_forecast.csv")
    cohort = load_csv("data/cohort_retention.csv")

    return cleaned, rfm, segmentation, churn, forecast, cohort


cleaned_data, rfm_data, segmentation_data, churn_data, forecast_data, cohort_data = (
    load_all_data()
)


# ============================================================
# CHECK DATA
# ============================================================

required_data = {
    "Cleaned Transactions": cleaned_data,
    "RFM Segments": rfm_data,
    "Customer Segmentation": segmentation_data,
    "Churn Features": churn_data,
    "Revenue Forecast": forecast_data,
    "Cohort Retention": cohort_data,
}

missing_files = [
    name for name, data in required_data.items()
    if data.empty
]

if missing_files:

    st.error(
        "Some required project files are missing:"
    )

    for item in missing_files:
        st.write(f"- {item}")

    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

if "InvoiceDate" in cleaned_data.columns:
    cleaned_data["InvoiceDate"] = pd.to_datetime(
        cleaned_data["InvoiceDate"],
        errors="coerce"
    )

if "Revenue" not in cleaned_data.columns:
    if "Quantity" in cleaned_data.columns and "Price" in cleaned_data.columns:
        cleaned_data["Revenue"] = (
            cleaned_data["Quantity"] * cleaned_data["Price"]
        )

if "Segment" in segmentation_data.columns:
    segmentation_data["Segment"] = (
        segmentation_data["Segment"].astype(str)
    )

if "CohortMonth" in cohort_data.columns:
    cohort_data["CohortMonth"] = pd.to_datetime(
        cohort_data["CohortMonth"],
        errors="coerce"
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("RetailPulse")

st.sidebar.markdown(
    """
    **Analytics Modules**

    • Executive Overview  
    • Customer Segmentation  
    • Churn Analysis  
    • Revenue Forecasting  
    • Cohort Retention  
    • Data Explorer
    """
)

st.sidebar.divider()

if "InvoiceDate" in cleaned_data.columns:

    min_date = cleaned_data["InvoiceDate"].min()
    max_date = cleaned_data["InvoiceDate"].max()

    st.sidebar.write("**Data Period**")
    st.sidebar.write(
        f"{min_date.strftime('%d %b %Y')} → "
        f"{max_date.strftime('%d %b %Y')}"
    )

st.sidebar.divider()

st.sidebar.info(
    "RetailPulse Analytics Platform\n\n"
    "Code-A-Nova Data Science Internship"
)


# ============================================================
# EXECUTIVE KPIs
# ============================================================

st.markdown(
    '<div class="section-title">📈 Executive Overview</div>',
    unsafe_allow_html=True
)

total_transactions = len(cleaned_data)

if "Customer ID" in cleaned_data.columns:
    total_customers = cleaned_data["Customer ID"].nunique()
else:
    total_customers = 0

if "Revenue" in cleaned_data.columns:
    total_revenue = cleaned_data["Revenue"].sum()
else:
    total_revenue = 0

if "Invoice" in cleaned_data.columns:
    total_invoices = cleaned_data["Invoice"].nunique()
else:
    total_invoices = 0

avg_order_value = (
    total_revenue / total_invoices
    if total_invoices > 0
    else 0
)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Customers",
    f"{total_customers:,}"
)

col3.metric(
    "Revenue",
    f"£{total_revenue:,.2f}"
)

col4.metric(
    "Invoices",
    f"{total_invoices:,}"
)

col5.metric(
    "Avg Order Value",
    f"£{avg_order_value:,.2f}"
)


# ============================================================
# REVENUE TREND
# ============================================================

st.markdown(
    '<div class="section-title">💰 Revenue Intelligence</div>',
    unsafe_allow_html=True
)

if "InvoiceDate" in cleaned_data.columns:

    daily_revenue = (
        cleaned_data
        .dropna(subset=["InvoiceDate"])
        .set_index("InvoiceDate")
        .resample("W")["Revenue"]
        .sum()
        .reset_index()
    )

    fig_revenue = px.line(
        daily_revenue,
        x="InvoiceDate",
        y="Revenue",
        markers=True,
        title="Weekly Revenue Trend"
    )

    fig_revenue.update_layout(
        xaxis_title="Date",
        yaxis_title="Revenue",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_revenue,
        use_container_width=True
    )


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

st.markdown(
    '<div class="section-title">👥 Customer Segmentation</div>',
    unsafe_allow_html=True
)

if "Segment" in segmentation_data.columns:

    segment_counts = (
        segmentation_data["Segment"]
        .value_counts()
        .reset_index()
    )

    segment_counts.columns = [
        "Segment",
        "Customers"
    ]

    col1, col2 = st.columns(2)

    with col1:

        fig_segments = px.bar(
            segment_counts,
            x="Segment",
            y="Customers",
            title="Customer Segment Distribution",
            text="Customers"
        )

        fig_segments.update_layout(
            xaxis_title="Customer Segment",
            yaxis_title="Customers"
        )

        st.plotly_chart(
            fig_segments,
            use_container_width=True
        )

    with col2:

        fig_pie = px.pie(
            segment_counts,
            names="Segment",
            values="Customers",
            title="Customer Segment Share"
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    st.dataframe(
        segment_counts,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# RFM ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">🎯 RFM Customer Analysis</div>',
    unsafe_allow_html=True
)

if all(
    col in rfm_data.columns
    for col in ["Recency", "Frequency", "Monetary"]
):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Recency",
            f"{rfm_data['Recency'].mean():.1f} days"
        )

    with col2:
        st.metric(
            "Average Frequency",
            f"{rfm_data['Frequency'].mean():.1f}"
        )

    with col3:
        st.metric(
            "Average Monetary Value",
            f"£{rfm_data['Monetary'].mean():,.2f}"
        )

    fig_rfm = px.scatter(
        rfm_data,
        x="Frequency",
        y="Monetary",
        size="Recency",
        hover_data=["Customer ID"],
        title="Customer Frequency vs Monetary Value"
    )

    st.plotly_chart(
        fig_rfm,
        use_container_width=True
    )


# ============================================================
# CHURN ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">⚠️ Churn Prediction</div>',
    unsafe_allow_html=True
)

if "Churn" in churn_data.columns:

    churn_counts = (
        churn_data["Churn"]
        .value_counts()
        .sort_index()
    )

    churned = int(
        churn_counts.get(1, 0)
    )

    retained = int(
        churn_counts.get(0, 0)
    )

    total_churn_customers = churned + retained

    churn_rate = (
        churned / total_churn_customers * 100
        if total_churn_customers > 0
        else 0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Churned Customers",
        f"{churned:,}"
    )

    col2.metric(
        "Retained Customers",
        f"{retained:,}"
    )

    col3.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

    churn_display = pd.DataFrame(
        {
            "Status": [
                "Churned",
                "Retained"
            ],
            "Customers": [
                churned,
                retained
            ]
        }
    )

    fig_churn = px.pie(
        churn_display,
        names="Status",
        values="Customers",
        title="Customer Churn Distribution"
    )

    st.plotly_chart(
        fig_churn,
        use_container_width=True
    )


# ============================================================
# CHURN FEATURE IMPORTANCE
# ============================================================

feature_importance = {
    "Recency": 0.520812,
    "AvgOrderValue": 0.232196,
    "TotalQuantity": 0.105173,
    "Monetary": 0.098810,
    "Frequency": 0.027631,
    "UniqueProducts": 0.015378
}

importance_df = pd.DataFrame(
    {
        "Feature": list(feature_importance.keys()),
        "Importance": list(feature_importance.values())
    }
).sort_values(
    "Importance",
    ascending=True
)

fig_importance = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Churn Model Feature Importance"
)

st.plotly_chart(
    fig_importance,
    use_container_width=True
)

st.caption(
    "Feature importance is based on the Random Forest churn model "
    "developed in Notebook 04."
)


# ============================================================
# REVENUE FORECAST
# ============================================================

st.markdown(
    '<div class="section-title">🔮 Revenue Forecast</div>',
    unsafe_allow_html=True
)

if (
    "Date" in forecast_data.columns
    and "Actual" in forecast_data.columns
    and "Forecast" in forecast_data.columns
):

    forecast_data["Date"] = pd.to_datetime(
        forecast_data["Date"],
        errors="coerce"
    )

    fig_forecast = go.Figure()

    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_data["Date"],
            y=forecast_data["Actual"],
            mode="lines+markers",
            name="Actual Revenue"
        )
    )

    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_data["Date"],
            y=forecast_data["Forecast"],
            mode="lines+markers",
            name="Forecast Revenue"
        )
    )

    fig_forecast.update_layout(
        title="Actual vs Forecast Revenue",
        xaxis_title="Date",
        yaxis_title="Revenue",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_forecast,
        use_container_width=True
    )


# Forecast Performance Metrics
forecast_metrics = load_csv("data/forecast_metrics.csv")

if not forecast_metrics.empty:

    col1, col2, col3 = st.columns(3)

    if "MAE" in forecast_metrics.columns:
        col1.metric(
            "Forecast MAE",
            f"{forecast_metrics['MAE'].iloc[0]:.2f}"
        )

    if "RMSE" in forecast_metrics.columns:
        col2.metric(
            "Forecast RMSE",
            f"{forecast_metrics['RMSE'].iloc[0]:.2f}"
        )

    if "MAPE" in forecast_metrics.columns:
        col3.metric(
            "Forecast MAPE",
            f"{forecast_metrics['MAPE'].iloc[0]:.2f}%"
        )

# ============================================================
# COHORT RETENTION
# ============================================================

st.markdown(
    '<div class="section-title">📅 Cohort Retention Analysis</div>',
    unsafe_allow_html=True
)

try:

    cohort_display = cohort_data.copy()

    if "CohortMonth" in cohort_display.columns:
        cohort_display["CohortMonth"] = pd.to_datetime(
            cohort_display["CohortMonth"],
            errors="coerce"
        )

    # Handle CSV saved from pivot table
    if "CohortIndex" not in cohort_display.columns:

        cohort_display = pd.read_csv(
            project_path("data/cohort_retention.csv"),
            index_col=0
        )

        cohort_display.index = pd.to_datetime(
            cohort_display.index,
            errors="coerce"
        )

        fig_cohort = px.imshow(
            cohort_display * 100,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale="Blues",
            labels={
                "x": "Months Since First Purchase",
                "y": "Cohort Month",
                "color": "Retention %"
            },
            title="Customer Cohort Retention Heatmap"
        )

        st.plotly_chart(
            fig_cohort,
            use_container_width=True
        )

    else:

        st.dataframe(
            cohort_display,
            use_container_width=True
        )

except Exception as e:

    st.warning(
        f"Cohort visualization could not be rendered: {e}"
    )


# ============================================================
# DATA EXPLORER
# ============================================================

st.markdown(
    '<div class="section-title">🔎 Data Explorer</div>',
    unsafe_allow_html=True
)

dataset_choice = st.selectbox(
    "Select dataset",
    [
        "Cleaned Transactions",
        "RFM Customer Segments",
        "Customer Segmentation",
        "Churn Features",
        "Revenue Forecast",
        "Cohort Retention"
    ]
)

dataset_map = {
    "Cleaned Transactions": cleaned_data,
    "RFM Customer Segments": rfm_data,
    "Customer Segmentation": segmentation_data,
    "Churn Features": churn_data,
    "Revenue Forecast": forecast_data,
    "Cohort Retention": cohort_data
}

selected_data = dataset_map[dataset_choice]

st.dataframe(
    selected_data.head(100),
    use_container_width=True
)


# ============================================================
# DOWNLOADS
# ============================================================

st.markdown(
    '<div class="section-title">⬇️ Download Project Outputs</div>',
    unsafe_allow_html=True
)

download_col1, download_col2, download_col3 = st.columns(3)

with download_col1:

    st.download_button(
        "Download Cleaned Data",
        data=cleaned_data.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_transactions.csv",
        mime="text/csv"
    )

with download_col2:

    st.download_button(
        "Download Customer Segments",
        data=segmentation_data.to_csv(index=False).encode("utf-8"),
        file_name="customer_segmentation.csv",
        mime="text/csv"
    )

with download_col3:

    st.download_button(
        "Download Revenue Forecast",
        data=forecast_data.to_csv(index=False).encode("utf-8"),
        file_name="revenue_forecast.csv",
        mime="text/csv"
    )


# ============================================================
# PROJECT SUMMARY
# ============================================================

st.divider()

st.markdown("### 📌 Project Summary")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:

    st.markdown(
        """
        **Analytics completed**

        ✅ Data cleaning and preprocessing  
        ✅ Exploratory data analysis  
        ✅ RFM customer analysis  
        ✅ K-Means customer segmentation  
        ✅ Churn prediction  
        ✅ SHAP / feature importance analysis  
        ✅ Revenue forecasting  
        ✅ Cohort retention analysis
        """
    )

with summary_col2:

    st.markdown(
        """
        **Models & Techniques**

        • Random Forest Churn Model  
        • K-Means Clustering  
        • SARIMA Revenue Forecasting  
        • RFM Segmentation  
        • Cohort Retention Analysis  
        • Interactive Streamlit BI Dashboard
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "RetailPulse — Customer Behavior Analytics & Revenue Intelligence Platform | "
    "Code-A-Nova Data Science Internship"
)