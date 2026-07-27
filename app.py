"""
Sales & Revenue Analytics Dashboard
Streamlit application for corporate sales analytics and forecasting.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing import apply_filters, calculate_kpis, preprocess_pipeline
from src.model import MODEL_PATH, SalesForecastModel, train_and_save_model
from src.visualization import (
    customer_segmentation,
    forecast_chart,
    kpi_card_html,
    monthly_revenue_trend,
    product_profitability,
    revenue_by_category,
    revenue_by_region,
    sales_channel_performance,
    top_products_chart,
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sales & Revenue Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading and processing sales data…")
def load_processed_data():
    """Load and preprocess sales data with caching."""
    return preprocess_pipeline()


@st.cache_resource(show_spinner="Training forecast model…")
def load_forecast_model(_df_hash: str, df):
    """Load or train the sales forecasting model."""
    try:
        if MODEL_PATH.exists():
            return SalesForecastModel.load()
    except Exception:
        pass
    return train_and_save_model(df)


def format_currency(value: float) -> str:
    """Format number as USD currency string."""
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.2f}"


def main() -> None:
    """Render the full dashboard."""
    st.markdown('<p class="main-header">Sales & Revenue Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Corporate sales performance, revenue insights, and forecasting</p>',
        unsafe_allow_html=True,
    )

    # Load data
    try:
        df, _ = load_processed_data()
    except FileNotFoundError:
        st.error(
            "Sales data not found. Run `python src/generate_data.py` to generate the dataset."
        )
        st.stop()
    except Exception as exc:
        st.error(f"Error loading data: {exc}")
        st.stop()

    # -----------------------------------------------------------------------
    # Sidebar Filters
    # -----------------------------------------------------------------------
    st.sidebar.header("Filters")
    min_date = df["Order_Date"].min().date()
    max_date = df["Order_Date"].max().date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    regions = st.sidebar.multiselect(
        "Region",
        options=sorted(df["Region"].unique()),
        default=sorted(df["Region"].unique()),
    )

    categories = st.sidebar.multiselect(
        "Product Category",
        options=sorted(df["Product_Category"].unique()),
        default=sorted(df["Product_Category"].unique()),
    )

    channels = st.sidebar.multiselect(
        "Sales Channel",
        options=sorted(df["Sales_Channel"].unique()),
        default=sorted(df["Sales_Channel"].unique()),
    )

    st.sidebar.markdown("---")
    st.sidebar.info(f"**{len(df):,}** total records in dataset")

    # Apply filters
    if isinstance(date_range, tuple) and len(date_range) == 2:
        filtered_df = apply_filters(df, date_range, regions, categories, channels)
    else:
        filtered_df = apply_filters(df, None, regions, categories, channels)

    if filtered_df.empty:
        st.warning("No data matches the selected filters. Adjust filters to continue.")
        st.stop()

    kpis = calculate_kpis(filtered_df)

    # -----------------------------------------------------------------------
    # Section 1: Executive Summary
    # -----------------------------------------------------------------------
    st.markdown("## Executive Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            kpi_card_html("Total Revenue", format_currency(kpis["total_revenue"])),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            kpi_card_html("Total Profit", format_currency(kpis["total_profit"])),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            kpi_card_html("Total Orders", f"{kpis['total_orders']:,}"),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            kpi_card_html("Average Order Value", format_currency(kpis["average_order_value"])),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Section 2: Revenue Analysis
    # -----------------------------------------------------------------------
    st.markdown("## Revenue Analysis")
    rev_col1, rev_col2 = st.columns(2)

    with rev_col1:
        st.plotly_chart(monthly_revenue_trend(filtered_df), use_container_width=True)

    with rev_col2:
        st.plotly_chart(revenue_by_region(filtered_df), use_container_width=True)

    st.plotly_chart(revenue_by_category(filtered_df), use_container_width=True)

    # Growth table
    with st.expander("Monthly Revenue Growth Details"):
        growth_df = kpis["monthly_revenue_growth"].copy()
        growth_df["Revenue"] = growth_df["Revenue"].apply(lambda x: f"${x:,.2f}")
        growth_df["Revenue_Growth_Pct"] = growth_df["Revenue_Growth_Pct"].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
        )
        st.dataframe(growth_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Section 3: Product Analysis
    # -----------------------------------------------------------------------
    st.markdown("## Product Analysis")
    prod_col1, prod_col2 = st.columns(2)

    with prod_col1:
        st.plotly_chart(top_products_chart(filtered_df), use_container_width=True)

    with prod_col2:
        st.plotly_chart(product_profitability(filtered_df), use_container_width=True)

    with st.expander("Top Selling Products Table"):
        top_df = kpis["top_selling_products"].head(15).copy()
        top_df["Revenue"] = top_df["Revenue"].apply(lambda x: f"${x:,.2f}")
        st.dataframe(top_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Section 4: Customer Analysis
    # -----------------------------------------------------------------------
    st.markdown("## Customer Analysis")
    cust_col1, cust_col2 = st.columns(2)

    with cust_col1:
        st.plotly_chart(customer_segmentation(filtered_df), use_container_width=True)

    with cust_col2:
        st.plotly_chart(sales_channel_performance(filtered_df), use_container_width=True)

    st.metric("Total Unique Customers", f"{kpis['total_customers']:,}")

    with st.expander("Best Performing Regions"):
        region_df = kpis["best_performing_regions"].copy()
        region_df["Revenue"] = region_df["Revenue"].apply(lambda x: f"${x:,.2f}")
        region_df["Profit"] = region_df["Profit"].apply(lambda x: f"${x:,.2f}")
        st.dataframe(region_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Section 5: Sales Forecasting
    # -----------------------------------------------------------------------
    st.markdown("## Sales Forecasting")
    st.caption("Machine learning models predict future monthly revenue trends.")

    forecast_periods = st.slider("Forecast Periods (months)", 3, 12, 6)

    try:
        df_hash = str(filtered_df["Revenue"].sum())
        model = load_forecast_model(df_hash, df)
        predictions = model.predict_future(df, periods=forecast_periods)

        historical = (
            df.groupby("YearMonth", as_index=False)["Revenue"]
            .sum()
            .sort_values("YearMonth")
        )

        st.plotly_chart(forecast_chart(historical, predictions), use_container_width=True)

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Best Model", model.best_model_name)
        with metric_col2:
            st.metric("Random Forest R²", f"{model.metrics.get('rf_r2', 0):.3f}")
        with metric_col3:
            st.metric("Linear Regression R²", f"{model.metrics.get('linear_r2', 0):.3f}")

        st.dataframe(
            predictions.rename(columns={"Predicted_Revenue": "Predicted Revenue ($)"}),
            use_container_width=True,
            hide_index=True,
        )
    except Exception as exc:
        st.warning(f"Forecast model unavailable: {exc}")

    # Footer
    st.markdown("---")
    st.caption(
        "Sales & Revenue Analytics Dashboard | Built with Streamlit, Pandas, Plotly & Scikit-learn"
    )


if __name__ == "__main__":
    main()
