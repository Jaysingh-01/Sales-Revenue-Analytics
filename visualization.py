"""Plotly and Matplotlib chart builders for the dashboard."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def kpi_card_html(label: str, value: str, delta: str | None = None) -> str:
    """Return HTML snippet for a KPI metric card."""
    delta_html = f'<p style="color:#16a34a;margin:0;font-size:0.9rem;">{delta}</p>' if delta else ""
    return f"""
    <div style="background:linear-gradient(135deg,#1e3a5f 0%,#2563eb 100%);
                padding:1.25rem;border-radius:12px;color:white;text-align:center;
                box-shadow:0 4px 6px rgba(0,0,0,0.1);">
        <p style="margin:0;font-size:0.85rem;opacity:0.9;text-transform:uppercase;
                  letter-spacing:0.05em;">{label}</p>
        <h2 style="margin:0.5rem 0 0 0;font-size:1.75rem;font-weight:700;">{value}</h2>
        {delta_html}
    </div>
    """


def monthly_revenue_trend(df: pd.DataFrame) -> go.Figure:
    """Line chart of monthly revenue trend."""
    monthly = df.groupby("YearMonth", as_index=False)["Revenue"].sum()
    fig = px.line(
        monthly,
        x="YearMonth",
        y="Revenue",
        title="Monthly Revenue Trend",
        markers=True,
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Revenue ($)",
        template="plotly_white",
        hovermode="x unified",
    )
    fig.update_traces(line_color="#2563eb", line_width=3)
    return fig


def revenue_by_region(df: pd.DataFrame) -> go.Figure:
    """Bar chart of revenue by region."""
    regional = df.groupby("Region", as_index=False)["Revenue"].sum().sort_values("Revenue")
    fig = px.bar(
        regional,
        x="Revenue",
        y="Region",
        orientation="h",
        title="Revenue by Region",
        color="Revenue",
        color_continuous_scale="Blues",
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def revenue_by_category(df: pd.DataFrame) -> go.Figure:
    """Pie chart of revenue share by product category."""
    category = df.groupby("Product_Category", as_index=False)["Revenue"].sum()
    fig = px.pie(
        category,
        names="Product_Category",
        values="Revenue",
        title="Revenue by Product Category",
        hole=0.4,
    )
    fig.update_layout(template="plotly_white")
    return fig


def top_products_chart(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart of top N products by revenue."""
    products = (
        df.groupby("Product_Name", as_index=False)["Revenue"]
        .sum()
        .nlargest(top_n, "Revenue")
        .sort_values("Revenue")
    )
    fig = px.bar(
        products,
        x="Revenue",
        y="Product_Name",
        orientation="h",
        title=f"Top {top_n} Products by Revenue",
        color="Revenue",
        color_continuous_scale="Teal",
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def product_profitability(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Scatter plot of revenue vs profit by product."""
    product_metrics = (
        df.groupby("Product_Name", as_index=False)
        .agg(Revenue=("Revenue", "sum"), Profit=("Profit", "sum"))
        .nlargest(top_n, "Revenue")
    )
    fig = px.scatter(
        product_metrics,
        x="Revenue",
        y="Profit",
        size="Revenue",
        color="Profit",
        hover_name="Product_Name",
        title="Product Profitability (Top Products)",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(template="plotly_white")
    return fig


def customer_segmentation(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Bar chart of top customers by total spend."""
    customers = (
        df.groupby(["Customer_ID", "Customer_Name"], as_index=False)["Revenue"]
        .sum()
        .nlargest(top_n, "Revenue")
        .sort_values("Revenue")
    )
    customers["Label"] = customers["Customer_Name"] + " (" + customers["Customer_ID"] + ")"
    fig = px.bar(
        customers,
        x="Revenue",
        y="Label",
        orientation="h",
        title=f"Top {top_n} Customers by Revenue",
        color="Revenue",
        color_continuous_scale="Purples",
    )
    fig.update_layout(template="plotly_white", showlegend=False, yaxis_title="Customer")
    return fig


def sales_channel_performance(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of revenue and profit by sales channel."""
    channel = (
        df.groupby("Sales_Channel", as_index=False)
        .agg(Revenue=("Revenue", "sum"), Profit=("Profit", "sum"))
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Revenue", x=channel["Sales_Channel"], y=channel["Revenue"]))
    fig.add_trace(go.Bar(name="Profit", x=channel["Sales_Channel"], y=channel["Profit"]))
    fig.update_layout(
        title="Sales Channel Performance",
        barmode="group",
        template="plotly_white",
        xaxis_title="Sales Channel",
        yaxis_title="Amount ($)",
    )
    return fig


def forecast_chart(historical: pd.DataFrame, predictions: pd.DataFrame) -> go.Figure:
    """Line chart comparing historical and forecasted revenue."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=historical["YearMonth"],
            y=historical["Revenue"],
            mode="lines+markers",
            name="Historical Revenue",
            line=dict(color="#2563eb", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predictions["YearMonth"],
            y=predictions["Predicted_Revenue"],
            mode="lines+markers",
            name="Forecasted Revenue",
            line=dict(color="#dc2626", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title="Sales Revenue Forecast",
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Revenue ($)",
    )
    return fig
