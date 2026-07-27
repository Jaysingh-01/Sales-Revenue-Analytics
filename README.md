# Sales & Revenue Analytics Dashboard

A production-ready business analytics dashboard for corporate sales data. This project delivers end-to-end data processing, exploratory analysis, interactive KPI visualization, and machine learning-based revenue forecasting.

## Project Overview

The **Sales & Revenue Analytics Dashboard** analyzes synthetic corporate sales transactions to provide actionable business insights. It includes:

- **Data Layer** — 10,000+ synthetic sales records with realistic business attributes
- **Processing Pipeline** — Cleaning, deduplication, date conversion, and KPI calculation
- **EDA Notebook** — Jupyter notebook for exploratory data analysis
- **Streamlit Dashboard** — Interactive executive summary, revenue/product/customer analytics
- **ML Forecasting** — Linear Regression and Random Forest models for revenue prediction
- **PostgreSQL Schema** — Normalized database design for production deployment
- **Docker Support** — Containerized deployment for consistent environments

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Database | PostgreSQL |
| Machine Learning | Scikit-learn |
| Deployment | Docker |

## Project Structure

```
Sales-Revenue-Analytics-Dashboard/
│
├── app.py                      # Streamlit dashboard application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── Dockerfile                  # Docker container configuration
│
├── data/
│   └── sales_data.csv          # Synthetic sales dataset (10,000+ records)
│
├── database/
│   └── schema.sql              # PostgreSQL table definitions
│
├── notebooks/
│   └── EDA.ipynb               # Exploratory data analysis notebook
│
├── models/
│   └── sales_model.pkl         # Trained forecasting model
│
└── src/
    ├── data_processing.py      # Data loading and KPI pipeline
    ├── visualization.py        # Chart builders (Plotly)
    ├── model.py                # ML forecasting module
    ├── database.py             # PostgreSQL utilities
    └── generate_data.py        # Dataset generator script
```

## Installation

### Prerequisites

- Python 3.12+
- pip
- (Optional) PostgreSQL 14+ for database features
- (Optional) Docker for containerized deployment

### Setup with Virtual Environment

```bash
# Clone or navigate to the project directory
cd Sales-Revenue-Analytics-Dashboard

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate sales dataset (if not present)
python src/generate_data.py

# Train forecasting model
python -c "from src.data_processing import preprocess_pipeline; from src.model import train_and_save_model; df, _ = preprocess_pipeline(); train_and_save_model(df)"
```

## Usage

### Run the Dashboard

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

### Dashboard Features

1. **Executive Summary** — KPI cards for Total Revenue, Profit, Orders, and AOV
2. **Revenue Analysis** — Monthly trends, regional breakdown, category pie chart
3. **Product Analysis** — Top 10 products and profitability scatter plot
4. **Customer Analysis** — Customer segmentation and sales channel performance
5. **Sales Forecasting** — ML-powered future revenue predictions
6. **Sidebar Filters** — Date range, region, category, and sales channel

### Run EDA Notebook

```bash
jupyter notebook notebooks/EDA.ipynb
```

### Database Setup (Optional)

```bash
# Create database
createdb sales_analytics

# Apply schema
psql -d sales_analytics -f database/schema.sql

# Load data into PostgreSQL
python -c "
from src.data_processing import preprocess_pipeline
from src.database import load_all_data
df, _ = preprocess_pipeline()
load_all_data(df)
"
```

Set environment variables for database connection:

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=sales_analytics
export DB_USER=postgres
export DB_PASSWORD=postgres
```

### Docker Deployment

```bash
# Build image
docker build -t sales-analytics-dashboard .

# Run container
docker run -p 8501:8501 sales-analytics-dashboard
```

Access the dashboard at **http://localhost:8501**

## Screenshots

> Add screenshots of your running dashboard here after deployment.

| Section | Description |
|---------|-------------|
| Executive Summary | KPI cards with revenue, profit, orders, AOV |
| Revenue Analysis | Line, bar, and pie charts |
| Product Analysis | Top products and profitability |
| Customer Analysis | Segmentation and channel performance |
| Forecasting | Historical vs predicted revenue |

## KPI Metrics

| Metric | Description |
|--------|-------------|
| Total Revenue | Sum of all order revenue |
| Total Profit | Sum of all order profit |
| Average Order Value | Total Revenue / Total Orders |
| Total Customers | Unique customer count |
| Monthly Revenue Growth | Month-over-month percentage change |
| Top Selling Products | Products ranked by revenue |
| Best Performing Regions | Regions ranked by revenue and profit |

## Future Improvements

- [ ] Connect to live PostgreSQL data source in the dashboard
- [ ] Add user authentication and role-based access
- [ ] Implement ARIMA/Prophet time-series forecasting
- [ ] Add export to PDF/Excel report generation
- [ ] Real-time data ingestion via Apache Airflow pipeline
- [ ] Add unit and integration tests with pytest
- [ ] Deploy to cloud (AWS ECS, Google Cloud Run, Azure Container Apps)
- [ ] Add anomaly detection for unusual sales patterns
- [ ] Multi-currency support and exchange rate conversion

## License

MIT License — free to use for learning and commercial projects.

## Author

Built as a production-level data engineering and analytics portfolio project.
