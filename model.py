"""Sales forecasting models using scikit-learn."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "sales_model.pkl"


class SalesForecastModel:
    """Train and predict monthly sales revenue using ML regressors."""

    def __init__(self) -> None:
        self.linear_model = LinearRegression()
        self.rf_model = RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=10
        )
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_columns: list[str] = []
        self.best_model_name: str = "Random Forest"
        self.metrics: dict[str, float] = {}

    def _prepare_monthly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate to monthly level and engineer lag features."""
        monthly = (
            df.groupby(["YearMonth", "Region", "Product_Category"], as_index=False)
            .agg(Revenue=("Revenue", "sum"), Quantity=("Quantity", "sum"))
            .sort_values(["Region", "Product_Category", "YearMonth"])
        )

        monthly["Month"] = pd.to_datetime(monthly["YearMonth"]).dt.month
        monthly["Previous_Revenue"] = monthly.groupby(
            ["Region", "Product_Category"]
        )["Revenue"].shift(1)
        monthly["Previous_Sales"] = monthly.groupby(
            ["Region", "Product_Category"]
        )["Quantity"].shift(1)

        monthly = monthly.dropna(
            subset=["Previous_Revenue", "Previous_Sales"]
        ).reset_index(drop=True)

        return monthly

    def _encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Label-encode categorical columns."""
        encoded = df.copy()
        for col in ["Region", "Product_Category"]:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                encoded[col] = self.label_encoders[col].fit_transform(encoded[col])
            else:
                encoded[col] = self.label_encoders[col].transform(encoded[col])
        return encoded

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Train Linear Regression and Random Forest models.

        Returns:
            Evaluation metrics for both models.
        """
        monthly = self._prepare_monthly_features(df)
        monthly = self._encode_features(monthly, fit=True)

        self.feature_columns = [
            "Month",
            "Region",
            "Product_Category",
            "Previous_Revenue",
            "Previous_Sales",
        ]
        X = monthly[self.feature_columns]
        y = monthly["Revenue"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.linear_model.fit(X_train, y_train)
        self.rf_model.fit(X_train, y_train)

        lr_pred = self.linear_model.predict(X_test)
        rf_pred = self.rf_model.predict(X_test)

        lr_rmse = float(np.sqrt(mean_squared_error(y_test, lr_pred)))
        rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_pred)))

        self.metrics = {
            "linear_rmse": lr_rmse,
            "linear_mae": float(mean_absolute_error(y_test, lr_pred)),
            "linear_r2": float(r2_score(y_test, lr_pred)),
            "rf_rmse": rf_rmse,
            "rf_mae": float(mean_absolute_error(y_test, rf_pred)),
            "rf_r2": float(r2_score(y_test, rf_pred)),
        }

        self.best_model_name = (
            "Random Forest" if rf_rmse <= lr_rmse else "Linear Regression"
        )
        return self.metrics

    def predict_future(
        self, df: pd.DataFrame, periods: int = 6
    ) -> pd.DataFrame:
        """
        Generate future revenue predictions for the next N months.

        Uses the best-performing model from training.
        """
        monthly = self._prepare_monthly_features(df)
        monthly = self._encode_features(monthly, fit=False)

        model = self.rf_model if self.best_model_name == "Random Forest" else self.linear_model

        # Aggregate overall monthly revenue for forecast display
        overall_monthly = (
            df.groupby("YearMonth", as_index=False)["Revenue"]
            .sum()
            .sort_values("YearMonth")
        )

        last_date = pd.to_datetime(overall_monthly["YearMonth"].iloc[-1])
        future_months = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=periods,
            freq="MS",
        )

        # Use latest segment averages for prediction baseline
        latest = monthly.iloc[-1]
        predictions = []
        prev_revenue = latest["Previous_Revenue"]
        prev_sales = latest["Previous_Sales"]

        for future_date in future_months:
            features = pd.DataFrame(
                [
                    {
                        "Month": future_date.month,
                        "Region": latest["Region"],
                        "Product_Category": latest["Product_Category"],
                        "Previous_Revenue": prev_revenue,
                        "Previous_Sales": prev_sales,
                    }
                ]
            )
            pred = float(model.predict(features[self.feature_columns])[0])
            pred = max(pred, 0)
            predictions.append(
                {
                    "YearMonth": future_date.strftime("%Y-%m"),
                    "Predicted_Revenue": round(pred, 2),
                }
            )
            prev_revenue = pred
            prev_sales = prev_sales * 0.95

        return pd.DataFrame(predictions)

    def save(self, path: Path | None = None) -> None:
        """Persist trained model to disk."""
        save_path = path or MODEL_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "linear_model": self.linear_model,
            "rf_model": self.rf_model,
            "label_encoders": self.label_encoders,
            "feature_columns": self.feature_columns,
            "best_model_name": self.best_model_name,
            "metrics": self.metrics,
        }
        with open(save_path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path | None = None) -> "SalesForecastModel":
        """Load a persisted model from disk."""
        load_path = path or MODEL_PATH
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        instance = cls()
        with open(load_path, "rb") as f:
            payload = pickle.load(f)

        instance.linear_model = payload["linear_model"]
        instance.rf_model = payload["rf_model"]
        instance.label_encoders = payload["label_encoders"]
        instance.feature_columns = payload["feature_columns"]
        instance.best_model_name = payload["best_model_name"]
        instance.metrics = payload["metrics"]
        return instance


def train_and_save_model(df: pd.DataFrame, path: Path | None = None) -> SalesForecastModel:
    """Convenience function to train model and save to disk."""
    model = SalesForecastModel()
    model.train(df)
    model.save(path)
    return model
