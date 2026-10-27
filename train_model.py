"""Train and evaluate a linear regression model for house-price prediction.

Expected input: Kaggle House Prices - Advanced Regression Techniques train.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SOURCE_FEATURES = [
    "GrLivArea",
    "BedroomAbvGr",
    "FullBath",
    "HalfBath",
    "BsmtFullBath",
    "BsmtHalfBath",
]
FEATURE_COLUMNS = ["GrLivArea", "BedroomAbvGr", "TotalBaths"]
TARGET = "SalePrice"


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the three required predictors from the official Kaggle columns.

    Half-baths count as 0.5 of a full bath. Basement bathrooms are included.
    Missing bathroom component values are treated as zero; other missing
    predictor values are handled by the model pipeline's median imputer.
    """
    missing = [column for column in SOURCE_FEATURES if column not in df.columns]
    if missing:
        raise ValueError(
            "The CSV is missing required columns: " + ", ".join(missing)
            + ". Use train.csv from Kaggle's House Prices - Advanced Regression Techniques dataset."
        )

    result = pd.DataFrame(index=df.index)
    result["GrLivArea"] = pd.to_numeric(df["GrLivArea"], errors="coerce")
    result["BedroomAbvGr"] = pd.to_numeric(df["BedroomAbvGr"], errors="coerce")

    bath_columns = ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
    baths = df[bath_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    result["TotalBaths"] = (
        baths["FullBath"]
        + 0.5 * baths["HalfBath"]
        + baths["BsmtFullBath"]
        + 0.5 * baths["BsmtHalfBath"]
    )
    return result[FEATURE_COLUMNS]


def save_plots(y_test: pd.Series, y_pred: np.ndarray, output_dir: Path) -> None:
    # Actual versus predicted values
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred, alpha=0.65, edgecolors="none")
    lower = float(min(y_test.min(), np.min(y_pred)))
    upper = float(max(y_test.max(), np.max(y_pred)))
    ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.5)
    ax.set_title("Actual vs. Predicted House Prices")
    ax.set_xlabel("Actual sale price (USD)")
    ax.set_ylabel("Predicted sale price (USD)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    # Residual plot
    residuals = np.asarray(y_test) - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.65, edgecolors="none")
    ax.axhline(0, linestyle="--", linewidth=1.5)
    ax.set_title("Residuals vs. Predicted Prices")
    ax.set_xlabel("Predicted sale price (USD)")
    ax.set_ylabel("Residual (actual - predicted, USD)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "residuals.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a house-price linear regression model.")
    parser.add_argument("--data", default="data/train.csv", help="Path to Kaggle train.csv")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for model, metrics, and plots")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    if not data_path.is_file():
        raise SystemExit(
            f"Dataset not found: {data_path}\n"
            "Download train.csv from Kaggle and place it at data/train.csv, "
            "or pass its location with --data."
        )

    df = pd.read_csv(data_path)
    if TARGET not in df.columns:
        raise SystemExit(
            f"Target column '{TARGET}' was not found. Use Kaggle's train.csv; "
            "the competition test.csv does not contain SalePrice."
        )

    y = pd.to_numeric(df[TARGET], errors="coerce")
    valid_target = y.notna()
    df = df.loc[valid_target].copy()
    y = y.loc[valid_target].astype(float)
    X = make_features(df)

    if len(df) < 10:
        raise SystemExit("The training file has too few valid rows. Check that the full train.csv was downloaded.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "model": "Linear Regression",
        "dataset": str(data_path),
        "rows_used": int(len(df)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURE_COLUMNS,
        "target": TARGET,
        "split": "80% training / 20% testing",
        "random_state": 42,
        "MAE_USD": round(float(mean_absolute_error(y_test, y_pred)), 2),
        "RMSE_USD": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "R2": round(float(r2_score(y_test, y_pred)), 4),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": FEATURE_COLUMNS, "metrics": metrics},
        output_dir / "house_price_model.joblib",
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    regressor = model.named_steps["regressor"]
    coefficients = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "coefficient_USD_per_unit": regressor.coef_}
    )
    coefficients.loc[len(coefficients)] = ["intercept", regressor.intercept_]
    coefficients.to_csv(output_dir / "model_coefficients.csv", index=False)
    save_plots(y_test, y_pred, output_dir)

    print("\nTraining completed successfully.\n")
    print(pd.Series(metrics).to_string())
    print(f"\nSaved model: {output_dir / 'house_price_model.joblib'}")
    print(f"Saved evaluation plots and coefficients in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
