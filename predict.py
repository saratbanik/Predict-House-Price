"""Predict one house price using the saved model."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a house price from three features.")
    parser.add_argument("--area", type=float, required=True, help="Above-ground living area in square feet")
    parser.add_argument("--bedrooms", type=float, required=True, help="Number of above-ground bedrooms")
    parser.add_argument("--bathrooms", type=float, required=True, help="Total bathroom equivalent; half-baths count as 0.5")
    parser.add_argument("--model", default="artifacts/house_price_model.joblib", help="Saved model path")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit("Saved model not found. Run: python train_model.py")
    if args.area <= 0 or args.bedrooms < 0 or args.bathrooms < 0:
        raise SystemExit("Area must be greater than zero; bedrooms and bathrooms cannot be negative.")

    bundle = joblib.load(model_path)
    X_new = pd.DataFrame(
        [[args.area, args.bedrooms, args.bathrooms]],
        columns=bundle["feature_columns"],
    )
    prediction = float(bundle["model"].predict(X_new)[0])
    print(f"Estimated house sale price: ${prediction:,.2f} USD")
    print("Note: This is a model estimate, not a professional property valuation.")


if __name__ == "__main__":
    main()
