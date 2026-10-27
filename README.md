# House Price Prediction Using Linear Regression

## 1. Project overview

This project trains a **Linear Regression** model to estimate a house's sale price from three predictors:

- **Living area (`GrLivArea`)** — above-ground living area in square feet.
- **Bedrooms (`BedroomAbvGr`)** — bedrooms above ground.
- **Total bathrooms (`TotalBaths`)** — full bathrooms count as 1 and half-baths count as 0.5; basement bathrooms are included.

The target is **`SalePrice`**, the recorded sale price in US dollars. The dataset is Kaggle's *House Prices - Advanced Regression Techniques* dataset, based on house sales in Ames, Iowa. The task intentionally uses only three predictors so the model matches the internship brief and remains easy to explain.

## 2. Project contents

```text
house_price_linear_regression_project/
├── house_price_prediction.ipynb # Step-by-step analysis notebook
├── app.py                 # Optional interactive Streamlit application
├── train_model.py         # Data preparation, model training, evaluation, charts
├── predict.py             # Predict from command-line inputs
├── requirements.txt       # Python dependencies
├── README.md              # Setup and use guide
├── PROJECT_REPORT_TEMPLATE.md
├── data/
│   ├── train.csv          # Labeled training data (included)
│   ├── test.csv           # Competition test data (included)
│   ├── sample_submission.csv
│   ├── data_description.txt
│   └── README.md           # Dataset notes
└── artifacts/             # Generated model, metrics, charts, coefficients
```

The dataset archive you supplied has been included under `data/`: `train.csv`, `test.csv`, `sample_submission.csv`, and `data_description.txt`. No additional dataset download is needed to train the model.

## 3. Tools and libraries

- Python 3.10 or later
- pandas and NumPy for data handling
- scikit-learn for splitting, preprocessing, linear regression, and evaluation
- Matplotlib for evaluation charts
- joblib for saving the trained model
- Streamlit for the optional interactive interface

## 4. Setup

### Windows (VS Code)

1. Install Python 3.10+ and VS Code. During Python installation, enable **Add Python to PATH**.
2. Extract the project ZIP and open the extracted folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt and run `.venv\\Scripts\\activate.bat`, or use `py -m pip` commands directly.
5. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. The supplied dataset is already included at `data/train.csv`. Check that this file exists before training. Do not use `data/test.csv` for model evaluation because it has no `SalePrice` target.

Official dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

## 5. Train and evaluate the model

From the project root, run:

```powershell
python train_model.py
```

The script:

1. Loads and validates `data/train.csv`.
2. Creates `TotalBaths` using full and half bathrooms, including basement bathrooms.
3. Uses an 80/20 train/test split with `random_state=42` so the split is reproducible.
4. Handles missing feature values with median imputation.
5. Fits scikit-learn's `LinearRegression` estimator.
6. Calculates MAE, RMSE, and R² on the held-out test set.
7. Saves the fitted model, metrics, coefficients, and diagnostic plots in `artifacts/`.

Expected generated files:

- `artifacts/house_price_model.joblib`
- `artifacts/metrics.json`
- `artifacts/model_coefficients.csv`
- `artifacts/actual_vs_predicted.png`
- `artifacts/residuals.png`

To point to a CSV elsewhere:

```powershell
python train_model.py --data "C:\path\to\train.csv" --output-dir artifacts
```

## 6. Predict from the terminal

Train the model first, then run an example:

```powershell
python predict.py --area 1500 --bedrooms 3 --bathrooms 2
```

Here, bathrooms means full-bath equivalents; for example, one full bath plus one half-bath is `1.5`. The output is an estimate in USD because that is the unit used by the target in the Kaggle dataset.

## 7. Run the interactive app (optional)

After training:

```powershell
streamlit run app.py
```

Streamlit will show a local URL in the terminal, usually `http://localhost:8501`. Open that address in a browser, enter the three features, and select **Estimate price**.

## 8. How to interpret the metrics

- **MAE (Mean Absolute Error):** average absolute difference between actual and predicted prices, in USD. Smaller is better.
- **RMSE (Root Mean Squared Error):** similar to MAE but penalizes larger errors more heavily, in USD. Smaller is better.
- **R² (coefficient of determination):** proportion of target variation explained by the model on the held-out data. Values closer to 1 indicate a better fit; it can be negative if the model performs worse than predicting the test-set mean.

The exact values are generated when you run the model. Do not copy made-up metric values into the internship report.

## 9. Charts and model diagnostics

- `actual_vs_predicted.png`: points closer to the diagonal line indicate more accurate predictions.
- `residuals.png`: residuals are actual minus predicted price. A random cloud around zero is generally preferable; visible patterns can indicate model limitations.
- `model_coefficients.csv`: estimated linear coefficient for each predictor and the intercept. A coefficient represents the model's expected change in predicted price per one-unit increase in that feature, holding the other included features constant. It is an association in this model, not proof of causation.

## 8. Results from the included dataset

The project was run against the included `data/train.csv` (1,460 labeled records) using an 80/20 split and `random_state=42`. The held-out split contains 292 records. The run produced:

- **MAE:** $34,395.80
- **RMSE:** $51,222.58
- **R²:** 0.6579

These metrics are specific to this split and this three-feature baseline. Review `artifacts/metrics.json` and the diagnostic plots for details. The dataset is historical housing data from Ames, Iowa, so predictions should not be interpreted as current real-estate values in India.

## 10. Important limitations

This is a beginner-friendly baseline, not a production valuation tool. It intentionally excludes location, neighborhood, construction quality, home age, lot size, garage, and many other factors that influence price. Linear regression also assumes a linear relationship between predictors and expected price. Predictions may be less accurate for unusual houses or values outside the training range. The data is historical and specific to Ames, Iowa, so its estimates should not be interpreted as current prices in India or another market.

## 11. Suggested internship deliverables

1. This source-code project.
2. A short report using `PROJECT_REPORT_TEMPLATE.md`, completed with your own measured metrics.
3. Screenshots of the trained-model metrics, both charts, and the Streamlit prediction page.
4. A brief explanation of the dataset, three input features, model equation, evaluation metrics, and limitations.

## 12. References

- Kaggle dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
- scikit-learn `train_test_split`: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- scikit-learn LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
