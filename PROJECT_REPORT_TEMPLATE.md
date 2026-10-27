# Internship Project Report: House Price Prediction Using Linear Regression

**Intern:** [Your Name]  
**Internship duration:** 1 month  
**Task:** Task 1 of 5  
**Project type:** Supervised machine learning — regression

## 1. Abstract

This project develops a baseline machine-learning model to estimate house sale prices from living area, bedroom count, and total bathroom count. The model is trained with linear regression using the Kaggle House Prices - Advanced Regression Techniques training data. It is evaluated on a held-out test split using MAE, RMSE, and R².

## 2. Problem statement

House sale prices depend on multiple property characteristics. The goal of this task is to examine whether a simple linear regression model can estimate sale price from three selected features: above-ground living area, bedrooms, and bathroom equivalents.

## 3. Objectives

- Load and inspect the supplied housing dataset.
- Select the target variable and three required predictors.
- Prepare the data and handle missing predictor values.
- Train a linear regression model.
- Evaluate performance with MAE, RMSE, and R².
- Save the model and demonstrate a prediction with an optional interface.

## 4. Dataset

**Source:** Kaggle, House Prices - Advanced Regression Techniques.  
**Target:** `SalePrice` (USD).  
**Features:** `GrLivArea`, `BedroomAbvGr`, and derived `TotalBaths`.  
**Rows used:** 1460 (1,168 training rows and 292 held-out test rows).

## 5. Methodology

1. Load `train.csv` and validate required columns.
2. Create `TotalBaths = FullBath + 0.5 × HalfBath + BsmtFullBath + 0.5 × BsmtHalfBath`.
3. Split the data into 80% training and 20% testing, with `random_state=42`.
4. Impute missing feature values using the median.
5. Fit a linear regression model on the training partition.
6. Predict prices for the test partition and calculate evaluation metrics.
7. Save the trained model, metric values, coefficient table, and diagnostic charts.

## 6. Model equation

`Predicted SalePrice = Intercept + (b1 × GrLivArea) + (b2 × BedroomAbvGr) + (b3 × TotalBaths)`

The model estimates the coefficients (`b1`, `b2`, `b3`) from the training data.

## 7. Results

Copy the values from `artifacts/metrics.json` after running the code.

| Metric | Result | Interpretation |
|---|---:|---|
| MAE (USD) | $34,395.80 | Average absolute prediction error |
| RMSE (USD) | $51,222.58 | Error metric that penalizes large errors more heavily |
| R² | 0.6579 | Variation explained on the held-out test split |

Insert `artifacts/actual_vs_predicted.png` and `artifacts/residuals.png` here.

## 8. Conclusion

On the held-out test split, the three-feature Linear Regression model achieved an MAE of $34,395.80, an RMSE of $51,222.58, and an R² of 0.6579. These results provide a baseline for the selected features, but prediction errors remain substantial. The model omits important price drivers such as neighborhood, quality, age, and condition, so it should be treated as an educational baseline rather than a formal valuation tool.

## 9. Limitations and future improvements

The model uses only three predictors and historical home-sale data from Ames, Iowa. Potential improvements include comparing Ridge and Lasso regression, adding property quality and location-related features, applying cross-validation, examining outliers, and comparing against tree-based models. These should be treated as future work beyond the current task scope.

## 10. References

- Kaggle: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
- scikit-learn documentation: https://scikit-learn.org/stable/
