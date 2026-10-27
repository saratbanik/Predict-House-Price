# Dataset files

These files were supplied in the Kaggle *House Prices - Advanced Regression Techniques* dataset archive.

- `train.csv`: 1,460 labeled house records, including `SalePrice`. Used to train and evaluate this project.
- `test.csv`: 1,459 competition records. It does not include `SalePrice`, so it is not used for the held-out evaluation in this project.
- `sample_submission.csv`: Kaggle's example submission format.
- `data_description.txt`: descriptions of the dataset columns.

Dataset source: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

The project uses `train.csv` and selects only `GrLivArea`, `BedroomAbvGr`, and bathroom counts derived from `FullBath`, `HalfBath`, `BsmtFullBath`, and `BsmtHalfBath`.
