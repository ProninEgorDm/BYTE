# Tabular Modeling

This folder contains Jupyter Notebooks for exploratory data analysis (EDA) and baseline model training on the cleaned Yandex Realty dataset.

## Files

- EDA.ipynb: Performs exploratory data analysis, including correlation heatmaps, unique value counts, and basic statistics on numeric and categorical features.
- baseline.ipynb: Trains a baseline CatBoost regressor for predicting apartment prices. Includes hyperparameter optimization with Optuna, model evaluation, and saving the trained model.

## Data

The notebooks use the cleaned dataset located at yandex_realty_cleaned.parquet.

Key features used:
- Numeric: `area`, `metro_time`, `photo_count`, `self_floor`, `max_floor`
- Categorical: `metro`, `title`, `author`, `description`, `address`

Target: `price_numeric`

## Dependencies

Install required packages:

```
pip install polars pandas matplotlib seaborn scikit-learn catboost optuna nbformat
```

## Usage

1. Ensure the data file exists.
2. Open the notebooks in Jupyter and run cells sequentially.
3. For baseline.ipynb, the model is trained with early stopping and saved as `catb_all_features_with_optim_60K.cbm`.

## Notes

- Models are evaluated using R², MAE, MAPE, and RMSE.
- Hyperparameter tuning is performed with Optuna for 100 trials.
- Text features are handled via CatBoost's built-in text processing.