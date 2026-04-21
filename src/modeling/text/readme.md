# Text Modeling

This folder contains Jupyter Notebooks and a trained model for text-based modeling on the Yandex Realty dataset, focusing on text features like descriptions and addresses.

## Files

- `catboost_with_text.ipynb`: Trains a CatBoost regressor incorporating text features for price prediction. Includes hyperparameter tuning and model evaluation.
- `encoding.ipynb`: Handles text encoding and preprocessing for the dataset.
- `catb_text_60K.cbm`: Saved CatBoost model trained with text features.

## Data

Uses the cleaned dataset at yandex_realty_cleaned.parquet.

Text features: `description`, `address`, `metro`.

## Dependencies

Install required packages:

```
pip install polars pandas catboost optuna scikit-learn
```

## Usage

1. Run `encoding.ipynb` to preprocess text data.
2. Open `catboost_with_text.ipynb` and execute cells to train the model.
3. The trained model is saved as `catb_text_60K.cbm`.

## Notes

- Model evaluation metrics: R², MAE, MAPE, RMSE.
- Hyperparameter optimization with Optuna.
- CatBoost handles text features natively.