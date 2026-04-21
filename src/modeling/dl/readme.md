# Deep Learning Modeling

This folder contains a Jupyter Notebook for deep learning-based modeling on the Yandex Realty dataset, implementing a custom PyTorch model for apartment price prediction.

## Files

- dl_modeling.ipynb: Implements a deep learning model using PyTorch, including data preprocessing, model training, and evaluation. The model combines text embeddings, categorical embeddings, and numeric features through a transformer-based architecture.

## Architecture

The model (`RoyaltyModel`) consists of:

- **Text Encoder**: Uses `cointegrated/rubert-tiny2` (a small BERT-based model) to encode descriptions and addresses into 312-dimensional vectors, then reduces to 128 dimensions via MLPs.
- **Embeddings**: Categorical features (`rooms`, `metro`, `title`, `self_floor`, `max_floor`) are embedded into 128-dimensional vectors.
- **Numeric Processing**: Area is processed through an MLP to 128 dimensions.
- **Transformer Blocks**: A stack of 3 RoPE (Rotary Position Embedding) Transformer blocks with 8 heads, 512 FFN dimensions, and LayerNorm.
- **Sequence**: Inputs are concatenated into a sequence (description, address, area, categorical embeddings) and processed by the transformers.
- **Head**: A 2-layer MLP (128 -> 128 -> 1) for regression output.

The model uses RoPE for positional encoding in the transformer blocks. Targets are log-transformed for training, and predictions are exponentiated back.

## Data

Uses the cleaned dataset at yandex_realty_cleaned.parquet.

Features:
- Numeric: `area`
- Categorical: `rooms`, `metro`, `title`, `self_floor`, `max_floor`
- Text: `description`, `address`

Target: `price_numeric` (log-transformed during training).

## Dependencies

Install required packages:

```
pip install pandas torch torchvision transformers scikit-learn matplotlib tqdm
```

## Usage

1. Open dl_modeling.ipynb in Jupyter.
2. Run cells to load data, preprocess (tokenize text, encode categories), create datasets/loaders, initialize the model, and train for 10 epochs.
3. Training uses AdamW optimizer with weight decay, MSE loss, gradient clipping, and cosine annealing LR scheduler.
4. Evaluation on validation set with R², MAE, MAPE, RMSE.

## Notes

- Trained on GPU if available; otherwise CPU.
- Batch size: 128 (train), 100 (val).
- Metrics reported per epoch: Train Loss, Val MSE, Val MAE, Val R², Val MAPE, Current LR.
- Text encoder parameters are frozen.