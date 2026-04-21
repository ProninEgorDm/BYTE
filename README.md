````markdown
# BYTE - Real Estate Price Prediction

A comprehensive machine learning project for predicting apartment prices using Yandex Realty data. The project includes web scraping, data cleaning, exploratory analysis, and multiple modeling approaches (traditional ML and deep learning).

## 🏗️ Project Structure

```
BYTE/
├── src/
│   ├── parsing/              # Web scraping pipeline
│   │   ├── ya_parser_raw.ipynb
│   │   └── readme.md
│   ├── data/                 # Data processing and storage
│   │   ├── s3_client.py      # MinIO S3 client
│   │   ├── pandera.ipynb     # Data validation & cleaning
│   │   └── tabular/
│   │       ├── yandex_realty_manual.csv
│   │       ├── yandex_realty_cleaned.parquet
│   │       └── ya_realty_with_txt_embeds.parquet
│   ├── modeling/
│   │   ├── tabular/          # Traditional ML models
│   │   │   ├── EDA.ipynb
│   │   │   ├── baseline.ipynb
│   │   │   └── readme.md
│   │   ├── text/             # Text-based models
│   │   │   ├── encoding.ipynb
│   │   │   ├── catboost_with_text.ipynb
│   │   │   ├── catb_text_60K.cbm
│   │   │   └── readme.md
│   │   └── dl/               # Deep learning models
│   │       ├── dl_modeling.ipynb
│   │       └── readme.md
│   └── RAG/
│       └── building_index.ipynb
├── docker-compose.yml        # MinIO container setup
└── README.md
```

## 📋 Pipeline Overview

### 1. **Data Collection** (`src/parsing/`)
- Scrapes apartment listings from Yandex Realty using Selenium
- Handles dynamic content and CAPTCHA detection
- Extracts 19+ features per listing (price, area, rooms, metro, etc.)
- Supports incremental parsing with duplicate detection
- Output: `yandex_realty_manual.csv`

### 2. **Data Cleaning** (`src/data/`)
- Validates data using Pandera schemas
- Cleans and normalizes values (prices, areas, floors, metro times)
- Handles missing values and duplicates
- Generates invalid records report
- Output: `yandex_realty_cleaned.parquet`

### 3. **Data Storage** (`src/data/s3_client.py`)
- MinIO S3-compatible storage for:
  - Tabular data (Parquet/CSV)
  - Text embeddings
  - Photos/images
- Supports upload/download with automatic directory creation
- Bucket management and object listing

### 4. **Exploratory Data Analysis** (`src/modeling/tabular/`)
- Correlation analysis and distribution plots
- Feature statistics and missing value analysis
- Unique value counts for categorical features

### 5. **Modeling**

#### **Tabular Models** (`src/modeling/tabular/`)
- CatBoost baseline with all numeric/categorical features
- Hyperparameter optimization via Optuna (100 trials)
- Metrics: R², MAE, MAPE, RMSE
- Early stopping on validation set
- Model: `catb_all_features_with_optim.cbm`

#### **Text Models** (`src/modeling/text/`)
- CatBoost with text features (descriptions, addresses, metro)
- Text feature encoding and preprocessing
- Native CatBoost text handling
- Model: `catb_text_60K.cbm`

#### **Deep Learning** (`src/modeling/dl/`)
- **Architecture**: Multi-modal transformer-based model
  - Text encoder: `cointegrated/rubert-tiny2` (BERT)
  - Categorical embeddings (128D)
  - Numeric processing (area)
  - 3-layer RoPE Transformer with 8 heads
  - 2-layer MLP head for regression
- **Training**: 10 epochs, AdamW optimizer, cosine annealing LR
- **Batch size**: 128 (train), 100 (val)
- Log-transformed targets for improved convergence

### 6. **RAG Pipeline** (`src/RAG/`)
- Building vector index for semantic search
- Supports retrieval-augmented generation tasks

## 📊 Data Features

### Numeric Features
- `price_numeric`: Price in rubles
- `area`: Apartment area in m²
- `metro_time`: Time to nearest metro (minutes)
- `photo_count`: Number of photos
- `self_floor`: Floor number
- `max_floor`: Total floors in building

### Categorical Features
- `rooms`: Number of rooms (0=studio, 1+, or "multiroom")
- `metro`: Nearest metro station
- `author`: Seller type (agency, owner, realtor)
- `title`: Apartment type/description
- `self_floor`, `max_floor`: Floor position

### Text Features
- `description`: Listing description
- `address`: Full address
- `badges`: Features (new building, online viewing, etc.)

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas polars
pip install selenium beautifulsoup4 fake-useragent lxml
pip install catboost scikit-learn optuna
pip install torch transformers
pip install boto3 minio
pip install pandera
```

### Setup MinIO Storage
```bash
docker-compose up -d
```
This starts MinIO on `http://localhost:9000` with credentials:
- Access Key: `minioadmin`
- Secret Key: `minioadmin123`
- Console: `http://localhost:9001`

### Run Pipeline

1. **Scrape Data**
   ```bash
   # Open src/parsing/ya_parser_raw.ipynb
   # Configure GRID_PARAMS for area/floor ranges
   # Run all cells
   ```

2. **Clean & Validate**
   ```bash
   # Open src/data/pandera.ipynb
   # Run main() to clean data
   ```

3. **Upload to Storage**
   ```bash
   # Initialize client and create buckets
   client = MinIOS3Client()
   client.create_bucket('tabular-data')
   client.create_bucket('embeddings')
   client.create_bucket('photos')
   
   # Upload data
   client.upload_dataframe(df, 'tabular-data', 'yandex_realty_cleaned.parquet')
   ```

4. **Train Models**
   ```bash
   # Tabular baseline
   src/modeling/tabular/baseline.ipynb
   
   # Text-based
   src/modeling/text/catboost_with_text.ipynb
   
   # Deep learning
   src/modeling/dl/dl_modeling.ipynb
   ```

## 📈 Model Performance

All models evaluated on validation set (60K+ records) using:
- **R²**: Coefficient of determination
- **MAE**: Mean absolute error
- **MAPE**: Mean absolute percentage error
- **RMSE**: Root mean squared error

## 🔧 Configuration

### Scraper (`ya_parser_raw.ipynb`)
```python
BASE_URL = "https://realty.yandex.ru/moskva_i_moskovskaya_oblast/kupit/kvartira/"
GRID_PARAMS = [{"areaMin": "150", "areaMax": "151", "floorMin": "0", "floorMax": "3"}, ...]
pages_per_param = 25
save_interval = 5
```

### S3 Client (`s3_client.py`)
```python
client = MinIOS3Client(
    endpoint_url='http://localhost:9000',
    access_key='minioadmin',
    secret_key='minioadmin123'
)
```

### Model Training
- **Optuna trials**: 100
- **Early stopping**: On validation MAE
- **LR scheduler**: Cosine annealing
- **Gradient clipping**: norm=1.0

## 📝 Notes

- All timestamps are in UTC+3 (Moscow)
- Prices stored in log scale during DL training
- Text embeddings frozen from pre-trained BERT
- Supports incremental data loading
- Auto-creates directories on upload
- Photos indexed by listing URL

## ⚠️ Disclaimer

This project is for educational purposes. Respect Yandex Realty's terms of service and robots.txt. Use responsible scraping practices and rate limiting.

## 📄 License

Educational use only.
````