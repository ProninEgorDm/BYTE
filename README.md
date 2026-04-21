# BYTE - Real Estate Price Prediction

A comprehensive machine learning project for predicting apartment prices using Yandex Realty data. Combines web scraping, data engineering, exploratory analysis, and multiple modeling approaches (traditional ML and deep learning).

---

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Data Features](#data-features)
- [Getting Started](#getting-started)
- [Model Performance](#model-performance)
- [Configuration](#configuration)
- [Notes](#notes)

---

## 🏗️ Project Structure

```
BYTE/
├── src/
│   ├── parsing/                      # Web scraping pipeline
│   │   ├── ya_parser_raw.ipynb
│   │   └── readme.md
│   ├── data/                         # Data processing & storage
│   │   ├── s3_client.py              # MinIO S3 client
│   │   ├── pandera.ipynb             # Validation & cleaning
│   │   └── tabular/
│   │       ├── yandex_realty_manual.csv
│   │       ├── yandex_realty_cleaned.parquet
│   │       └── ya_realty_with_txt_embeds.parquet
│   ├── modeling/
│   │   ├── tabular/                  # Traditional ML models
│   │   │   ├── EDA.ipynb
│   │   │   ├── baseline.ipynb
│   │   │   └── readme.md
│   │   ├── text/                     # Text-based models
│   │   │   ├── encoding.ipynb
│   │   │   ├── catboost_with_text.ipynb
│   │   │   ├── catb_text_60K.cbm
│   │   │   └── readme.md
│   │   └── dl/                       # Deep learning models
│   │       ├── dl_modeling.ipynb
│   │       └── readme.md
│   └── RAG/                          # Semantic search
│       └── building_index.ipynb
├── docker-compose.yml                # MinIO container setup
└── README.md
```

---

## 📊 Pipeline Overview

### 1️⃣ Data Collection (`src/parsing/`)

**Yandex Realty Web Scraper**
- Scrapes apartment listings using Selenium
- Handles dynamic content and CAPTCHA detection
- Extracts 19+ features per listing (price, area, rooms, metro distance, etc.)
- Incremental parsing with duplicate detection
- **Output**: `yandex_realty_manual.csv`

### 2️⃣ Data Cleaning & Validation (`src/data/`)

**Data Processing Pipeline**
- Validates data using Pandera schemas
- Normalizes values (prices, areas, floors, metro times)
- Handles missing values and duplicates
- Generates invalid records report
- **Output**: `yandex_realty_cleaned.parquet`

### 3️⃣ Data Storage (`src/data/s3_client.py`)

**MinIO S3-Compatible Storage**
- Stores tabular data (Parquet/CSV)
- Manages text embeddings
- Indexes photos and images
- Upload/download with auto directory creation
- Bucket management and object listing

### 4️⃣ Exploratory Data Analysis (`src/modeling/tabular/EDA.ipynb`)

**Statistical Analysis**
- Correlation matrices and distribution plots
- Feature statistics and missing value analysis
- Categorical value counts and distributions

### 5️⃣ Model Training

#### **Tabular Models** (`src/modeling/tabular/`)
- **Algorithm**: CatBoost
- **Features**: Numeric & categorical only
- **Optimization**: Hyperparameter tuning via Optuna (100 trials)
- **Early Stopping**: Validation set monitoring
- **Output**: `catb_all_features_with_optim.cbm`

| Metric | Value |
|--------|-------|
| R² | — |
| MAE | — |
| MAPE | — |
| RMSE | — |

#### **Text-Enhanced Models** (`src/modeling/text/`)
- **Algorithm**: CatBoost with text features
- **Features**: Descriptions, addresses, metro, badges
- **Encoding**: Native CatBoost text handling
- **Output**: `catb_text_60K.cbm`

#### **Deep Learning** (`src/modeling/dl/`)
- **Architecture**: Multi-modal Transformer
  - Text Encoder: `cointegrated/rubert-tiny2` (RuBERT)
  - Categorical Embeddings: 128D
  - Numeric Features: Direct input (area)
  - Transformer: 3 layers, 8 attention heads, RoPE
  - Head: 2-layer MLP for regression
- **Training**: 10 epochs, AdamW optimizer, cosine annealing
- **Batch Size**: 128 (train) / 100 (val)
- **Target**: Log-transformed prices

### 6️⃣ RAG Pipeline (`src/RAG/`)

**Semantic Search & Retrieval**
- Builds vector index for semantic similarity
- Supports retrieval-augmented generation tasks

---

## 📈 Data Features

### Numeric Features
| Feature | Description | Unit |
|---------|-------------|------|
| `price_numeric` | Apartment price | RUB |
| `area` | Total area | m² |
| `metro_time` | Distance to nearest metro | minutes |
| `photo_count` | Number of listing photos | count |
| `self_floor` | Current floor | floor |
| `max_floor` | Building height | floor |

### Categorical Features
| Feature | Description | Values |
|---------|-------------|--------|
| `rooms` | Number of rooms | 0 (studio), 1+, multiroom |
| `metro` | Nearest metro station | string |
| `author` | Seller type | agency, owner, realtor |
| `title` | Apartment type | string |

### Text Features
| Feature | Description |
|---------|-------------|
| `description` | Full listing description |
| `address` | Complete address |
| `badges` | Special features (new building, etc.) |

---

## 🚀 Getting Started

### Prerequisites

```bash
# Core ML & Data Processing
pip install pandas polars catboost scikit-learn optuna

# Web Scraping
pip install selenium beautifulsoup4 fake-useragent lxml

# Deep Learning
pip install torch transformers

# Storage & Validation
pip install boto3 minio pandera
```

### Setup MinIO Storage

```bash
docker-compose up -d
```

**Access Credentials**:
- **Web Console**: http://localhost:9001
- **Access Key**: `minioadmin`
- **Secret Key**: `minioadmin123`

### Run the Pipeline

#### Step 1: Scrape Data
```bash
# Open: src/parsing/ya_parser_raw.ipynb
# Configure GRID_PARAMS for desired areas and floor ranges
# Run all cells
```

#### Step 2: Clean & Validate
```bash
# Open: src/data/pandera.ipynb
# Run main() to clean and validate data
```

#### Step 3: Upload to Storage
```python
from src.data.s3_client import MinIOS3Client

client = MinIOS3Client()
client.create_bucket('tabular-data')
client.create_bucket('embeddings')
client.create_bucket('photos')

client.upload_dataframe(
    df, 
    'tabular-data', 
    'yandex_realty_cleaned.parquet'
)
```

#### Step 4: Train Models
```bash
# Tabular baseline
src/modeling/tabular/baseline.ipynb

# Text-enhanced model
src/modeling/text/catboost_with_text.ipynb

# Deep learning model
src/modeling/dl/dl_modeling.ipynb
```

---

## 📈 Model Evaluation

All models evaluated on 60K+ records using:

- **R²**: Coefficient of determination (goodness of fit)
- **MAE**: Mean Absolute Error (average prediction error)
- **MAPE**: Mean Absolute Percentage Error (% error)
- **RMSE**: Root Mean Squared Error (penalizes large errors)

---

## 🔧 Configuration

### Scraper (`src/parsing/ya_parser_raw.ipynb`)

```python
BASE_URL = "https://realty.yandex.ru/moskva_i_moskovskaya_oblast/kupit/kvartira/"

GRID_PARAMS = [
    {"areaMin": "150", "areaMax": "151", "floorMin": "0", "floorMax": "3"},
    # ... additional parameter combinations
]

pages_per_param = 25
save_interval = 5  # Save every N pages
```

### S3 Client (`src/data/s3_client.py`)

```python
from src.data.s3_client import MinIOS3Client

client = MinIOS3Client(
    endpoint_url='http://localhost:9000',
    access_key='minioadmin',
    secret_key='minioadmin123'
)
```

### Model Training

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Optuna Trials | 100 | Hyperparameter search space |
| Early Stopping | MAE | Validation metric |
| LR Scheduler | Cosine Annealing | Learning rate decay |
| Gradient Clip | 1.0 (norm) | Training stability |

---

## ℹ️ Important Notes

- **Timezone**: All timestamps in UTC+3 (Moscow Standard Time)
- **Price Scaling**: Log-transformed during DL training for better convergence
- **Text Embeddings**: Frozen from pre-trained RuBERT
- **Incremental Loading**: Supports resuming from checkpoints
- **Auto-creation**: Directories created automatically on upload
- **Indexing**: Photos indexed by listing URL for traceability

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Please respect:
- Yandex Realty's Terms of Service
- `robots.txt` guidelines
- Rate limiting and responsible scraping practices

---

## 📄 License

Educational use only.