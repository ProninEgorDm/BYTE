# RAG System for Real Estate Market Analysis

Semantic search and AI-powered deal analysis for apartment listings using embeddings and vector databases.

## 📋 Overview

This system implements a complete RAG (Retrieval Augmented Generation) pipeline:

1. **Embeddings Pipeline** - Convert apartment descriptions to vectors using `sentence-transformers`
2. **Vector Database** - Store embeddings in ChromaDB for fast semantic search
3. **Semantic Search** - Find similar apartments based on text queries
4. **Deal Analysis** - Generate LLM-powered reports comparing properties to market alternatives
5. **DVC Tracking** - Version control all artifacts

## 🚀 Quick Start

### Installation

```bash
pip install chromadb sentence-transformers pandas numpy
```

### Build Index from Data

```python
from src.RAG.rag_system import RealEstateRAG
import pandas as pd

# Load your cleaned apartment data
df = pd.read_parquet('src/data/tabular/yandex_realty_cleaned.parquet')

# Initialize and build RAG system
rag = RealEstateRAG()
rag.build_index(df)
```

### Search for Similar Apartments

```python
# Natural language query
query = "2-room apartment in city center, max 10 million"
results = rag.search(query, top_k=5)

for apt in results:
    print(f"Price: ₽ {apt['metadata']['price']:,.0f}")
    print(f"Address: {apt['metadata']['address']}")
    print(f"Similarity: {apt['similarity']:.2%}\n")
```

### Generate Deal Analysis

```python
# Generate report comparing top result to others
best_apartment = results[0]
competitors = results[1:]
report = rag.generate_report(best_apartment, competitors)
print(report)
```

## 📁 Project Structure

```
src/RAG/
├── buliding_index.ipynb          # Main notebook for building pipeline
├── rag_system.py                 # Production-ready RAG module
├── readme.md                      # This file
├── artifacts/                     # DVC-tracked artifacts
│   ├── embeddings.npy            # NumPy array of embeddings
│   ├── metadata.json             # Apartment metadata
│   └── config.json               # Pipeline configuration
└── chroma_db/                    # ChromaDB persistent storage
    └── (ChromaDB files)
```

## 🔧 Configuration

Key parameters in `rag_system.py`:

```python
EMBEDDING_MODEL = 'sentence-transformers/multilingual-MiniLM-L12-v2'  # Supports Russian + English
BATCH_SIZE = 32                                                       # Embedding batch size
TOP_K = 5                                                             # Default search results
CHROMA_DB_PATH = 'src/RAG/chroma_db'                                 # Vector DB location
```

## 📊 Available Embedding Models

For Russian language support, recommended models:

- `sentence-transformers/multilingual-MiniLM-L12-v2` (384-dim) - **Best balance**
- `sentence-transformers/multilingual-mpnet-base-v2` (768-dim) - **Best quality**
- `sentence-transformers/LaBSE` (768-dim) - **For multiple languages**

## 💾 DVC Tracking

Track vector database and embeddings with DVC:

```bash
# Add artifacts to DVC
dvc add src/RAG/artifacts/embeddings.npy
dvc add src/RAG/artifacts/metadata.json
dvc add src/RAG/artifacts/config.json
dvc add src/RAG/chroma_db

# Commit to Git
git add src/RAG/artifacts/*.dvc src/RAG/chroma_db.dvc .gitignore
git commit -m "Add RAG system artifacts"
```

## 🤖 LLM Integration

### OpenAI API

```python
import os
os.environ['OPENAI_API_KEY'] = 'sk-...'

report = rag.generate_report(
    best_apartment,
    competitors,
    use_llm=True,
    openai_key=os.environ.get('OPENAI_API_KEY')
)
```

### Template-Based Analysis (No API)

Default behavior uses statistical comparison without LLM:
- Price deviation from market average
- Deal rating (Good/Favorable/Average/Overpriced)
- Competitor comparison

## 📝 API Reference

### RealEstateRAG Class

#### `__init__(chroma_db_path, embedding_model, collection_name)`
Initialize RAG system and load embedding model.

#### `build_index(df, text_columns, rebuild, batch_size)`
Build embeddings index from apartment DataFrame.

- **df**: DataFrame with apartment data
- **text_columns**: Columns to concatenate for embedding
- **rebuild**: Delete existing collection
- **batch_size**: Embedding batch size

#### `search(query, top_k, filters)`
Search for similar apartments.

- **query**: Text query or apartment description
- **top_k**: Number of results
- **Returns**: List of similar apartments with similarity scores

#### `generate_report(target_apartment, similar_apartments, use_llm, openai_key)`
Generate deal analysis report.

- **target_apartment**: Apartment being evaluated
- **similar_apartments**: Competitor apartments
- **use_llm**: Use LLM for analysis
- **Returns**: Formatted analysis report

## 🔍 Search Examples

```python
# By description
results = rag.search("Bright apartment with panoramic views")

# By requirements
results = rag.search("Studio 25 sqm near metro Belorusskaya up to 5 million")

# By similar apartment ID (embedding existing apartment)
apt_description = df[df['offer_id'] == '12345']['description'].iloc[0]
results = rag.search(apt_description, top_k=10)
```

## ⚙️ Advanced Usage

### Custom Text Preparation

```python
df['custom_text'] = (
    "Property: " + df['title'] +
    " | Location: " + df['address'] +
    " | Features: " + df['description']
)

rag.build_index(df, text_columns=['custom_text'])
```

### Batch Analysis

```python
# Analyze multiple apartments
for idx, row in df.head(10).iterrows():
    query_results = rag.search(row['description'], top_k=5)
    report = rag.generate_report(query_results[0], query_results[1:])
    print(f"Apartment {idx}: {report}\n")
```

## 📊 Performance Notes

- **Embedding Generation**: ~60-100 apartments/second (RTX 2080 GPU)
- **Vector Database Size**: ~93K apartments ≈ 140MB (384-dim embeddings)
- **Search Speed**: <10ms for top-k retrieval

## 🐛 Troubleshooting

**Q: ChromaDB not persisting**
- Check directory permissions at `CHROMA_DB_PATH`
- Ensure ChromaDB is using persistent client

**Q: Slow embedding generation**
- Increase `BATCH_SIZE` (if GPU memory allows)
- Use smaller embedding model (multilingual-MiniLM-L12-v2)

**Q: Poor search results**
- Improve query phrasing (more detailed is better)
- Try different embedding model
- Check that apartments have good descriptions

## 📚 Related Documents

- [Data Pipeline](../data/readme.md) - Data cleaning and validation
- [Text Modeling](../modeling/text/readme.md) - Text feature engineering
- [Parsing](../parsing/readme.md) - Web scraping pipeline

## 📄 License

Part of BYTE project - Real Estate Price Prediction Pipeline

---

**Next Steps:**
1. Run `src/RAG/buliding_index.ipynb` to build the system
2. Test search functionality with sample queries
3. Integrate API wrapper for production deployment
4. Consider UI for search and report viewing
