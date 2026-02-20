# NHANES Age Intelligence

A production-grade ML pipeline for biological age prediction using the NHANES dataset.

## 📁 Project Structure
- `data/`: Raw and processed data.
- `src/`: Core modular pipeline components.
- `api/`: FastAPI backend for model serving.
- `dashboard/`: Streamlit interactive dashboard.
- `models/`: Saved models and preprocessors.
- `config/`: YAML configurations.

## 🚀 Getting Started

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run Data Pipeline
```bash
python src/data_ingestion.py
python src/preprocessing.py
python src/model_training.py
```

### 3. Start API
```bash
uvicorn api.main:app --reload
```

### 4. Start Dashboard
```bash
streamlit run dashboard/app.py
```

## 🧠 Features
- **Production Grade**: Modular scripts, config-driven, logging, and validation.
- **MLOps**: MLflow tracking for experiments.
- **Explainability**: Integrated evaluation plots and SHAP support readiness.
- **Deployable**: Dockerized and ready for cloud deployment.
