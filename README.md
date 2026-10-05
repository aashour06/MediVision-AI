# 🧬 MediVision-AI

**MediVision-AI** is a modular, microservice-driven healthcare platform that combines **Predictive Health Analytics** with **Computer Vision Diagnostics** to assist medical professionals with AI-powered patient risk assessment.

By leveraging multi-modal patient data—ranging from demographic and physiological tabular data to complex medical imaging—MediVision-AI provides a holistic, AI-powered **Comprehensive Health Risk Report**.

---

## 🏗️ System Architecture & Modules

### 1. 📊 Predictive Health Analytics (ML Module)
**Status:** Active | **Directory:** `notebooks/`, `data/`

Predicts the risk of **6 chronic diseases simultaneously** from real US population health data (CDC NHANES 2017–2020).

| Disease | Test ROC-AUC | Test PR-AUC | Positive Rate |
|---------|:-----------:|:-----------:|:------------:|
| **Diabetes** | **0.934** | high | ~15% |
| **Hypertension** | **0.811** | moderate | ~37% |
| **Heart Disease** | **0.810** | moderate | ~8% |
| **Stroke** | **0.784** | low-moderate | ~5% |
| **Arthritis** | **0.782** | moderate | ~31% |
| **Asthma** | **0.607** | low | ~16% |

> **ROC-AUC** is the primary metric — it is robust to class imbalance and measures the model's ability to correctly rank sick patients above healthy ones.

**Key design decisions:**
- `HistGradientBoostingClassifier` — handles missing lab values natively (no imputation needed).
- One model per disease — each tuned individually to its class distribution.
- `class_weight='balanced'` — corrects for the heavily skewed sick:healthy ratio.
- Per-disease anti-overfitting hyperparameters (depth, regularization, early stopping).

---

### 2. ⚙️ Backend API Service
**Status:** Active | **Directory:** `backend/`

A RESTful web service built with **FastAPI**. It loads the serialized machine learning models on startup, accepts 17 patient parameters via HTTP POST, and returns JSON risk reports containing exact probability percentages.

**Endpoints:**

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Health check — confirms the API is running |
| `POST` | `/predict` | Accepts patient data, returns 6-disease risk profile |
| `GET` | `/docs` | Auto-generated interactive Swagger UI documentation |

**Patient Input Schema (17 parameters):**

| # | Parameter | Type | Category |
|---|-----------|------|----------|
| 1 | `age` | float | Demographics |
| 2 | `gender` | int (1=Male, 2=Female) | Demographics |
| 3 | `ethnicity` | int (1–7) | Demographics |
| 4 | `education` | int (1–5) | Demographics |
| 5 | `income_poverty_ratio` | float | Demographics |
| 6 | `height_cm` | float | Body Measurements |
| 7 | `weight_kg` | float | Body Measurements |
| 8 | `bmi` | float | Body Measurements |
| 9 | `waist_cm` | float | Body Measurements |
| 10 | `hba1c` | float (%) | Lab Results |
| 11 | `fasting_glucose` | float (mg/dL) | Lab Results |
| 12 | `total_cholesterol` | float (mg/dL) | Lab Results |
| 13 | `hdl` | float (mg/dL) | Lab Results |
| 14 | `triglycerides` | float (mg/dL) | Lab Results |
| 15 | `ldl` | float (mg/dL) | Lab Results |
| 16 | `ever_smoked` | int (1=Yes, 2=No) | Lifestyle |
| 17 | `sedentary_minutes` | float | Lifestyle |

**Example Response:**
```json
{
  "patient_profile": { "age": 55, "bmi": 27.8 },
  "predictions": {
    "Diabetes": {
      "risk_score": 0.12,
      "risk_percentage": "12.0%",
      "prediction": "Low Risk",
      "is_at_risk": false
    }
  }
}
```

---

### 3. 🖥️ Frontend Clinical Dashboard
**Status:** Active | **Directory:** `frontend/`

A professional, zero-build clinical web UI built with **Vue 3**, **Tailwind CSS**, and **Font Awesome 6** (all via CDN — no Node.js required).

**Features:**
- 🏥 Clean 17-parameter input form grouped by clinical categories (Demographics, Measurements, Lab Results, Lifestyle).
- 🧮 **Auto-BMI calculation** — BMI is computed automatically when height/weight are entered.
- 📋 **Auto-Fill Sample** button for rapid testing with a realistic high-risk patient profile.
- 📊 6 dynamic risk cards with animated progress bars and color-coded severity indicators (<span style="color:red">AT RISK</span> vs <span style="color:green">Low Risk</span>).
- 📱 Responsive layout — adapts to desktop and mobile screens with automatic scroll-to-results on mobile.
- 🔄 Loading states with spinner animations and error handling for backend connectivity.

---

### 4. 👁️ Computer Vision Diagnostics
**Status:** In Development | **Directory:** `models/`, `notebooks/`

Ingestion and diagnostic analysis of medical imagery (chest X-rays, retinal scans, MRIs).
- **Planned capabilities:** Image classification, anomaly segmentation, automated radiology reporting.
- **Technology:** CNNs / Vision Transformers (PyTorch or TensorFlow).

---

## 📂 Project Structure

```text
MediVision-AI/
├── backend/
│   ├── main.py              # FastAPI application serving the ML models
│   └── requirements.txt     # API dependencies (FastAPI, Uvicorn, scikit-learn, etc.)
├── data/
│   ├── raw/                 # Raw CDC NHANES .xpt files (12 survey modules)
│   └── processed/           # Cleaned, merged datasets
├── frontend/
│   └── index.html           # Vue 3 + Tailwind CSS clinical dashboard (zero-build)
├── models/
│   ├── trained_models.joblib    # Serialized HistGradientBoosting models (6 diseases)
│   └── selected_features.joblib # Feature list used during training
├── notebooks/
│   ├── 01_MediVision_Predictive_Health_Analytics.ipynb  # Full ML pipeline
│   └── NOTEBOOK_GUIDE.md    # Detailed explanation of every notebook cell and metric
├── reports/
│   └── nhanes_dataset_report.pdf  # Dataset analysis report
├── start_backend.bat        # Windows shortcut to start the FastAPI server
├── requirements.txt         # Core ML dependencies (scikit-learn, pandas, matplotlib, etc.)
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML / Data Science** | Python · scikit-learn · pandas · NumPy · matplotlib · seaborn · SciPy · XGBoost |
| **Backend API** | FastAPI · Uvicorn · Pydantic · joblib |
| **Frontend** | Vue 3 (CDN) · Tailwind CSS (CDN) · Font Awesome 6 |
| **Data Format** | CDC NHANES `.xpt` (SAS Transport) |
| **Model Serialization** | joblib (`.joblib`) |

---

## 🚀 Getting Started

Follow these steps to run the full-stack application locally.

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) — a fast Python package manager
- A modern web browser (Chrome, Firefox, Edge)

### 1. Clone & Setup Environment

```bash
git clone https://github.com/<your-username>/MediVision-AI.git
cd MediVision-AI

uv venv

# On Windows:
.venv\Scripts\activate
# On macOS / Linux:
# source .venv/bin/activate

# Install all dependencies
uv pip install -r requirements.txt
uv pip install -r backend/requirements.txt
```

### 2. Generate the AI Models

The backend needs the trained models to exist in the `models/` folder. Open the Jupyter Notebook and run all cells — the final cell saves the models automatically:

```bash
jupyter notebook notebooks/01_MediVision_Predictive_Health_Analytics.ipynb
```

> **Note:** The notebook automatically downloads the 12 NHANES data files from the CDC on first run. Subsequent runs skip the download.

### 3. Start the Backend API

**Option A — Windows batch script:**
```bash
.\start_backend.bat
```

**Option B — Manual:**
```bash
cd backend
uvicorn main:app --reload
```

The API will be running at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive Swagger documentation.

### 4. Launch the Frontend Dashboard

No build step required. Simply open the file in your browser:

1. Navigate to the `frontend/` folder in your file explorer.
2. Double-click **`index.html`**.
3. Click **"Auto-Fill Sample"** on the dashboard and hit **"Generate Risk Report"** to see the AI in action!

> **Tip:** The frontend connects to `http://localhost:8000/predict` — make sure the backend is running first.

---

## 📚 Documentation

For a comprehensive explanation of every machine learning design decision, evaluation metric, and data cleaning step, please read the Notebook Guide:
→ [`notebooks/NOTEBOOK_GUIDE.md`](notebooks/NOTEBOOK_GUIDE.md)

---

## 🔬 Data Provenance

All tabular data comes from the **CDC NHANES (National Health and Nutrition Examination Survey)**, Cycle P (August 2017 – March 2020). Data is publicly available at [wwwn.cdc.gov/nchs/nhanes](https://wwwn.cdc.gov/nchs/nhanes/).

**12 survey modules** are used covering demographics, body measurements, blood pressure, diabetes, glycohaemoglobin, plasma glucose, HDL cholesterol, medical conditions, physical activity, smoking, total cholesterol, and triglycerides. Files are downloaded automatically by the notebook; no manual download is required.

---

## 📄 License

This project was developed as a graduation project for the **DEPI (Digital Egypt Pioneers Initiative)** program.
