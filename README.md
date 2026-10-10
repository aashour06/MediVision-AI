<p align="center">
  <img src="assets/MediVision - AI.png" alt="MediVision-AI Logo" width="220">
</p>

<h1 align="center">🧬 MediVision-AI</h1>

<p align="center">
  <strong>Multi-Modal AI Healthcare Intelligence Platform</strong><br>
  Predictive Health Analytics &bull; Chronic Disease Risk Assessment &bull; Clinical Computer Vision
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Vue.js-3.x-4FC08D?style=flat&logo=vuedotjs&logoColor=white" alt="Vue 3">
  <img src="https://img.shields.io/badge/Tailwind_CSS-3.x-38B2AC?style=flat&logo=tailwind-css&logoColor=white" alt="Tailwind CSS">
  <img src="https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/CDC_NHANES-2017--2020-blue?style=flat" alt="CDC NHANES">
  <img src="https://img.shields.io/badge/DEPI-Graduation_Project-EA580C?style=flat" alt="DEPI Graduation Project">
</p>

---

## 📌 Executive Summary

**MediVision-AI** is a modular, clinical-grade healthcare intelligence platform engineered to assist physicians, clinical researchers, and healthcare providers in proactive, data-driven patient risk assessment.

By combining **Predictive Health Analytics** (evaluating 17 physiological and demographic biomarkers from real-world population surveys) with a **RESTful microservice architecture** and a modern, high-readability **Clinical Dashboard**, MediVision-AI provides instant, multi-disease risk profiles before acute clinical escalation.

---

## 🏗️ System Architecture & Modules

```
                    ┌──────────────────────────────────────────────┐
                    │          CDC NHANES 2017–2020 Data           │
                    │         (12 Modular SAS .XPT Files)          │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │      01_MediVision ML Training Pipeline      │
                    │   HistGradientBoosting Multi-Disease Engine   │
                    └──────────────────────┬───────────────────────┘
                                           │ Serialized Models (.joblib)
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │             FastAPI Backend API              │
                    │        (POST /predict • 17 Biomarkers)       │
                    └──────────────┬───────────────────────────────┘
                                   │ HTTP JSON (Real-Time)
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    MediVision-AI Clinical Dashboard (Frontend)                    │
│  - Warm Medical Orange Theme        - Inline Biomarker Telemetry (17 markers)   │
│  - Live WHO BMI Categorization      - Dual Patient Presets (High-Risk/Healthy)   │
│  - 6-Disease Probability Meters     - Clinical Summary Export & Print           │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

### 1. 📊 Predictive Health Analytics (Machine Learning Module)
**Status:** ✅ Active & Operational &bull; **Location:** [`notebooks/`](notebooks/) &bull; [`models/`](models/)

The core predictive engine simultaneously analyzes individual risk across **6 major chronic conditions** derived from CDC NHANES Cycle P (2017–2020 Pre-Pandemic) survey data:

| Chronic Disease | Test ROC-AUC | Test PR-AUC | Population Positive Rate | Clinical Focus |
|:----------------|:------------:|:-----------:|:-----------------------:|:---------------|
| **Diabetes** | **0.934** | High | ~15% | Glycemic control & metabolic dysfunction |
| **Hypertension** | **0.811** | Moderate | ~37% | Cardiovascular tension & arterial health |
| **Heart Disease** | **0.810** | Moderate | ~8% | Coronary pathology, angina & myocardial infarction |
| **Stroke** | **0.784** | Low-Mod | ~5% | Cerebrovascular risk & perfusion deficit |
| **Arthritis** | **0.782** | Moderate | ~31% | Chronic joint inflammation & degeneration |
| **Asthma** | **0.607** | Low | ~16% | Chronic respiratory airway sensitivity |

> **Why ROC-AUC?** ROC-AUC is our primary evaluation metric because it is robust against severe class imbalance (e.g., Stroke at ~5% and Heart Disease at ~8%) and reliably assesses how effectively the models rank high-risk individuals above healthy baselines.

#### Core Machine Learning Design Decisions:
- **`HistGradientBoostingClassifier`**: Natively handles missing clinical laboratory values without requiring synthetic imputation.
- **Dedicated Independent Classifiers**: One tuned gradient boosted tree per disease target to match each condition's distinct class balance.
- **Class Balancing (`class_weight='balanced'`)**: Corrects for skewed sick-to-healthy prevalence.
- **Strict Anti-Overfitting Safeguards**: Early stopping with validation score monitoring, constrained max tree depth, and L2 regularization.

---

### 2. ⚙️ High-Performance Backend API Service
**Status:** ✅ Active & Operational &bull; **Location:** [`backend/`](backend/)

A lightweight, asynchronous REST microservice built with **FastAPI** and served via **Uvicorn**.

#### API Endpoints:

| Method | Path | Description | Access |
|:-------|:-----|:------------|:-------|
| `GET` | `/` | Health check & service readiness probe | Public |
| `POST` | `/predict` | Ingests 17 biomarkers, returns comprehensive risk scores | Public |
| `GET` | `/docs` | Interactive Swagger UI API documentation & testing sandbox | Public |
| `GET` | `/redoc` | OpenAPI ReDoc alternative documentation | Public |

#### Ingested Biomarker Schema (17 Clinical Parameters):

| # | Parameter Name | Data Type | Units / Range | Physiological Category | Normal Clinical Benchmark |
|:-:|:---------------|:---------:|:-------------:|:-----------------------|:--------------------------|
| 1 | `age` | `float` | 1 – 120 yrs | Demographics | Adult demographic |
| 2 | `gender` | `int` | 1 = Male, 2 = Female | Demographics | Biological sex |
| 3 | `ethnicity` | `int` | 1 – 7 (NHANES code) | Demographics | Demographic category |
| 4 | `education` | `int` | 1 – 5 scale | Demographics | Socioeconomic factor |
| 5 | `income_poverty_ratio`| `float` | 0.0 – 5.0+ | Demographics | `< 1.0` denotes poverty threshold |
| 6 | `height_cm` | `float` | 100 – 230 cm | Anthropometrics | Stature baseline |
| 7 | `weight_kg` | `float` | 30 – 250 kg | Anthropometrics | Weight baseline |
| 8 | `bmi` | `float` | 10 – 60 kg/m² | Anthropometrics | WHO standard: `18.5 – 24.9` |
| 9 | `waist_cm` | `float` | 50 – 180 cm | Anthropometrics | `< 88` (F) / `< 102` (M) cm |
| 10 | `hba1c` | `float` | % | Laboratory Panel | `< 5.7%` (Normal) |
| 11 | `fasting_glucose` | `float` | mg/dL | Laboratory Panel | `70 – 99` mg/dL (Fasting) |
| 12 | `total_cholesterol` | `float` | mg/dL | Laboratory Panel | `< 200` mg/dL |
| 13 | `hdl` | `float` | mg/dL | Laboratory Panel | `> 40` (M) / `> 50` (F) mg/dL (Good) |
| 14 | `triglycerides` | `float` | mg/dL | Laboratory Panel | `< 150` mg/dL |
| 15 | `ldl` | `float` | mg/dL | Laboratory Panel | `< 100` mg/dL (Optimal) |
| 16 | `ever_smoked` | `int` | 1 = Yes, 2 = No | Lifestyle | Non-smoker baseline |
| 17 | `sedentary_minutes` | `float` | minutes/day | Lifestyle | Typical: `180 – 360` min/day |

#### Sample Prediction Request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 62, "gender": 2, "ethnicity": 4, "education": 3, "income_poverty_ratio": 1.2,
    "height_cm": 162, "weight_kg": 92, "bmi": 35.1, "waist_cm": 108,
    "hba1c": 7.2, "fasting_glucose": 145, "total_cholesterol": 240,
    "hdl": 35, "triglycerides": 210, "ldl": 160,
    "ever_smoked": 1, "sedentary_minutes": 480
  }'
```

#### Sample Prediction Response:
```json
{
  "patient_profile": {
    "age": 62.0,
    "bmi": 35.1
  },
  "predictions": {
    "Diabetes": {
      "risk_score": 0.842,
      "risk_percentage": "84.2%",
      "prediction": "High Risk",
      "is_at_risk": true
    },
    "Hypertension": {
      "risk_score": 0.761,
      "risk_percentage": "76.1%",
      "prediction": "High Risk",
      "is_at_risk": true
    },
    "Heart Disease": {
      "risk_score": 0.589,
      "risk_percentage": "58.9%",
      "prediction": "High Risk",
      "is_at_risk": true
    },
    "Stroke": {
      "risk_score": 0.384,
      "risk_percentage": "38.4%",
      "prediction": "Low Risk",
      "is_at_risk": false
    },
    "Arthritis": {
      "risk_score": 0.628,
      "risk_percentage": "62.8%",
      "prediction": "High Risk",
      "is_at_risk": true
    },
    "Asthma": {
      "risk_score": 0.295,
      "risk_percentage": "29.5%",
      "prediction": "Low Risk",
      "is_at_risk": false
    }
  }
}
```

---

### 3. 🖥️ Modern Clinical Intelligence Dashboard
**Status:** ✅ Active & Operational &bull; **Location:** [`frontend/index.html`](frontend/index.html)

A browser-native, zero-build clinical dashboard powered by **Vue 3**, **Tailwind CSS**, and **Font Awesome 6**.

#### Key UI/UX Highlights:
- 🎨 **Medical Orange Brand Palette**: Modern gradient orange identity (`#EA580C`, `#F97316`, `#C2410C`, cream `#FAF7F2`) paired with Google Fonts (*Inter* + *Poppins*).
- 🏷️ **Integrated Branding Assets**: Displays the official `MediVision - AI.png` medical cross & leaf icon across the header, favicon, and empty-state telemetry displays.
- 📐 **High-Readability Telemetry Layout**: Generous `max-w-[1420px]` responsive canvas with `1.4rem` card padding, inline input fields, and clear clinical reference labels.
- 🧮 **Live WHO BMI Classifier**: Automatically recalculates Body Mass Index on height/weight input and applies color-coded WHO chips (*Underweight*, *Normal*, *Overweight*, *Obese*).
- ⚡ **Dual Sample Data Presets**:
  - **High-Risk**: Populates a realistic multi-morbid profile (elevated HbA1c, high BMI, dyslipidemia, smoker).
  - **Healthy**: Populates a normative baseline profile.
- 📊 **6-Disease Dynamic Risk Grid**: Live probability percentages, animated progress bars, ROC-AUC benchmarks, and clinical monitoring badges.
- 🖨️ **Print & Export Ready**: Native browser print styling isolates the diagnostic report for patient charts or PDF generation.
- 🚀 **Zero-Build Architecture**: Can be opened directly via `file:///` double-click in any browser or served through any HTTP server.

---

### 4. 👁️ Computer Vision Diagnostics (Roadmap)
**Status:** 🔬 In Development &bull; **Location:** [`models/`](models/) &bull; [`notebooks/`](notebooks/)

An upcoming extension enabling automated imaging diagnostics:
- **Modalities**: Chest Radiographs (X-Rays), Retinal Fundus Photography, Brain CT/MRI.
- **Targets**: Cardiomegaly, pulmonary consolidation, diabetic retinopathy staging.
- **Architecture**: Deep Convolutional Networks (ResNet, EfficientNet) & Vision Transformers (ViT).

---

## 🧪 Testing Presets Reference

The frontend includes dual pre-calibrated sample profiles to demonstrate the predictive pipeline immediately:

| Parameter | High-Risk Preset | Healthy Preset | Normal Reference |
|:----------|:----------------:|:--------------:|:-----------------|
| **Age** | 62 yrs | 32 yrs | Adult |
| **Gender** | Female (2) | Male (1) | — |
| **Height / Weight** | 162 cm / 92 kg | 178 cm / 72 kg | — |
| **Calculated BMI** | **35.1 kg/m²** (Obese Class II) | **22.7 kg/m²** (Normal) | 18.5 – 24.9 kg/m² |
| **Waist Circumference** | **108 cm** | 82 cm | < 88 (F) / < 102 (M) |
| **HbA1c** | **7.2%** (Diabetic range) | 5.1% | < 5.7% |
| **Fasting Glucose** | **145 mg/dL** | 88 mg/dL | 70 – 99 mg/dL |
| **Total Cholesterol** | **240 mg/dL** | 175 mg/dL | < 200 mg/dL |
| **HDL Cholesterol** | **35 mg/dL** (Low) | 58 mg/dL | > 40 (M) / > 50 (F) |
| **Triglycerides** | **210 mg/dL** (High) | 95 mg/dL | < 150 mg/dL |
| **LDL Cholesterol** | **160 mg/dL** (High) | 98 mg/dL | < 100 mg/dL |
| **Smoking History** | Yes (1) | No (2) | Non-smoker |
| **Sedentary Time** | **480 min/day** (8 hrs) | 180 min/day (3 hrs) | Minimally sedentary |

---

## 📂 Project Directory Structure

```text
MediVision-AI/
├── assets/
│   ├── MediVision - AI.png          # Official project logo & branding emblem
│   └── heart_disease.png            # 3D cardiovascular pathology medical illustration
├── backend/
│   ├── main.py                      # FastAPI REST microservice & prediction logic
│   └── requirements.txt             # API runtime dependencies (FastAPI, Uvicorn, etc.)
├── data/
│   ├── raw/                         # Raw CDC NHANES .XPT files (12 survey modules)
│   └── processed/                   # Cleaned, merged tabular clinical datasets
├── frontend/
│   └── index.html                   # Vue 3 + Tailwind CSS clinical dashboard (zero-build)
├── models/
│   ├── trained_models.joblib        # Serialized HistGradientBoosting models (6 diseases)
│   └── selected_features.joblib     # Pre-selected training feature indices & names
├── notebooks/
│   ├── 01_MediVision_Predictive_Health_Analytics.ipynb  # Complete ML pipeline notebook
│   └── NOTEBOOK_GUIDE.md            # Detailed cell-by-cell notebook explanation
├── reports/
│   └── nhanes_dataset_report.pdf    # Comprehensive exploratory dataset analysis report
├── start_backend.bat                # Windows quick launcher for the backend server
├── requirements.txt                 # Core data science & ML pipeline dependencies
├── .gitignore
└── README.md                        # Project documentation (this file)
```

---

## 🛠️ Complete Technology Stack

| Domain | Technologies & Libraries |
|:-------|:-------------------------|
| **Data Engineering & ML** | Python 3.9+ &bull; scikit-learn &bull; pandas &bull; NumPy &bull; SciPy &bull; matplotlib &bull; seaborn &bull; XGBoost |
| **Backend & Microservice** | FastAPI &bull; Uvicorn (ASGI) &bull; Pydantic &bull; joblib |
| **Frontend UI/UX** | Vue.js 3 &bull; Tailwind CSS &bull; Font Awesome 6 &bull; Google Fonts (*Inter*, *Poppins*) |
| **Data Standard** | CDC NHANES SAS Transport Format (`.xpt`) Cycle P (2017–2020) |
| **Model Serialization** | joblib (`.joblib`) binary compression |

---

## 🚀 Quickstart Guide

Run the full MediVision-AI stack locally in minutes:

### Prerequisites
- Python 3.9 or higher
- [uv](https://docs.astral.sh/uv/) (recommended for fast package installation) or standard `pip`
- Modern web browser (Chrome, Edge, Firefox, Safari)

---

### Step 1: Clone the Repository & Setup Environment

```bash
git clone https://github.com/<your-username>/MediVision-AI.git
cd MediVision-AI

# Create virtual environment with uv (or: python -m venv .venv)
uv venv

# Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (CMD):
.venv\Scripts\activate.bat
# On macOS / Linux:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
uv pip install -r backend/requirements.txt
```

---

### Step 2: Generate the Model Artifacts (First Time Only)

Ensure `models/trained_models.joblib` and `models/selected_features.joblib` are generated. Launch the notebook and run all cells:

```bash
jupyter notebook notebooks/01_MediVision_Predictive_Health_Analytics.ipynb
```

> **Automated Ingestion:** The notebook will automatically download the 12 CDC NHANES survey `.xpt` files on the first execution. Subsequent executions utilize the cached files in `data/raw/`.

---

### Step 3: Launch the Backend Microservice

**Option A — Windows Quick Launcher:**
```cmd
start_backend.bat
```

**Option B — Direct Command:**
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Verify service operation:
- **API Status:** `http://localhost:8000/` &rarr; `{"status": "healthy", ...}`
- **Interactive Documentation:** `http://localhost:8000/docs`

---

### Step 4: Open the Clinical Dashboard

The frontend is completely zero-build and ready out of the box:

1. Open `frontend/index.html` in your web browser (direct double-click).
2. Alternatively, serve via Python's built-in web server:
   ```bash
   cd frontend
   python -m http.server 3000
   ```
   and navigate to `http://localhost:3000`.
3. Click the **"High-Risk"** or **"Healthy"** preset buttons, then click **"Generate Risk Report"** to view real-time risk stratification.

---

## 📚 Supplementary Documentation

- 📖 **[Notebook Guide](notebooks/NOTEBOOK_GUIDE.md)**: In-depth breakdown of every ML step, from NHANES survey weighting to model calibration and threshold selection.
- 📄 **[Dataset Report](reports/nhanes_dataset_report.pdf)**: Detailed distribution tables and demographic summaries for the Cycle P survey cohort.

---

## 🔬 Data Provenance & Ethics

All training data originates from the **National Health and Nutrition Examination Survey (NHANES)** conducted by the National Center for Health Statistics (NCHS), Centers for Disease Control and Prevention (CDC).
- **Survey Cycle:** Pre-Pandemic 2017–March 2020 (Cycle P).
- **Public Domain:** Data is publicly accessible at [wwwn.cdc.gov/nchs/nhanes](https://wwwn.cdc.gov/nchs/nhanes/).
- **Ethical Use:** Used in compliance with NCHS data usage agreements for research and educational purposes.

---

## 🎓 Academic Attribution

This software platform was developed as a Capstone Graduation Project for the **Digital Egypt Pioneers Initiative (DEPI)** under the Ministry of Communications and Information Technology (MCIT), Egypt.
