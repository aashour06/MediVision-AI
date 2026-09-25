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
| **Arthritis** | **0.782** | moderate | ~31% |
| **Stroke** | **0.784** | low-moderate | ~5% |
| **Asthma** | **0.607** | low | ~16% |

> **ROC-AUC** is the primary metric — it is robust to class imbalance and measures the model's ability to correctly rank sick patients above healthy ones.

**Key design decisions:**
- `HistGradientBoostingClassifier` — handles missing lab values natively (no imputation).
- One model per disease — each tuned individually to its class distribution.
- `class_weight='balanced'` — corrects for the heavily skewed sick:healthy ratio.
- Per-disease anti-overfitting hyperparameters (depth, regularization, early stopping).

---

### 2. ⚙️ Backend API Service
**Status:** Active | **Directory:** `backend/`

A RESTful web service built with **FastAPI**. It loads the serialized machine learning models on startup, accepts 17 patient parameters via HTTP POST requests, and returns JSON risk reports containing exact probability percentages.

---

### 3. 🖥️ Frontend Clinical Dashboard
**Status:** Active | **Directory:** `frontend/`

A professional, zero-build clinical web UI built with **Vue 3** and **Tailwind CSS**. 
- Features a clean 17-parameter input form grouped by clinical categories.
- Displays 6 dynamic risk cards with progress bars and severity indicators (e.g., <span style="color:red">AT RISK</span> vs <span style="color:green">Low Risk</span>).
- Includes an "Auto-Fill Sample" feature for rapid testing.

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
│   ├── main.py           # FastAPI application serving the ML models
│   └── requirements.txt  # API dependencies (FastAPI, Uvicorn, Pydantic)
├── data/                 # Raw and processed CDC NHANES datasets
├── frontend/
│   └── index.html        # Professional Vue + Tailwind clinical dashboard
├── models/               # Saved model weights (.joblib files)
├── notebooks/
│   ├── 01_MediVision_Predictive_Health_Analytics.ipynb
│   └── NOTEBOOK_GUIDE.md # Detailed explanation of every notebook cell and metric
├── start_backend.bat     # Windows shortcut to start the FastAPI server
├── README.md
└── requirements.txt      # Core Machine Learning dependencies
```

---

## 🚀 Getting Started

Follow these steps to run the full full-stack application locally.

### 1. Environment Setup

Create a virtual environment and install the required dependencies for both the ML pipeline and the backend API:

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On macOS / Linux:
# source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### 2. Generate the AI Models

The backend needs the trained models to exist in the `models/` folder. Open the Jupyter Notebook, run all cells to process the CDC data, and the final cell will save the models automatically:

```bash
jupyter notebook notebooks/01_MediVision_Predictive_Health_Analytics.ipynb
```

### 3. Start the Backend API

Run the provided batch script (Windows) to start the FastAPI server:

```bash
.\start_backend.bat
```
*(Alternatively, run: `cd backend && uvicorn main:app --reload`)*

The API will now be running at `http://localhost:8000`. You can view the automatic interactive API documentation at `http://localhost:8000/docs`.

### 4. Launch the Frontend Dashboard

There are no complex Node.js build steps required. Simply open the `index.html` file in your preferred web browser:

1. Navigate to the `frontend/` folder in your file explorer.
2. Double-click **`index.html`**.
3. Click **"Auto-Fill Sample"** on the dashboard and hit **"Generate Risk Report"** to see the AI in action!

---

## 📚 Documentation

For a comprehensive explanation of every machine learning design decision, evaluation metric, and data cleaning step, please read the Notebook Guide:
→ [`notebooks/NOTEBOOK_GUIDE.md`](notebooks/NOTEBOOK_GUIDE.md)

---

## 🔬 Data Provenance

All tabular data comes from the **CDC NHANES (National Health and Nutrition Examination Survey)**, Cycle P (August 2017 – March 2020). Data is publicly available at [wwwn.cdc.gov/nchs/nhanes](https://wwwn.cdc.gov/nchs/nhanes/). Files are downloaded automatically by the scripts; no manual download is required.
