# 🧬 MediVision-AI

**MediVision-AI** is a comprehensive, microservice-driven healthcare platform designed to assist medical professionals by combining **Predictive Health Analytics** with **Computer Vision Diagnostics**. 

By leveraging multi-modal patient data—ranging from demographic and physiological tabular data to complex medical imaging—MediVision-AI provides a holistic, AI-powered **Comprehensive Health Risk Report** for patients.

---

## 🏗️ System Architecture & Microservices

The project is designed with a scalable, modular microservices architecture. Each module operates independently but integrates to form a complete clinical diagnostic ecosystem.

### 1. 📊 Predictive Health Analytics (ML Module)
**Status:** Active | **Directory:** `notebooks/`, `data/`

This microservice focuses on predicting systemic health conditions using tabular patient data (clinical labs, demographics, physical exams, and lifestyle histories).
* **Data Source:** CDC NHANES (2017-2020 Pre-Pandemic dataset).
* **Capabilities:** Predicts the risk of **6 major conditions** simultaneously:
  1. Diabetes
  2. Hypertension
  3. Heart Disease
  4. Stroke
  5. Arthritis
  6. Asthma
* **Core Technology:** Highly efficient `HistGradientBoostingClassifier` models with strict L2 regularization and depth constraints to prevent overfitting on imbalanced medical data.

### 2. 👁️ Computer Vision Diagnostics (CV Module)
**Status:** In Development | **Directory:** `models/`, `notebooks/`

This microservice handles the ingestion and diagnostic analysis of medical imagery (e.g., Chest X-Rays for pneumonia, Retinal scans for diabetic retinopathy, or MRIs for tumor detection).
* **Capabilities:** Image classification, anomaly segmentation, and automated radiology reporting.
* **Core Technology:** Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) built with PyTorch or TensorFlow.

### 3. ⚙️ Backend API Service
**Status:** Planned | **Directory:** `backend/`

The central routing hub that serves the AI models to the frontend applications.
* **Capabilities:** RESTful endpoints for patient data ingestion, image uploading, executing model inferences, and returning JSON risk reports.
* **Core Technology:** FastAPI / Flask.

### 4. 🖥️ Frontend User Interface
**Status:** Planned | **Directory:** `frontend/`

The clinical web dashboard for healthcare providers.
* **Capabilities:** View patient profiles, upload medical scans, and visualize the generated Unified Health Risk Reports in an intuitive UI.
* **Core Technology:** React / Vue.js.

---

## 📈 ML Module Performance Metrics

The Predictive Analytics module has been fully implemented and validated on an unseen 20% hold-out test set. We prioritize the **ROC-AUC** metric to properly account for the heavy class imbalances typical in medical datasets.

| Disease Target | Test ROC-AUC Score | Generalization Check (Overfitting) |
| :--- | :--- | :--- |
| **Diabetes** | **0.949** | ✅ Excellent (Stable) |
| **Hypertension** | **0.819** | ✅ Good (Stable) |
| **Heart Disease** | **0.800** | ✅ Good (Stable) |
| **Arthritis** | **0.779** | ✅ Good (Stable) |
| **Stroke** | **0.776** | ✅ Acceptable (Rare Event) |
| **Asthma** | **0.635** | ✅ Baseline (Difficult via standard labs) |

---

## 📂 Project Structure

```text
MediVision-AI/
├── backend/          # API microservice and routing (FastAPI)
├── data/             # Raw and processed datasets (NHANES files)
├── frontend/         # Web dashboard UI for clinical use
├── models/           # Saved model weights (.pkl, .pt, .h5)
├── notebooks/        # Jupyter notebooks for EDA, ML training, and CV prototyping
└── requirements.txt  # Python dependencies
```

---

## 🚀 Getting Started

### 1. Environment Setup
Create a virtual environment and install the required dependencies:
```bash
python -m venv .venv
source .venv/Scripts/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Running the Predictive Analytics Pipeline
1. Navigate to the `notebooks/` directory.
2. Open `01_MediVision_Predictive_Health_Analytics.ipynb` in Jupyter.
3. The notebook will automatically download the required CDC NHANES datasets into the `data/raw/` directory.
4. Run the notebook sequentially to:
   * Execute the data cleaning pipeline.
   * View Exploratory Data Analysis (EDA) charts.
   * Train the 6 predictive models.
   * Generate a sample Comprehensive Patient Risk Report.
