# MediVision AI — Notebook Guide

> **File:** `notebooks/01_MediVision_Predictive_Health_Analytics.ipynb`  
> **Dataset:** NHANES Cycle P (2017–2020) · CDC, United States  
> **Goal:** Predict 6 chronic disease risks from real patient data using a clean, reproducible ML pipeline.

---

## Table of Contents

1. [What This Notebook Does](#1-what-this-notebook-does)
2. [Data Source](#2-data-source)
3. [Section 1 — Setup](#3-section-1--setup)
4. [Section 2 — Download Data](#4-section-2--download-data)
5. [Section 3 — Load & Clean Data](#5-section-3--load--clean-data)
6. [Section 4 — Exploratory Data Analysis](#6-section-4--exploratory-data-analysis)
7. [Section 5 — Model Training](#7-section-5--model-training)
8. [Section 5.2 — Model Evaluation](#8-section-52--model-evaluation)
9. [Section 6 — Patient Risk Report](#9-section-6--patient-risk-report)
10. [Overfitting: Diagnosis & Fix](#10-overfitting-diagnosis--fix)
11. [Evaluation Metrics Explained](#11-evaluation-metrics-explained)
12. [Glossary](#12-glossary)

---

## 1. What This Notebook Does

This notebook builds an end-to-end machine learning pipeline that:

- Downloads real US health survey data (NHANES) automatically
- Cleans and merges 12 data files into one analysis-ready table
- Explores the data visually with 12 chart sections
- Trains **one model per disease** — 6 models total
- Evaluates every model with ROC curves, PR curves, and confusion matrices
- Generates a per-patient risk report across all 6 diseases simultaneously

**Diseases predicted:**

| # | Disease | Positive Rate |
|---|---------|:------------:|
| 1 | Diabetes | ~15% |
| 2 | Hypertension | ~37% |
| 3 | Heart Disease | ~8% |
| 4 | Asthma | ~16% |
| 5 | Stroke | ~5% |
| 6 | Arthritis | ~31% |

---

## 2. Data Source

**NHANES** = National Health and Nutrition Examination Survey  
Published by the **CDC (Centers for Disease Control and Prevention)**, USA.  
Cycle P covers **August 2017 – March 2020** (pre-pandemic).

Each respondent has a unique ID (`SEQN`) linking records across all survey modules.

### Files Used

| File | Topic | Key columns |
|------|-------|-------------|
| `P_DEMO.xpt` | Demographics | Age, gender, ethnicity, education, income |
| `P_BMX.xpt` | Body measurements | Height, weight, BMI, waist circumference |
| `P_BPQ.xpt` | Blood pressure questions | Hypertension diagnosis |
| `P_DIQ.xpt` | Diabetes questions | Diabetes diagnosis |
| `P_GHB.xpt` | Glycohaemoglobin | HbA1c (3-month blood sugar average) |
| `P_GLU.xpt` | Plasma glucose | Fasting blood glucose |
| `P_HDL.xpt` | HDL cholesterol | Good cholesterol |
| `P_MCQ.xpt` | Medical conditions | Heart disease, asthma, stroke, arthritis |
| `P_PAQ.xpt` | Physical activity | Sedentary minutes per day |
| `P_SMQ.xpt` | Smoking | Ever smoked |
| `P_TCHOL.xpt` | Total cholesterol | Total blood cholesterol |
| `P_TRIGLY.xpt` | Triglycerides | Blood fat + calculated LDL |

---

## 3. Section 1 — Setup

**What it does:** Imports all required Python libraries in one place.

**Libraries used:**

| Library | Purpose |
|---------|---------|
| `numpy`, `pandas` | Data manipulation |
| `matplotlib`, `seaborn` | Charts and visualisations |
| `scipy.stats` | Statistical tests (Mann-Whitney U) |
| `statsmodels` | Variance Inflation Factor (VIF) |
| `sklearn` | Machine learning models and evaluation |

---

## 4. Section 2 — Download Data

**What it does:** Downloads each NHANES `.xpt` file from CDC servers if it doesn't already exist locally.

Files are stored in `data/raw/`. The download is skipped if the file is already present — so running the notebook twice won't re-download anything.

---

## 5. Section 3 — Load & Clean Data

Five steps in one cell:

### Step 1 & 2 — Merge All Files

All 12 files are joined into one table using a **left join on `SEQN`**:

```
main table = Demographics (P_DEMO)
for each other file:
    main table = main table LEFT JOIN file ON SEQN
```

A left join keeps every respondent from Demographics even if they didn't complete other modules. Missing modules appear as `NaN`.

### Step 3 — Rename Columns

CDC variable codes (e.g. `BMXBMI`, `LBXGH`) are renamed to plain English:

| Original | Renamed | Description |
|----------|---------|-------------|
| `RIDAGEYR` | `age` | Age in years |
| `RIAGENDR` | `gender` | 1=Male, 2=Female |
| `BMXBMI` | `bmi` | Body Mass Index |
| `BMXWAIST` | `waist_cm` | Waist circumference |
| `LBXGH` | `hba1c` | 3-month blood sugar average |
| `LBXGLU` | `fasting_glucose` | Blood sugar after fasting |
| `LBDHDD` | `hdl` | Good cholesterol |
| `LBXTC` | `total_cholesterol` | Total cholesterol |
| `LBXTR` | `triglycerides` | Blood fat |
| `LBDLDL` | `ldl` | Bad cholesterol |
| `PAD680` | `sedentary_minutes` | Minutes sitting per day |

### Step 4 — Adults Only

Only respondents aged **18+** are kept. Chronic diseases like heart disease and stroke are almost exclusively adult conditions — including children would add noise.

### Step 5 — Disease Targets

Each disease gets a column with values `1` (has disease), `0` (does not), or `NaN` (refused / unknown).

| Target | Source | Logic |
|--------|--------|-------|
| `target_diabetes` | `DIQ010` | 1 if answered "Yes" |
| `target_hypertension` | `BPQ020` | 1 if answered "Yes" |
| `target_heart_disease` | `MCQ160C/D/E` | 1 if any of the three sub-conditions |
| `target_asthma` | `MCQ010` | 1 if answered "Yes" |
| `target_stroke` | `MCQ160F` | 1 if answered "Yes" |
| `target_arthritis` | `MCQ160A` | 1 if answered "Yes" |

> Heart disease combines **Coronary Heart Disease**, **Angina**, and **Heart Attack** into one positive label.

---

## 6. Section 4 — Exploratory Data Analysis

12 sub-sections, each with a dedicated code cell.

### 4.1 EDA Setup
Defines shared colours (`BLUE`, `ORANGE`) and groups columns into `NUM_COLS` (continuous) and `CAT_COLS` (categorical).

### 4.2 Dataset Overview
Prints rows, columns, memory, and a sample of 5 rows.

### 4.3 Missing Values
Ranks every column by its missing percentage. Blood-test columns typically have 40–60% missing because they were only collected from a fasting sub-sample of respondents.

> **Why we don't fill in missing values:** `HistGradientBoostingClassifier` learns the optimal split direction for missing values automatically — so imputation would actually add noise.

### 4.4 Basic Statistics
For every numeric feature: count, mean, median, std, min, max, **skewness**, **kurtosis**.

- `|skewness| > 1` → distribution is heavily lopsided
- `kurtosis > 3` → heavy tails, more extreme outliers

### 4.5 Disease Distribution
Bar charts showing positive vs. negative counts for each disease, with imbalance ratios annotated.

> A 10:1 ratio means 10 healthy patients for every 1 sick patient. The model handles this with `class_weight='balanced'`, which automatically upweights the minority class.

### 4.6 Feature Distributions
Histograms for every numeric feature with mean (red) and median (orange) overlaid. Skewness is annotated in orange when `|skew| > 1`.

### 4.7 Categorical Counts
Horizontal bar charts for gender, ethnicity, education, marital status, and smoking status.

> **Known bug fixed here:** The original index contained float NaN values that matplotlib couldn't render as category labels. The fix converts the index explicitly: `str(int(v)) if str(v) != "nan" else "Unknown"`.

### 4.8 Biomarkers by Disease Status
For each disease, boxplots compare every biomarker between negative and positive groups.

Each subplot is annotated with a **Mann-Whitney U test** p-value:

| Stars | p-value | Meaning |
|-------|---------|---------|
| `***` | < 0.001 | Very strong difference |
| `**` | < 0.01 | Strong difference |
| `*` | < 0.05 | Significant |
| `ns` | ≥ 0.05 | No significant difference |

Mann-Whitney U is used instead of a t-test because clinical biomarker distributions are rarely Gaussian.

### 4.9 Correlation Heatmap
Lower-triangle Pearson correlation matrix.

- **Red** = strong positive correlation (both go up together)
- **Blue** = strong negative correlation (one goes up, the other goes down)
- **White** = no relationship

### 4.10 VIF (Multicollinearity)
**Variance Inflation Factor** — measures if a feature is redundant because it's highly correlated with others.

| VIF | Interpretation |
|-----|---------------|
| 1 | No redundancy |
| 1–5 | Acceptable |
| 5–10 | High — consider removing |
| > 10 | Severe — likely redundant |

Expected high-VIF pairs: `bmi` ↔ `weight_kg`, `hba1c` ↔ `fasting_glucose`, `total_cholesterol` ↔ `ldl`.

### 4.11 Outliers
IQR method: values outside `Q1 − 1.5×IQR` or `Q3 + 1.5×IQR` are flagged.

> Medical outliers represent real extreme cases (e.g. HbA1c > 14% in uncontrolled diabetes). They are **preserved**, not removed, because they carry the strongest disease signal.

### 4.12 Co-Morbidity Matrix
**Phi coefficient** between every pair of disease targets — measures how often two diseases appear in the same patient.

| φ | Interpretation |
|---|---------------|
| 0.0–0.1 | Negligible |
| 0.1–0.3 | Weak |
| 0.3–0.5 | Moderate |
| > 0.5 | Strong — shared underlying mechanism |

### 4.13 EDA Summary
Prints: imbalance ratios, top correlations, and columns with >30% missing data.

---

## 7. Section 5 — Model Training

### Algorithm: HistGradientBoostingClassifier

| Feature | Why it matters here |
|---------|-------------------|
| Handles `NaN` natively | 40–60% of lab values are missing — no imputation needed |
| Histogram binning | Fast training on 9,000–10,000 patients |
| Many regularisation controls | Easy to tune per-disease to prevent overfitting |
| `class_weight='balanced'` | Automatically corrects for imbalanced disease rates |

### Train / Test Split

- **80% training / 20% test**, `random_state=42` for reproducibility
- **Stratified** on the disease label — ensures the same disease rate in both splits
- Performed separately per disease after removing unknown labels

### Per-Disease Hyperparameters

After an initial run revealed overfitting in 4 diseases, each disease got individual settings:

| Setting | Diabetes / Hypertension | Heart Disease | Asthma | Stroke | Arthritis |
|---------|:-:|:-:|:-:|:-:|:-:|
| `max_depth` | 5 | 3 | **2** | 3 | 4 |
| `min_samples_leaf` | 20 | 40 | **60** | 50 | 30 |
| `l2_regularization` | 1.0 | 5.0 | **10.0** | 8.0 | 3.0 |
| `learning_rate` | 0.05 | 0.03 | **0.02** | 0.02 | 0.04 |
| `early_stopping` | No | Yes | Yes | Yes | Yes |

---

## 8. Section 5.2 — Model Evaluation

Three charts are produced **per disease**:

### ROC Curve
Plots True Positive Rate (Recall) vs. False Positive Rate at every possible decision threshold.
- **Area Under the Curve (AUC)** summarises the whole curve in one number
- 0.5 = random guessing; 1.0 = perfect

### Precision-Recall Curve
More informative than ROC when the positive class is rare (e.g. Stroke at 5%).
- **Average Precision (AP)** summarises the curve
- A flat, high curve means the model finds sick patients reliably

### Confusion Matrix

|  | Predicted Negative | Predicted Positive |
|--|:--:|:--:|
| **Actually Negative** | True Negative (TN) | False Positive (FP) |
| **Actually Positive** | False Negative (FN) | True Positive (TP) |

A **False Negative** (missed sick patient) is clinically worse than a **False Positive** (unnecessary follow-up). Recall therefore matters more than Precision in this use case.

### Final Results

| Disease | ROC-AUC | PR-AUC | Recall | Notes |
|---------|:-------:|:------:|:------:|-------|
| Diabetes | ~0.934 | high | high | Strong signal from HbA1c + glucose |
| Hypertension | ~0.811 | moderate | moderate | Common — many features contribute |
| Heart Disease | ~0.810 | moderate | moderate | Tightened to remove overfit |
| Asthma | ~0.607 | low | moderate | Weak signal — no allergen/genetics data |
| Stroke | ~0.784 | low | moderate | Rare event — very few positive cases |
| Arthritis | ~0.782 | moderate | moderate | Inflammatory markers partially captured |

---

## 9. Section 6 — Patient Risk Report

The `patient_report()` function runs all 6 trained models on a single patient's measurements and prints a unified risk profile.

**Example output:**
```
[ PATIENT 1 ]
Patient: Age 55, BMI 23.6
---------------------------------------------
  Diabetes        Low Risk    1.5%  [                    ]
  Hypertension    Low Risk   17.4%  [###                 ]
  Heart Disease   Low Risk   15.8%  [###                 ]
  Asthma          Low Risk   44.1%  [########            ]
  Stroke          Low Risk   31.3%  [######              ]
  Arthritis       AT RISK    51.1%  [##########          ]
---------------------------------------------
```

The progress bar `[###]` visually encodes the risk score on a 0–20 character scale.

---

## 10. Overfitting: Diagnosis & Fix

### What is overfitting?
The model memorises the training data instead of learning general rules. It scores high on training patients but performs poorly on new patients it has never seen.

### How we measured it
```
Overfit Delta = Train AUC − Test AUC
```

| Delta | Status |
|-------|--------|
| < 0.08 | Good generalisation |
| 0.08 – 0.12 | Mild overfit |
| 0.12 – 0.18 | Moderate overfit |
| > 0.18 | Severe overfit |

### Before vs. After

| Disease | Initial Delta | Fixed Delta | Root Cause |
|---------|:-----------:|:-----------:|-----------|
| Diabetes | +0.042 | +0.042 | No overfit — unchanged |
| Hypertension | +0.074 | +0.074 | No overfit — unchanged |
| Heart Disease | **+0.159** | **+0.070** | Model complexity too high |
| Asthma | **+0.301** | **+0.058** | Very weak feature signal |
| Stroke | **+0.201** | **+0.044** | Rare positive class |
| Arthritis | **+0.105** | **+0.061** | Musculoskeletal signal weak |

### Levers used to fix it

| Lever | Effect |
|-------|--------|
| `max_depth ↓` | Shallower trees → less memorisation |
| `min_samples_leaf ↑` | Each split needs more data support |
| `l2_regularization ↑` | Penalises large leaf weights |
| `learning_rate ↓` | Slower learning → less noise fitting |
| `early_stopping=True` | Stops training automatically when validation AUC plateaus |

---

## 11. Evaluation Metrics Explained

| Metric | Formula | When to use |
|--------|---------|-------------|
| **Accuracy** | (TP+TN) / Total | Only when classes are balanced |
| **Precision** | TP / (TP+FP) | When false alarms are costly |
| **Recall** | TP / (TP+FN) | When missing sick patients is costly |
| **F1** | 2 × P×R / (P+R) | Balance between Precision and Recall |
| **ROC-AUC** | Area under ROC | Overall ranking ability, robust to imbalance |
| **PR-AUC** | Area under PR | Best for rare positive class |

> In a clinical context, **Recall** is usually the most important metric — missing a sick patient (False Negative) is more harmful than a false alarm (False Positive).

---

## 12. Glossary

| Term | Definition |
|------|-----------|
| **AUC** | Area Under the Curve — summarises ROC or PR curve in one number |
| **BMI** | Body Mass Index = weight(kg) / height(m)² |
| **Class imbalance** | One class (sick) is much rarer than the other (healthy) |
| **Confusion Matrix** | Table of TP, TN, FP, FN counts |
| **Early Stopping** | Halts training when validation performance stops improving |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **False Negative** | Sick patient predicted as healthy — most dangerous error |
| **False Positive** | Healthy patient predicted as sick — leads to unnecessary tests |
| **HbA1c** | Glycohaemoglobin — reflects average blood sugar over 3 months |
| **HDL** | High-Density Lipoprotein — "good" cholesterol |
| **IQR** | Interquartile Range = Q3 − Q1 |
| **kurtosis** | Heaviness of distribution tails |
| **LDL** | Low-Density Lipoprotein — "bad" cholesterol |
| **Mann-Whitney U** | Non-parametric test for difference between two independent groups |
| **NHANES** | National Health and Nutrition Examination Survey (CDC) |
| **Overfit Delta** | Train AUC − Test AUC; measures how much a model memorised training data |
| **Phi coefficient (φ)** | Correlation between two binary (yes/no) variables |
| **PR Curve** | Precision-Recall curve — useful when positive class is rare |
| **Recall** | Proportion of actual positive cases the model correctly identifies |
| **ROC Curve** | Receiver Operating Characteristic — Recall vs. False Alarm Rate |
| **SEQN** | Unique respondent ID in NHANES, used to join all data modules |
| **Skewness** | Asymmetry of a distribution (0 = symmetric) |
| **Stratified split** | Train/test split that preserves the original class ratio |
| **VIF** | Variance Inflation Factor — measures redundancy between features |
| **XPT** | SAS Transport format used by CDC for NHANES data distribution |
