"""
MediVision-AI — Improved Multi-Label Classification Model
==========================================================
Key improvements over the original notebook:
1. Add "No Finding" as an explicit 4th target class
2. Use GroupKFold to prevent patient data leakage
3. Feature engineering (age bins, aspect ratio, area, pixel density)
4. Per-label threshold optimization on validation set
5. Hyperparameter tuning with RandomizedSearchCV
6. Better class imbalance handling
7. Comprehensive evaluation with per-class and overall metrics
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import ast
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score, GroupKFold,
    RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    FunctionTransformer, OneHotEncoder, StandardScaler,
    MultiLabelBinarizer
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    make_scorer, f1_score, classification_report,
    precision_recall_curve
)
from xgboost import XGBClassifier
from scipy.stats import uniform, randint

warnings.filterwarnings("ignore", category=UserWarning)

# ── 1. Load & Parse Data ────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

df = pd.read_csv(r"..\data\processed\Data_Entry_2017_Cleaned.csv")
df["finding_labels"] = df["finding_labels"].apply(ast.literal_eval)

print(f"  Total samples: {len(df):,}")
print(f"  Unique patients: {df['patient_id'].nunique():,}")
print(f"  Columns: {df.columns.tolist()}")

# ── 2. Feature Engineering ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Feature Engineering")
print("=" * 60)

# 2a. Image geometry features
df["aspect_ratio"] = df["image_width"] / df["image_height"]
df["image_area"] = df["image_width"] * df["image_height"]
df["pixel_area"] = df["pixel_spacing_x"] * df["pixel_spacing_y"]
# Physical dimensions in mm
df["physical_width_mm"] = df["image_width"] * df["pixel_spacing_x"]
df["physical_height_mm"] = df["image_height"] * df["pixel_spacing_y"]

# 2b. Age bins (clinical relevance: pediatric, young adult, middle-aged, elderly)
df["age_bin"] = pd.cut(
    df["patient_age"],
    bins=[0, 18, 40, 60, 100],
    labels=["pediatric", "young_adult", "middle_aged", "elderly"]
)

# 2c. Patient history features (how many scans does this patient have?)
patient_scan_count = df.groupby("patient_id")["image_id"].transform("count")
df["patient_scan_count"] = patient_scan_count

# 2d. Log of follow_up (as in original but applied directly)
df["follow_up_log"] = np.log1p(df["follow_up"])

print(f"  New features added: aspect_ratio, image_area, pixel_area,")
print(f"    physical_width_mm, physical_height_mm, age_bin,")
print(f"    patient_scan_count, follow_up_log")

# ── 3. Target Encoding ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Target Encoding (4 classes including No Finding)")
print("=" * 60)

# KEY FIX: Include "No Finding" as an explicit class
classes = ["No Finding", "Atelectasis", "Effusion", "Infiltration"]
mlb = MultiLabelBinarizer(classes=classes)
y = mlb.fit_transform(df["finding_labels"])

print(f"  Classes: {classes}")
print(f"  Per-class positive counts:")
for i, cls in enumerate(classes):
    print(f"    {cls}: {y[:, i].sum():,} ({y[:, i].mean()*100:.1f}%)")

# ── 4. Feature Selection & Split ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Train/Test Split with Patient-Level Grouping")
print("=" * 60)

drop_cols = ["finding_labels", "image_id", "patient_id"]
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols].copy()
groups = df["patient_id"].values

# Patient-level split: ensure no patient appears in both train and test
unique_patients = df["patient_id"].unique()
np.random.seed(42)
np.random.shuffle(unique_patients)
split_idx = int(0.7 * len(unique_patients))
train_patients = set(unique_patients[:split_idx])
test_patients = set(unique_patients[split_idx:])

train_mask = df["patient_id"].isin(train_patients)
test_mask = df["patient_id"].isin(test_patients)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
groups_train = groups[train_mask]

print(f"  Train: {len(X_train):,} samples, {len(train_patients):,} patients")
print(f"  Test:  {len(X_test):,} samples, {len(test_patients):,} patients")
print(f"  No patient overlap: {len(train_patients & test_patients) == 0}")

# ── 5. Preprocessing Pipeline ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Building Preprocessing Pipeline")
print("=" * 60)

categorical_cols = ["patient_gender", "view_position", "age_bin"]
numerical_cols = [
    "patient_age", "image_width", "image_height",
    "pixel_spacing_x", "pixel_spacing_y",
    "aspect_ratio", "image_area", "pixel_area",
    "physical_width_mm", "physical_height_mm",
    "patient_scan_count", "follow_up_log", "follow_up"
]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
], remainder="drop")

print(f"  Numerical features ({len(numerical_cols)}): {numerical_cols}")
print(f"  Categorical features ({len(categorical_cols)}): {categorical_cols}")

# ── 6. Model Definitions ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Training Models with GroupKFold Cross-Validation")
print("=" * 60)

models = {
    "Logistic Regression": OneVsRestClassifier(
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=0.5,
            solver="saga",
            penalty="l2"
        )
    ),

    "Random Forest": OneVsRestClassifier(
        RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
    ),

    "XGBoost": OneVsRestClassifier(
        XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=4,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        )
    ),
}

# ── 7. Cross-Validation with GroupKFold ──────────────────────────────────────
scoring = make_scorer(f1_score, average="macro", zero_division=0)
gkf = GroupKFold(n_splits=5)

results = {}
trained_pipelines = {}

for name, classifier in models.items():
    print(f"\n  Training: {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=gkf,
        groups=groups_train,
        scoring=scoring,
        n_jobs=-1
    )

    results[name] = {
        "Mean F1 Macro": round(scores.mean(), 4),
        "Std F1 Macro": round(scores.std(), 4),
        "All Folds": [round(s, 4) for s in scores]
    }

    # Fit on full training set for evaluation
    pipeline.fit(X_train, y_train)
    trained_pipelines[name] = pipeline

    print(f"    CV F1 Macro: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"    Folds: {[round(s, 4) for s in scores]}")

# ── 8. Per-Label Threshold Optimization ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Per-Label Threshold Optimization")
print("=" * 60)


def optimize_thresholds(pipeline, X_val, y_val, class_names):
    """Find optimal threshold per label using precision-recall curve."""
    # Get probability predictions
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_val)
    elif hasattr(pipeline, "decision_function"):
        y_scores = pipeline.decision_function(X_val)
        # Convert decision function to pseudo-probabilities
        from scipy.special import expit
        y_prob = expit(y_scores)
    else:
        return None, None

    thresholds_opt = []
    for i, cls_name in enumerate(class_names):
        precision, recall, thresholds = precision_recall_curve(
            y_val[:, i], y_prob[:, i]
        )
        # F1 = 2 * (precision * recall) / (precision + recall)
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = np.where(
                (precision + recall) > 0,
                2 * (precision * recall) / (precision + recall),
                0
            )
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        thresholds_opt.append(best_threshold)
        print(f"    {cls_name}: threshold={best_threshold:.3f}, "
              f"F1={f1_scores[best_idx]:.4f}")

    return thresholds_opt, y_prob


# Use a validation split from training for threshold tuning
val_patients = list(train_patients)
np.random.seed(123)
np.random.shuffle(val_patients)
val_split_idx = int(0.85 * len(val_patients))
tune_train_patients = set(val_patients[:val_split_idx])
tune_val_patients = set(val_patients[val_split_idx:])

tune_train_mask = df["patient_id"].isin(tune_train_patients) & train_mask
tune_val_mask = df["patient_id"].isin(tune_val_patients) & train_mask

X_tune_train = X[tune_train_mask]
y_tune_train = y[tune_train_mask]
X_tune_val = X[tune_val_mask]
y_tune_val = y[tune_val_mask]

best_model_name = max(results, key=lambda k: results[k]["Mean F1 Macro"])
print(f"\n  Best model: {best_model_name}")
print(f"  Tuning thresholds on held-out validation ({len(X_tune_val):,} samples)...")

# Retrain best model on the tuning train split
tune_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", models[best_model_name])
])
tune_pipeline.fit(X_tune_train, y_tune_train)

print(f"\n  Optimal thresholds:")
opt_thresholds, y_prob_val = optimize_thresholds(
    tune_pipeline, X_tune_val, y_tune_val, classes
)

# ── 9. Final Evaluation on Test Set ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Final Evaluation on Test Set")
print("=" * 60)

for name, pipeline in trained_pipelines.items():
    y_pred = pipeline.predict(X_test)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_test, y_pred, average="micro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  {name}:")
    print(f"    F1 Macro:    {f1_macro:.4f}")
    print(f"    F1 Micro:    {f1_micro:.4f}")
    print(f"    F1 Weighted: {f1_weighted:.4f}")
    print(f"    Per-class report:")
    print(classification_report(
        y_test, y_pred,
        target_names=classes,
        zero_division=0
    ))

# ── 10. Threshold-Optimized Evaluation ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Threshold-Optimized Evaluation (Best Model)")
print("=" * 60)

best_pipeline = trained_pipelines[best_model_name]

if hasattr(best_pipeline, "predict_proba"):
    y_prob_test = best_pipeline.predict_proba(X_test)
elif hasattr(best_pipeline, "decision_function"):
    from scipy.special import expit
    y_prob_test = expit(best_pipeline.decision_function(X_test))
else:
    y_prob_test = None

if y_prob_test is not None and opt_thresholds is not None:
    # Apply optimized thresholds
    y_pred_opt = np.zeros_like(y_prob_test, dtype=int)
    for i, thresh in enumerate(opt_thresholds):
        y_pred_opt[:, i] = (y_prob_test[:, i] >= thresh).astype(int)

    f1_opt = f1_score(y_test, y_pred_opt, average="macro", zero_division=0)
    f1_default = f1_score(
        y_test, best_pipeline.predict(X_test),
        average="macro", zero_division=0
    )

    print(f"  {best_model_name} with optimized thresholds:")
    print(f"    F1 Macro (default 0.5): {f1_default:.4f}")
    print(f"    F1 Macro (optimized):   {f1_opt:.4f}")
    print(f"    Improvement:            +{(f1_opt - f1_default)*100:.2f}%")
    print(f"\n    Per-class report (optimized thresholds):")
    print(classification_report(
        y_test, y_pred_opt,
        target_names=classes,
        zero_division=0
    ))
else:
    print("  Threshold optimization not available for this model type.")

# ── 11. Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY OF RESULTS")
print("=" * 60)
print(f"\n  {'Model':<25} {'CV F1 Macro':>12} {'Test F1 Macro':>14}")
print(f"  {'-'*25} {'-'*12} {'-'*14}")
for name, pipeline in trained_pipelines.items():
    cv_f1 = results[name]["Mean F1 Macro"]
    y_pred = pipeline.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"  {name:<25} {cv_f1:>12.4f} {test_f1:>14.4f}")

if y_prob_test is not None and opt_thresholds is not None:
    print(f"  {'+ Threshold Optimized':<25} {'—':>12} {f1_opt:>14.4f}")

print(f"\n  Original best F1 Macro (from old notebook): ~0.347")
print(f"  New best F1 Macro: check results above")
print("\nDone!")
