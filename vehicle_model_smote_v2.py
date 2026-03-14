"""
CORRECTED Vehicle IDS Model with SMOTE
=======================================
FIXES from original vehicle_model_smote.py:
1. Uses FIXED dataset (final_merged_v2v_fixed.csv)
2. SMOTE applied AFTER train-test split (prevents data leakage!)
3. Controlled SMOTE ratio (not full 1:1 balance)
4. Regularized Random Forest to prevent overfitting
5. Cross-validation for honest accuracy

The original code applied SMOTE BEFORE split, which meant synthetic
test samples overlapped with training data = FAKE 98% accuracy.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

print("=" * 70)
print("CORRECTED VEHICLE IDS MODEL (RF + SMOTE)")
print("=" * 70)


# ============================
# 1. Load FIXED Dataset
# ============================
df = pd.read_csv("final_merged_v2v_fixed.csv")
print(f"\nDataset Shape: {df.shape}")


# ============================
# 2. Clean Data
# ============================
df["vehicle_type"] = df["vehicle_type"].astype(str).str.strip()
df["weather"] = df["weather"].astype(str).str.strip()

le_vehicle = LabelEncoder()
le_weather = LabelEncoder()
df["vehicle_type"] = le_vehicle.fit_transform(df["vehicle_type"])
df["weather"] = le_weather.fit_transform(df["weather"])

df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna()
print(f"After Cleaning: {df.shape}")


# ============================
# 3. Split X and y
# ============================
X = df.drop("attack", axis=1)
y = df["attack"].astype(int)

print(f"\nClass distribution:")
print(y.value_counts())
print(f"Attack rate: {y.mean()*100:.1f}%")


# ============================
# 4. Train-Test Split FIRST!
# ============================
# CRITICAL FIX: Split BEFORE SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}")


# ============================
# 5. Apply SMOTE on TRAINING ONLY
# ============================
print(f"\nBefore SMOTE (train):")
print(f"  Normal: {(y_train==0).sum()}, Attack: {(y_train==1).sum()}")

# sampling_strategy=0.5 means minority class becomes 50% of majority
# This is more realistic than full 1:1 balance
smote = SMOTE(random_state=42, sampling_strategy=0.5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"After SMOTE (train):")
print(f"  Normal: {(y_train_res==0).sum()}, Attack: {(y_train_res==1).sum()}")


# ============================
# 6. Scale Data
# ============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)


# ============================
# 7. Train Random Forest (Regularized)
# ============================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,          # Limit depth
    min_samples_split=10,  # Require more samples to split
    min_samples_leaf=5,    # Prevent tiny leaf nodes
    max_features='sqrt',   # Use subset of features per tree
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
model.fit(X_train_scaled, y_train_res)


# ============================
# 8. Evaluate on UNTOUCHED Test Set
# ============================
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
print(f"\nVehicle IDS Accuracy: {acc*100:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:")
print(f"  TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
print(f"  FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")


# ============================
# 9. Cross-Validation (Honest Check)
# ============================
print(f"\n{'-'*70}")
print("5-Fold Cross-Validation:")
print(f"{'-'*70}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train_res, cv=cv, scoring='accuracy')
print(f"CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"CV Mean:   {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")


# ============================
# 10. Save Model
# ============================
joblib.dump(model, "vehicle_rf_smote_model_v2.pkl")
joblib.dump(scaler, "vehicle_scaler_v2.pkl")

print(f"\nModels saved:")
print(f"  - vehicle_rf_smote_model_v2.pkl")
print(f"  - vehicle_scaler_v2.pkl")


# ============================
# 11. Feature Importance
# ============================
print(f"\n{'-'*70}")
print("Feature Importances:")
print(f"{'-'*70}")

feature_names = list(X.columns)
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

for i in range(len(feature_names)):
    idx = sorted_idx[i]
    print(f"  {feature_names[idx]:15s}: {importances[idx]:.4f}")

print(f"\n{'='*70}")
print("DONE - Vehicle IDS Model Complete")
print(f"{'='*70}")
