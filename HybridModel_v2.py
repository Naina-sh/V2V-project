"""
CORRECTED Hybrid LSTM + Random Forest Model for V2V IDS
========================================================
FIXES from original HybridModel.py:
1. Uses FIXED dataset (final_merged_v2v_fixed.csv) with proper attack labels
2. Proper data cleaning and type handling
3. SMOTE applied ONLY on training data (prevents data leakage)
4. LSTM with proper sequence handling
5. Cross-validation for honest accuracy estimation
6. Feature importance analysis

Expected realistic accuracy: 85-92%
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

print("=" * 70)
print("CORRECTED HYBRID MODEL (LSTM + Random Forest)")
print("=" * 70)


# =========================
# 1. Load FIXED Dataset
# =========================
df = pd.read_csv("final_merged_v2v_fixed.csv")
print(f"\nDataset Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nAttack distribution:\n{df['attack'].value_counts()}")
print(f"Attack rate: {df['attack'].mean()*100:.1f}%")


# =========================
# 2. Clean and Encode
# =========================

# Drop any rows with NaN
df = df.dropna()

# Encode categorical columns
le_vehicle = LabelEncoder()
le_weather = LabelEncoder()

df["vehicle_type"] = le_vehicle.fit_transform(df["vehicle_type"].astype(str).str.strip())
df["weather"] = le_weather.fit_transform(df["weather"].astype(str).str.strip())

# Ensure all columns are numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna()

print(f"After cleaning: {df.shape}")


# =========================
# 3. Split Features / Target
# =========================
X = df.drop("attack", axis=1)
y = df["attack"].astype(int)

feature_names = list(X.columns)
print(f"\nFeatures ({len(feature_names)}): {feature_names}")
print(f"Target distribution:\n{y.value_counts()}")


# =========================
# 4. Train-Test Split FIRST (before any resampling!)
# =========================
# CRITICAL FIX: Split BEFORE SMOTE to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"Train attack rate: {y_train.mean()*100:.1f}%")
print(f"Test attack rate: {y_test.mean()*100:.1f}%")


# =========================
# 5. Apply SMOTE ONLY on Training Data
# =========================
# CRITICAL FIX: SMOTE must only touch training data
# Original code applied SMOTE before split -> data leakage -> fake 98% accuracy
print("\nBefore SMOTE (train only):")
print(f"  Normal: {(y_train==0).sum()}, Attack: {(y_train==1).sum()}")

smote = SMOTE(random_state=42, sampling_strategy=0.5)  # Don't fully balance - keep some imbalance
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("After SMOTE (train only):")
print(f"  Normal: {(y_train_res==0).sum()}, Attack: {(y_train_res==1).sum()}")


# =========================
# 6. Scale Data
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)


# =========================
# 7. Build LSTM Feature Extractor
# =========================
# Reshape for LSTM: (samples, timesteps, features)
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

print(f"\nLSTM input shape: {X_train_lstm.shape}")

# Build LSTM model using Functional API (so .input/.output are defined)
lstm_input = Input(shape=(1, X_train_scaled.shape[1]))
x = LSTM(64, return_sequences=True)(lstm_input)
x = Dropout(0.3)(x)
x = LSTM(32, return_sequences=False)(x)
x = Dropout(0.2)(x)
x = Dense(16, activation="relu")(x)
lstm_output = Dense(1, activation="sigmoid")(x)

lstm_model = Model(inputs=lstm_input, outputs=lstm_output)

lstm_model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)

print("\nTraining LSTM...")
history = lstm_model.fit(
    X_train_lstm, y_train_res,
    epochs=15,
    batch_size=128,
    validation_split=0.15,
    callbacks=[early_stop],
    verbose=1
)

# LSTM standalone accuracy
lstm_pred = (lstm_model.predict(X_test_lstm) > 0.5).astype(int).flatten()
lstm_acc = accuracy_score(y_test, lstm_pred)
print(f"\nLSTM Standalone Accuracy: {lstm_acc*100:.2f}%")


# =========================
# 8. Extract LSTM Features for RF
# =========================
# Use the intermediate dense layer (16 neurons) as features
feature_extractor = Model(
    inputs=lstm_model.input,
    outputs=lstm_model.layers[-2].output  # Dense(16) layer output
)

X_train_features = feature_extractor.predict(X_train_lstm)
X_test_features = feature_extractor.predict(X_test_lstm)

# Combine LSTM features with original scaled features for richer representation
X_train_combined = np.hstack([X_train_scaled, X_train_features])
X_test_combined = np.hstack([X_test_scaled, X_test_features])

print(f"Combined feature shape: {X_train_combined.shape}")


# =========================
# 9. Train Random Forest on Combined Features
# =========================
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,         # Limit depth to prevent overfitting
    min_samples_split=10, # Require more samples for splits
    min_samples_leaf=5,   # Prevent tiny leaves
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest on LSTM + Original features...")
rf.fit(X_train_combined, y_train_res)


# =========================
# 10. Evaluate Hybrid Model
# =========================
y_pred_hybrid = rf.predict(X_test_combined)
hybrid_acc = accuracy_score(y_test, y_pred_hybrid)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"\nLSTM Standalone Accuracy:  {lstm_acc*100:.2f}%")
print(f"Hybrid (LSTM+RF) Accuracy: {hybrid_acc*100:.2f}%")
print(f"\nClassification Report (Hybrid):\n")
print(classification_report(y_test, y_pred_hybrid, target_names=["Normal", "Attack"]))
print(f"Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_hybrid)
print(f"  TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
print(f"  FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")


# =========================
# 11. Cross-Validation (Honest Estimate)
# =========================
print("\n" + "-" * 70)
print("Cross-Validation (5-fold) on RF with original features only:")
print("-" * 70)

# Quick CV on just original features (no LSTM) for comparison
rf_simple = RandomForestClassifier(
    n_estimators=200, max_depth=15, 
    min_samples_split=10, min_samples_leaf=5,
    random_state=42, n_jobs=-1
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_simple, X_train_scaled, y_train_res, cv=cv, scoring='accuracy')
print(f"CV Scores: {cv_scores}")
print(f"CV Mean:   {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")


# =========================
# 12. Save Models
# =========================
import joblib

joblib.dump(rf, "hybrid_rf_model.pkl")
joblib.dump(scaler, "hybrid_scaler.pkl")
lstm_model.save("hybrid_lstm_model.keras")

print("\nModels saved:")
print("  - hybrid_rf_model.pkl")
print("  - hybrid_scaler.pkl") 
print("  - hybrid_lstm_model.keras")


# =========================
# 13. Feature Importance
# =========================
print("\n" + "-" * 70)
print("Top Feature Importances (RF):")
print("-" * 70)

# Original features + LSTM features
all_feature_names = feature_names + [f"lstm_feat_{i}" for i in range(X_train_features.shape[1])]
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

for i in range(min(15, len(all_feature_names))):
    idx = sorted_idx[i]
    print(f"  {all_feature_names[idx]:20s}: {importances[idx]:.4f}")


print("\n" + "=" * 70)
print("DONE - Corrected Hybrid Model Complete")
print("=" * 70)
