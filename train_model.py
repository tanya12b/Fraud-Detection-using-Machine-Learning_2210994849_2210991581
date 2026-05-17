# ==========================================================
# Fraud Detection using Machine Learning
# COMPLETE WORKING train_model.py
# ==========================================================

import warnings
warnings.filterwarnings('ignore')

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==========================================================
# CREATE FOLDERS
# ==========================================================

os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("Loading Dataset...")

df = pd.read_csv('data/creditcard.csv')

print("\nDataset Loaded Successfully")
print("Dataset Shape:", df.shape)

# ==========================================================
# FEATURES & TARGET
# ==========================================================

X = df.drop('Class', axis=1)
y = df['Class']

# ==========================================================
# FEATURE SCALING
# ==========================================================

scaler = StandardScaler()

X['Amount'] = scaler.fit_transform(
    X['Amount'].values.reshape(-1, 1)
)

X['Time'] = scaler.fit_transform(
    X['Time'].values.reshape(-1, 1)
)

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

print("\nApplying Feature Engineering...")

X['Transaction_Frequency'] = (
    X.groupby('Amount')['Amount']
    .transform('count')
)

X['Behavior_Score'] = X['V1'] * X['V2']

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data :", X_test.shape)

# ==========================================================
# DEFINE MODELS
# ==========================================================

models = {

    'Logistic Regression': LogisticRegression(
        class_weight='balanced',
        max_iter=1000
    ),

    'Decision Tree': DecisionTreeClassifier(
        max_depth=10,
        class_weight='balanced',
        random_state=42
    ),

    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight='balanced',
        random_state=42
    ),

    'SVM': SVC(
        kernel='rbf',
        probability=True,
        class_weight='balanced'
    )
}

# ==========================================================
# TRAIN MODELS
# ==========================================================

results = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    auc = roc_auc_score(y_test, probabilities)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        auc
    ])

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-score :", round(f1, 4))
    print("AUC      :", round(auc, 4))

# ==========================================================
# ISOLATION FOREST
# ==========================================================

print("\nTraining Isolation Forest...")

iso_model = IsolationForest(
    contamination=0.001,
    random_state=42
)

iso_model.fit(X_train)

iso_predictions = iso_model.predict(X_test)

# Convert labels
iso_predictions = np.where(
    iso_predictions == -1,
    1,
    0
)

iso_accuracy = accuracy_score(y_test, iso_predictions)

iso_precision = precision_score(
    y_test,
    iso_predictions
)

iso_recall = recall_score(
    y_test,
    iso_predictions
)

iso_f1 = f1_score(
    y_test,
    iso_predictions
)

iso_auc = roc_auc_score(
    y_test,
    iso_predictions
)

results.append([
    'Isolation Forest',
    iso_accuracy,
    iso_precision,
    iso_recall,
    iso_f1,
    iso_auc
])

# ==========================================================
# SAVE RESULTS CSV
# ==========================================================

results_df = pd.DataFrame(results, columns=[
    'Model',
    'Accuracy',
    'Precision',
    'Recall',
    'F1-score',
    'AUC'
])

results_df.to_csv(
    'outputs/fraud_detection_results.csv',
    index=False
)

print("\nResults CSV Saved Successfully")

# ==========================================================
# BEST MODEL
# ==========================================================

rf_model = models['Random Forest']

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    rf_model,
    'models/saved_model.pkl'
)

print("Model Saved Successfully")

# ==========================================================
# PREDICTIONS
# ==========================================================

pred = rf_model.predict(X_test)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(y_test, pred)

cm_df = pd.DataFrame(cm)

cm_df.to_csv(
    'outputs/confusion_matrix.csv',
    index=False
)

print("Confusion Matrix Saved")

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

report = classification_report(
    y_test,
    pred
)

with open(
    'outputs/classification_report.txt',
    'w'
) as file:

    file.write(report)

print("Classification Report Saved")

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance = rf_model.feature_importances_

feature_df = pd.DataFrame({

    'Feature': X.columns,
    'Importance': importance
})

feature_df = feature_df.sort_values(
    by='Importance',
    ascending=False
)

feature_df.to_csv(
    'outputs/feature_importance.csv',
    index=False
)

print("Feature Importance Saved")

# ==========================================================
# SAMPLE PREDICTIONS
# ==========================================================

sample_df = pd.DataFrame({

    'Actual': y_test.values[:100],
    'Predicted': pred[:100]
})

sample_df.to_csv(
    'outputs/predictions_sample.csv',
    index=False
)

print("Sample Predictions Saved")

# ==========================================================
# PROJECT SUMMARY
# ==========================================================

summary = f"""

====================================================
Fraud Detection Project Summary
====================================================

Dataset Shape:
{df.shape}

Training Samples:
{X_train.shape[0]}

Testing Samples:
{X_test.shape[0]}

====================================================
MODEL PERFORMANCE
====================================================

{results_df}

====================================================
PROJECT COMPLETED SUCCESSFULLY
====================================================

"""

with open(
    'outputs/model_summary.txt',
    'w'
) as file:

    file.write(summary)

print("\nProject Summary Saved")

# ==========================================================
# FINAL MESSAGE
# ==========================================================

print("\n======================================")
print("PROJECT EXECUTED SUCCESSFULLY")
print("======================================")
