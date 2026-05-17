# ==========================================================
# utils.py
# Fraud Detection Utility Functions
# ==========================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
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
# DATA PREPROCESSING
# ==========================================================

def preprocess_data(df):

    """
    Preprocess dataset:
    - Handle missing values
    - Scale numerical features
    - Feature engineering
    """

    # Remove missing values
    df = df.dropna()

    # Feature Scaling
    scaler = StandardScaler()

    df['Amount'] = scaler.fit_transform(
        df['Amount'].values.reshape(-1, 1)
    )

    df['Time'] = scaler.fit_transform(
        df['Time'].values.reshape(-1, 1)
    )

    # ======================================================
    # FEATURE ENGINEERING
    # ======================================================

    # Transaction Frequency Feature
    df['Transaction_Frequency'] = (
        df.groupby('Amount')['Amount']
        .transform('count')
    )

    # Behavioral Feature
    df['Behavior_Score'] = df['V1'] * df['V2']

    return df


# ==========================================================
# SPLIT FEATURES AND TARGET
# ==========================================================

def split_data(df):

    """
    Split dataset into X and y
    """

    X = df.drop('Class', axis=1)
    y = df['Class']

    return X, y


# ==========================================================
# EVALUATE MODEL
# ==========================================================

def evaluate_model(model, X_test, y_test):

    """
    Evaluate ML model performance
    """

    predictions = model.predict(X_test)

    # For AUC
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = predictions

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    auc = roc_auc_score(y_test, probabilities)

    return {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4),
        "AUC": round(auc, 4)
    }


# ==========================================================
# SAVE RESULTS
# ==========================================================

def save_results(results, output_path):

    """
    Save results into CSV file
    """

    results_df = pd.DataFrame(results)

    results_df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")


# ==========================================================
# SAVE CONFUSION MATRIX
# ==========================================================

def save_confusion_matrix(model, X_test, y_test, output_path):

    """
    Generate and save confusion matrix
    """

    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)

    cm_df = pd.DataFrame(
        cm,
        columns=['Predicted_Normal', 'Predicted_Fraud'],
        index=['Actual_Normal', 'Actual_Fraud']
    )

    cm_df.to_csv(output_path)

    print(f"Confusion Matrix saved to {output_path}")


# ==========================================================
# SAVE CLASSIFICATION REPORT
# ==========================================================

def save_classification_report(
    model,
    X_test,
    y_test,
    output_path
):

    """
    Save classification report into txt file
    """

    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions
    )

    with open(output_path, "w") as file:
        file.write(report)

    print(f"Classification Report saved to {output_path}")


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def save_feature_importance(model, feature_names, output_path):

    """
    Save feature importance for tree models
    """

    if hasattr(model, "feature_importances_"):

        importance = model.feature_importances_

        feature_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        })

        feature_df = feature_df.sort_values(
            by='Importance',
            ascending=False
        )

        feature_df.to_csv(output_path, index=False)

        print(f"Feature Importance saved to {output_path}")

    else:
        print("Model does not support feature importance")


# ==========================================================
# SAMPLE PREDICTIONS
# ==========================================================

def save_sample_predictions(
    model,
    X_test,
    y_test,
    output_path,
    sample_size=100
):

    """
    Save sample predictions
    """

    predictions = model.predict(X_test)

    sample_df = pd.DataFrame({
        "Actual": y_test.values[:sample_size],
        "Predicted": predictions[:sample_size]
    })

    sample_df.to_csv(output_path, index=False)

    print(f"Sample Predictions saved to {output_path}")


# ==========================================================
# REAL-TIME PREDICTION
# ==========================================================

def predict_transaction(model, transaction_data):

    """
    Predict single transaction
    """

    prediction = model.predict(transaction_data)[0]

    probability = model.predict_proba(
        transaction_data
    )[0][1]

    if prediction == 1:
        result = "Fraudulent Transaction"
    else:
        result = "Legitimate Transaction"

    return {
        "Prediction": result,
        "Fraud Probability": round(probability, 4)
    }


# ==========================================================
# PROJECT SUMMARY
# ==========================================================

def generate_project_summary(
    dataset_shape,
    train_shape,
    test_shape,
    results_df,
    output_path
):

    """
    Save project summary
    """

    summary = f"""
=================================================
Fraud Detection Project Summary
=================================================

Dataset Shape:
{dataset_shape}

Training Samples:
{train_shape}

Testing Samples:
{test_shape}

=================================================
MODEL PERFORMANCE
=================================================

{results_df}

=================================================
Project Completed Successfully
=================================================
"""

    with open(output_path, "w") as file:
        file.write(summary)

    print(f"Project Summary saved to {output_path}")