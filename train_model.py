"""
train_model.py
================
End-to-End Machine Learning Model Deployment - Heart Disease Prediction

Covers:
  Task 1: Data Understanding and Preprocessing
  Task 2: Model Development

Dataset: Heart Disease Prediction Dataset (Kaggle)
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

RANDOM_STATE = 42


def main():
    # ------------------------------------------------------------------
    # Task 1: Data Understanding and Preprocessing
    # ------------------------------------------------------------------
    print("=" * 60)
    print("TASK 1: Data Understanding and Preprocessing")
    print("=" * 60)

    # 1. Load the dataset using Pandas
    df = pd.read_csv("heart.csv")

    # 2. Display the first five records
    print("\nFirst 5 records:")
    print(df.head())

    # 3. Identify numerical features and the target variable
    target_col = "target"
    numerical_features = [c for c in df.columns if c != target_col]
    print(f"\nNumerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Target variable: '{target_col}' "
          f"(1 = presence of heart disease, 0 = absence)")

    # 4. Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # 5. Split the dataset into 80% training and 20% testing
    X = df[numerical_features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Testing set shape:  {X_test.shape}")

    # Feature scaling (helps stability of the model / API inputs)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # Task 2: Model Development
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TASK 2: Model Development")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy Score: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save trained model + scaler + feature order using Joblib
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_names": numerical_features,
            "accuracy": accuracy,
        },
        "model.pkl",
    )
    print("\nSaved trained model, scaler, and feature order to model.pkl")


if __name__ == "__main__":
    main()
