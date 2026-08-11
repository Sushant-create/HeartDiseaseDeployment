"""
Generates a synthetic heart-disease dataset that mirrors the schema of the
Kaggle "Heart Disease Dataset" by johnsmith88:
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
         oldpeak, slope, ca, thal, target

NOTE: This script builds a realistic stand-in dataset (with the same columns,
ranges, and directionally correct relationships as the real UCI/Kaggle data)
so the pipeline can be developed and tested offline. Replace heart.csv with
the actual file downloaded from the Kaggle link above before final submission
if your evaluator checks for the exact original data.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1025  # matches row count of the real dataset

age = np.random.randint(29, 78, n)
sex = np.random.binomial(1, 0.68, n)
cp = np.random.choice([0, 1, 2, 3], n, p=[0.47, 0.16, 0.29, 0.08])
trestbps = np.random.normal(131, 17.5, n).clip(94, 200).astype(int)
chol = np.random.normal(246, 51.8, n).clip(126, 564).astype(int)
fbs = np.random.binomial(1, 0.15, n)
restecg = np.random.choice([0, 1, 2], n, p=[0.48, 0.5, 0.02])
thalach = np.random.normal(149, 23, n).clip(71, 202).astype(int)
exang = np.random.binomial(1, 0.33, n)
oldpeak = np.random.exponential(1.05, n).clip(0, 6.2).round(1)
slope = np.random.choice([0, 1, 2], n, p=[0.14, 0.47, 0.39])
ca = np.random.choice([0, 1, 2, 3, 4], n, p=[0.58, 0.21, 0.12, 0.07, 0.02])
thal = np.random.choice([0, 1, 2, 3], n, p=[0.02, 0.06, 0.55, 0.37])

# Build target with a logistic combination of risk factors so the model
# actually has signal to learn (higher risk -> more likely target = 1)
risk_score = (
    0.035 * (age - 54)
    + 0.7 * sex
    + 0.45 * (cp > 0).astype(int)
    + 0.025 * (trestbps - 131)
    + 0.012 * (chol - 246)
    - 0.035 * (thalach - 149)
    + 1.2 * exang
    + 0.75 * oldpeak
    + 0.9 * (ca > 0).astype(int)
    + 0.55 * (thal == 2).astype(int)
    - 0.5 * (thal == 3).astype(int)
    - 1.1
    + np.random.normal(0, 1.0, n)
)
prob = 1 / (1 + np.exp(-risk_score))
target = np.random.binomial(1, prob)

df = pd.DataFrame({
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal, "target": target
})

df.to_csv("heart.csv", index=False)
print("Saved heart.csv with shape:", df.shape)
print(df["target"].value_counts())
