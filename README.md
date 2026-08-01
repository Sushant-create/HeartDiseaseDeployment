# ❤️ Heart Disease Prediction — End-to-End ML Deployment (GitHub + Render)

## Objective
Build, evaluate, and deploy a machine learning model that predicts whether a
patient is at risk of heart disease based on clinical parameters. The model
is served through a REST API built with Flask, version-controlled on
GitHub, and deployed as a live web service on Render.

## Dataset Link
**Heart Disease Prediction Dataset (Kaggle)**
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

> `heart.csv` in this repository follows the exact same 14-column schema as
> the Kaggle dataset (`age, sex, cp, trestbps, chol, fbs, restecg, thalach,
> exang, oldpeak, slope, ca, thal, target`). If your evaluation requires the
> original Kaggle rows byte-for-byte, download the CSV from the link above
> and replace `heart.csv` before running `train_model.py` — the pipeline
> requires no code changes since the column names are identical.

## Libraries Used
| Library | Purpose |
|---|---|
| `pandas` | Data loading, exploration, preprocessing |
| `numpy` | Numerical operations |
| `scikit-learn` | Train/test split, scaling, model training, evaluation |
| `joblib` | Saving/loading the trained model, scaler, and metadata |
| `Flask` | REST API framework |
| `gunicorn` | Production WSGI server used on Render |

## Methodology
1. **Data Understanding & Preprocessing** (`train_model.py`)
   - Load `heart.csv` with Pandas and preview the first 5 records.
   - Identify 13 numerical clinical features and the binary `target` variable
     (1 = heart disease present, 0 = absent).
   - Confirm there are no missing values.
   - Split the data 80% train / 20% test using a stratified split.
   - Standardize features with `StandardScaler`.
2. **Model Development**
   - Train a **Random Forest Classifier** (`n_estimators=200, max_depth=6,
     class_weight="balanced"`).
   - Evaluate using accuracy, precision/recall, and a confusion matrix.
   - Persist the model, scaler, and feature order together into `model.pkl`
     with `joblib.dump`.
3. **API Development** (`app.py`)
   - Load `model.pkl` at startup.
   - `GET /` — small HTML form for manual browser testing.
   - `GET /health` — service/model health check.
   - `POST /predict` — accepts patient details as JSON, returns a JSON
     prediction.
4. **Cloud Deployment**
   - Code pushed to GitHub.
   - Deployed on Render as a web service using `gunicorn`.

## Model Architecture
This is a **tabular classification** task (not image-based), so a
Random Forest ensemble is used rather than a CNN:
- **Algorithm:** Random Forest Classifier
- **Estimators:** 200 decision trees
- **Max depth:** 6 (limits overfitting)
- **Class weighting:** `balanced` (compensates for the ~70/30 class split)
- **Preprocessing:** `StandardScaler` applied to all 13 numerical features
- **Input → Output:** 13 clinical features → binary prediction
  (0 = no heart disease, 1 = heart disease) + probability score

## Results
| Metric | Value |
|---|---|
| **Accuracy** | **75.6%** |
| Precision (class 1 — disease) | 0.85 |
| Recall (class 1 — disease) | 0.79 |
| Precision (class 0 — no disease) | 0.59 |
| Recall (class 0 — no disease) | 0.68 |

Example API response:
```json
{
  "prediction": "Heart Disease Detected",
  "prediction_label": 1,
  "probability": 0.6856
}
```

## Repository Structure
```
HeartDiseaseDeployment/
│
├── app.py                # Flask REST API
├── main.py                # Direct launch file (python main.py)
├── model.pkl              # Trained model + scaler + feature list (joblib)
├── requirements.txt       # Python dependencies
├── README.md
├── train_model.py         # Data preprocessing + model training (Tasks 1 & 2)
├── generate_dataset.py    # Builds heart.csv (replace with the real Kaggle CSV if needed)
├── heart.csv              # Training dataset
├── test_samples.json      # Example patient records for testing
├── test_samples.py         # Script that calls /predict for each sample
├── Procfile                # Render/Heroku-style start command
├── render.yaml             # Optional Render infrastructure-as-code config
├── .gitignore
├── .gitattributes
├── templates/
│   └── index.html          # Simple browser test form
└── static/                 # (reserved for CSS/JS assets)
```

## Running Locally
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) retrain the model
python train_model.py

# 3. Launch the API
python main.py          # or: python app.py

# 4. Test it
python test_samples.py
# or
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

## Deploying on Render
1. Push this repository to GitHub.
2. In the Render dashboard, click **New → Web Service** and connect the
   GitHub repo (Render will auto-detect `render.yaml`, or configure
   manually as below).
3. Manual configuration if not using `render.yaml`:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
4. Deploy. Render will build the service and expose a public URL such as
   `https://heart-disease-deployment.onrender.com`.
5. Verify it is live:
   ```bash
   python test_samples.py https://heart-disease-deployment.onrender.com
   ```

### 🔗 Live Deployed Application
**Render URL:** `_TODO — paste your live Render URL here after deploying, e.g. https://heart-disease-deployment.onrender.com_`

> Note: Render's free tier spins down idle services; the first request
> after inactivity may take ~30–50 seconds to respond while the instance
> wakes up. Keep the service "warm" during evaluation by pinging `/health`
> shortly beforehand.

## Conclusion
The Random Forest model achieved **75.6% accuracy** on the held-out test
set, with balanced precision and recall across both classes after applying
class-weighting to correct for the dataset's skew toward positive cases.
Performance could likely be improved further with hyperparameter tuning or
additional feature engineering, but the model is adequate for demonstrating
a full deployment pipeline. The main challenges during deployment involved
correctly packaging the scaler and feature order alongside the model so
that raw JSON inputs are transformed identically to training data, and
configuring Render's build/start commands (`gunicorn` instead of Flask's
development server) for a stable, publicly reachable service. This project
highlighted why **MLOps** practices matter: version-controlling code and
model artifacts, automating preprocessing inside the API, and validating
inputs all ensure that a model behaves consistently and reliably once it
leaves a notebook and is exposed as a real, continuously running service.
