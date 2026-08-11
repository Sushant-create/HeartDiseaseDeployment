"""
main.py
=======
Direct launch file for local development.

Run with:
    python main.py

This is a thin wrapper around app.py so the project can be started either
way (`python app.py` or `python main.py`). On Render, the app is served via
gunicorn using the command in the Procfile / render.yaml, not this file.
"""

from app import app

if __name__ == "__main__":
    print("Starting Heart Disease Prediction API on http://127.0.0.1:5000")
    print("Try:  python test_samples.py")
    print('Or:   curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d \'{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}\'')
    app.run(host="0.0.0.0", port=5000, debug=True)
