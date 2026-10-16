import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "linear2.pkl")
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Insurance Cost Predictor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(23, 32, 54, 0.7);
      --border: rgba(255, 255, 255, 0.1);
      --accent: #4f46e5;
      --accent-hover: #4338ca;
      --accent-gradient: linear-gradient(135deg, #6366f1 0%, #4338ca 100%);
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --input-bg: rgba(15, 23, 42, 0.6);
      --input-border: rgba(255, 255, 255, 0.12);
      --focus-ring: rgba(99, 102, 241, 0.35);
      --success: #10b981;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background-color: var(--bg);
      background-image: 
        radial-gradient(circle at 15% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
        radial-gradient(circle at 85% 80%, rgba(79, 70, 229, 0.12) 0%, transparent 40%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 2rem 1rem;
    }

    .container {
      width: 100%;
      max-width: 580px;
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 2.5rem;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.45);
    }

    .header {
      margin-bottom: 2rem;
      text-align: center;
    }

    .header h1 {
      font-size: 1.75rem;
      font-weight: 700;
      letter-spacing: -0.025em;
      margin-bottom: 0.5rem;
      background: linear-gradient(to right, #ffffff, #cbd5e1);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .header p {
      color: var(--text-muted);
      font-size: 0.95rem;
    }

    .form-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.25rem;
    }

    .full-width {
      grid-column: span 2;
    }

    .form-group {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }

    label {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text);
      letter-spacing: 0.01em;
    }

    input, select {
      width: 100%;
      background: var(--input-bg);
      border: 1px solid var(--input-border);
      border-radius: 10px;
      padding: 0.75rem 1rem;
      color: var(--text);
      font-size: 0.95rem;
      outline: none;
      transition: all 0.2s ease;
      font-family: inherit;
    }

    input:focus, select:focus {
      border-color: #6366f1;
      box-shadow: 0 0 0 3px var(--focus-ring);
    }

    select option {
      background-color: #1e293b;
      color: #fff;
    }

    button {
      margin-top: 1.5rem;
      width: 100%;
      padding: 0.875rem;
      border: none;
      border-radius: 10px;
      background: var(--accent-gradient);
      color: #ffffff;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.15s ease, opacity 0.2s ease;
    }

    button:hover {
      opacity: 0.92;
      transform: translateY(-1px);
    }

    button:active {
      transform: translateY(0);
    }

    .result-box {
      margin-top: 1.75rem;
      padding: 1.25rem;
      background: rgba(16, 185, 129, 0.08);
      border: 1px solid rgba(16, 185, 129, 0.25);
      border-radius: 12px;
      text-align: center;
      animation: fadeIn 0.3s ease-out;
    }

    .result-box h3 {
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 0.4rem;
    }

    .result-value {
      font-size: 1.85rem;
      font-weight: 700;
      color: var(--success);
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 520px) {
      .container {
        padding: 1.75rem;
      }
      .form-grid {
        grid-template-columns: 1fr;
      }
      .full-width {
        grid-column: span 1;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Insurance Estimator</h1>
      <p>Estimate medical insurance charges with your personal metrics</p>
    </div>

    <form method="POST" action="/">
      <div class="form-grid">
        <div class="form-group">
          <label for="age">Age</label>
          <input type="number" id="age" name="age" min="1" max="120" placeholder="e.g. 35" required value="{{ form_data.get('age', '') }}">
        </div>

        <div class="form-group">
          <label for="sex">Sex</label>
          <select id="sex" name="sex" required>
            <option value="" disabled {% if not form_data.get('sex') %}selected{% endif %}>Select</option>
            <option value="male" {% if form_data.get('sex') == 'male' %}selected{% endif %}>Male</option>
            <option value="female" {% if form_data.get('sex') == 'female' %}selected{% endif %}>Female</option>
          </select>
        </div>

        <div class="form-group">
          <label for="bmi">BMI</label>
          <input type="number" step="0.1" id="bmi" name="bmi" min="10" max="60" placeholder="e.g. 26.5" required value="{{ form_data.get('bmi', '') }}">
        </div>

        <div class="form-group">
          <label for="children">Children</label>
          <input type="number" id="children" name="children" min="0" max="10" placeholder="e.g. 2" required value="{{ form_data.get('children', '') }}">
        </div>

        <div class="form-group">
          <label for="smoker">Smoker</label>
          <select id="smoker" name="smoker" required>
            <option value="" disabled {% if not form_data.get('smoker') %}selected{% endif %}>Select</option>
            <option value="yes" {% if form_data.get('smoker') == 'yes' %}selected{% endif %}>Yes</option>
            <option value="no" {% if form_data.get('smoker') == 'no' %}selected{% endif %}>No</option>
          </select>
        </div>

        <div class="form-group">
          <label for="region">Region</label>
          <select id="region" name="region" required>
            <option value="" disabled {% if not form_data.get('region') %}selected{% endif %}>Select</option>
            <option value="northeast" {% if form_data.get('region') == 'northeast' %}selected{% endif %}>Northeast</option>
            <option value="northwest" {% if form_data.get('region') == 'northwest' %}selected{% endif %}>Northwest</option>
            <option value="southeast" {% if form_data.get('region') == 'southeast' %}selected{% endif %}>Southeast</option>
            <option value="southwest" {% if form_data.get('region') == 'southwest' %}selected{% endif %}>Southwest</option>
          </select>
        </div>
      </div>

      <button type="submit">Calculate Estimate</button>
    </form>

    {% if prediction is not none %}
    <div class="result-box">
      <h3>Estimated Annual Premium</h3>
      <div class="result-value">${{ prediction }}</div>
    </div>
    {% endif %}
  </div>
</body>
</html>
"""

def prepare_input(form_data):
    """
    Constructs input matching model features:
    ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    """
    age = float(form_data["age"])
    bmi = float(form_data["bmi"])
    children = int(form_data["children"])

    # If the model raw pipeline expects encoded integers (0/1):
    sex = 1 if form_data["sex"].lower() == "male" else 0
    smoker = 1 if form_data["smoker"].lower() == "yes" else 0
    
    # Map regions to numeric codes if model was trained with numeric encoding
    region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
    region = region_map.get(form_data["region"].lower(), 0)

    # DataFrame with feature names preserved
    return pd.DataFrame([[age, sex, bmi, children, smoker, region]], 
                        columns=["age", "sex", "bmi", "children", "smoker", "region"])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    form_data = {}

    if request.method == "POST":
        form_data = request.form
        try:
            input_df = prepare_input(form_data)
            pred = model.predict(input_df)[0]
            prediction = f"{max(0, float(pred)):,.2f}"
        except Exception:
            # Fallback if model was trained directly on string categories
            raw_df = pd.DataFrame([[
                float(form_data["age"]),
                form_data["sex"],
                float(form_data["bmi"]),
                int(form_data["children"]),
                form_data["smoker"],
                form_data["region"]
            ]], columns=["age", "sex", "bmi", "children", "smoker", "region"])
            pred = model.predict(raw_df)[0]
            prediction = f"{max(0, float(pred)):,.2f}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction, form_data=form_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
