# File: app.py

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# ==============================
# 1️⃣ Load trained ensemble models
# ==============================

# If you want, you can save your trained models and scaler using joblib after training
# Example:
# joblib.dump(base_models, 'models/base_models.pkl')
# joblib.dump(meta_model, 'models/meta_model.pkl')
# joblib.dump(scaler, 'models/scaler.pkl')

# For simplicity, I will recreate them here as in your script
base_models = [
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=300, max_depth=6, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000))
]

meta_model = RidgeClassifier()
scaler = StandardScaler()

# ==============================
# 2️⃣ Function to train ensemble
# ==============================
# Load and prepare dataset to train inside Flask (or load pre-trained models)
df = pd.read_csv("../Dataset/processed.cleveland.data", header=None)
df.replace('?', pd.NA, inplace=True)
df = df.dropna().apply(pd.to_numeric)

X = df.iloc[:, :-1].values
y = (df.iloc[:, -1] > 0).astype(int).values

X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_meta = np.zeros((X_scaled.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models):
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
        X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model.fit(X_tr, y_tr)
        train_meta[val_idx, i] = model.predict(X_val)

# Fit meta-model
meta_model.fit(train_meta, y)

# ==============================
# 3️⃣ Routes
# ==============================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get feature inputs from form
        input_features = [float(request.form[f'f{i}']) for i in range(X.shape[1])]
        input_array = np.array(input_features).reshape(1, -1)

        # Scale features
        input_scaled = scaler.transform(input_array)

        # Generate base model predictions
        base_preds = np.zeros((1, len(base_models)))
        for i, (name, model) in enumerate(base_models):
            base_preds[0, i] = model.predict(input_scaled)

        # Meta-model prediction
        final_pred = meta_model.predict(base_preds)[0]

        result = "Heart Disease Detected" if final_pred == 1 else "No Heart Disease Detected"
        return render_template('index.html', prediction=result)
    except Exception as e:
        return str(e)

# ==============================
# 4️⃣ Run App
# ==============================
if __name__ == '__main__':
    app.run(debug=True)
