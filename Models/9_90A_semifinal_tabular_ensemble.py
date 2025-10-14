# File: 12_final_ensemble_powerboost.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 1️⃣ Load & preprocess dataset
# ==============================
df = pd.read_csv("Dataset/processed.cleveland.data", header=None)
df.replace('?', pd.NA, inplace=True)
df = df.dropna().apply(pd.to_numeric)

X = df.iloc[:, :-1].values
y = (df.iloc[:, -1] > 0).astype(int).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==============================
# 2️⃣ Define base models
# ==============================
base_models = [
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=300, max_depth=6, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000))
]

# ==============================
# 3️⃣ Generate stacking features using K-Fold
# ==============================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_meta = np.zeros((X_train.shape[0], len(base_models)))
test_meta = np.zeros((X_test.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models):
    test_fold_preds = np.zeros((X_test.shape[0], kf.n_splits))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model.fit(X_tr, y_tr)
        train_meta[val_idx, i] = model.predict(X_val)
        test_fold_preds[:, fold] = model.predict(X_test)

    test_meta[:, i] = test_fold_preds.mean(axis=1)  # average predictions across folds

# ==============================
# 4️⃣ Meta-model
# ==============================
meta_model = RidgeClassifier()
meta_model.fit(train_meta, y_train)
stack_pred = meta_model.predict(test_meta)

# ==============================
# 5️⃣ Evaluate
# ==============================
acc = accuracy_score(y_test, stack_pred)
report = classification_report(y_test, stack_pred)

print("Power Boost Ensemble Accuracy:", acc)
print("\nClassification Report:\n", report)

with open("results/power_ensemble_report.txt", "w") as f:
    f.write(f"Power Boost Ensemble Accuracy: {acc}\n\n")
    f.write(report)
