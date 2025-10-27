import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle
import os

# ==============================
# 1️⃣ Load & merge all 4 datasets (with safety check)
# ==============================
files = [
    "Dataset/processed.cleveland.data",
    "Dataset/processed.hungarian.data",
    "Dataset/processed.switzerland.data",
    "Dataset/processed.va.data"
]

df_list = []
for file in files:
    temp = pd.read_csv(file, header=None)
    temp.replace('?', pd.NA, inplace=True)
    temp = temp.dropna().apply(pd.to_numeric, errors='coerce')
    temp = temp.dropna(how='all')  # ✅ remove fully empty rows
    if not temp.empty:
        df_list.append(temp)

# ✅ Concatenate only non-empty datasets
df = pd.concat(df_list, ignore_index=True)

# ==============================
# 2️⃣ Preprocess data
# ==============================
X = df.iloc[:, :-1].values
y = (df.iloc[:, -1] > 0).astype(int).values  # binary 0/1

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==============================
# 3️⃣ Define base models (tuned)
# ==============================
base_models = [
    ('rf', RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_split=4,
        min_samples_leaf=2, random_state=42
    )),
    ('et', ExtraTreesClassifier(
        n_estimators=500, max_depth=10, min_samples_split=4,
        min_samples_leaf=2, random_state=42
    )),
    ('gb', GradientBoostingClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=5,
        subsample=0.9, random_state=42
    )),
    ('lr', LogisticRegression(
        solver='liblinear', C=1.5, max_iter=1500, random_state=42
    ))
]

# ==============================
# 4️⃣ Stacking using K-Fold
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

    test_meta[:, i] = test_fold_preds.mean(axis=1)

# ==============================
# 5️⃣ Meta-model (Ridge Classifier)
# ==============================
meta_model = RidgeClassifier(alpha=0.5)
meta_model.fit(train_meta, y_train)
stack_pred = meta_model.predict(test_meta)

# ==============================
# 6️⃣ Evaluate results
# ==============================
acc = accuracy_score(y_test, stack_pred)
report = classification_report(y_test, stack_pred)

print("Power Boost Ensemble Accuracy:", acc)
print("\nClassification Report:\n", report)

# ==============================
# 7️⃣ Save results
# ==============================
folder_path = r"C:\Users\InduS\OneDrive\Desktop\UROP Project\Project-Test-1\HeartDisease\results\centralised"
os.makedirs(folder_path, exist_ok=True)

output_path = os.path.join(folder_path, "power_ensemble_report.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Power Boost Ensemble Accuracy: {acc}\n\n")
    f.write(report)

print(f"\nResults saved successfully at: {output_path}")
