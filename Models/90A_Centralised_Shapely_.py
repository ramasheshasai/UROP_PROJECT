import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import shap
import matplotlib.pyplot as plt

# ==============================
# 1️⃣ Load & merge all 4 datasets
# ==============================
dataset_paths = [
    "Dataset/processed.cleveland.data",
    "Dataset/processed.hungarian.data",
    "Dataset/processed.switzerland.data",
    "Dataset/processed.va.data"
]

df_list = []
for path in dataset_paths:
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        continue
    df = pd.read_csv(path, header=None)
    df.replace('?', np.nan, inplace=True)

    # Impute missing values using median
    imputer = SimpleImputer(strategy="median")
    df = pd.DataFrame(imputer.fit_transform(df))

    if df.empty:
        print(f"Skipping {path} (empty after cleaning)")
        continue
    df_list.append(df)

# Combine all hospitals into one centralized dataset
df = pd.concat(df_list, ignore_index=True)
print(f"\nCombined dataset shape: {df.shape}")

# ==============================
# 2️⃣ Preprocessing
# ==============================
X = df.iloc[:, :-1].values
y = (df.iloc[:, -1] > 0).astype(int).values  # Convert to binary

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 3️⃣ Define base models
# ==============================
base_models = [
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=300, max_depth=8, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
]

# ==============================
# 4️⃣ Stacking (K-Fold based)
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
meta_model = RidgeClassifier()
meta_model.fit(train_meta, y_train)
stack_pred = meta_model.predict(test_meta)

# ==============================
# 6️⃣ Evaluate
# ==============================
acc = accuracy_score(y_test, stack_pred)
report = classification_report(y_test, stack_pred)

print(f"\nPower Boost Ensemble Accuracy (All 4 Datasets): {acc:.4f}")
print("\nClassification Report:\n", report)

# ---------- Save results ----------
save_path = r"C:\Users\InduS\OneDrive\Desktop\UROP Project\Project-Test-1\HeartDisease\results\Centralised_Shapely"
os.makedirs(save_path, exist_ok=True)
output_path = os.path.join(save_path, "power_ensemble_all_datasets_report.txt")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Power Boost Ensemble Accuracy (All 4 Datasets): {acc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print(f"\nResults saved at: {output_path}")

# ==============================
# 7️⃣ SHAP Explainability (Meta-Model)
# ==============================
print("\nRunning SHAP explainability analysis...")

masker = shap.maskers.Independent(train_meta)
explainer = shap.Explainer(meta_model, masker)
shap_values = explainer(test_meta)

# ---------- SHAP Summary Plot ----------
plt.figure()
shap.summary_plot(
    shap_values.values,
    test_meta,
    feature_names=[name for name, _ in base_models],
    show=False
)
plt.title("SHAP Summary Plot - Centralized Meta-Model (All 4 Datasets)")
plt.savefig(os.path.join(save_path, "shap_summary_meta_model.png"), bbox_inches="tight")
plt.close()

# ---------- SHAP Bar Plot ----------
plt.figure()
shap.summary_plot(
    shap_values.values,
    test_meta,
    feature_names=[name for name, _ in base_models],
    plot_type="bar",
    show=False
)
plt.title("SHAP Feature Importance (Bar Plot) - Centralized Meta-Model")
plt.savefig(os.path.join(save_path, "shap_bar_plot_meta_model.png"), bbox_inches="tight")
plt.close()

# ---------- SHAP Single Prediction ----------
sample_index = 5
plt.figure()
shap.plots.force(
    explainer.expected_value,
    shap_values.values[sample_index, :],
    feature_names=[name for name, _ in base_models],
    matplotlib=True,
    show=False
)
plt.savefig(os.path.join(save_path, "shap_single_prediction.png"), bbox_inches="tight")
plt.close()

print(f"SHAP plots saved successfully in: {save_path}")
