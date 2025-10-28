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

# ==============================
# 8️⃣ SHAP Feature Importance Analysis (on Gradient Boosting)
# ==============================
import shap
import matplotlib.pyplot as plt

# Path where results will be stored
folder_path = r"C:\Users\InduS\OneDrive\Desktop\UROP Project\Project-Test-1\HeartDisease\results\centralised"
os.makedirs(folder_path, exist_ok=True)

print("\nStarting SHAP feature importance analysis...")

# Select and fit Gradient Boosting model
gb_model = [m for n, m in base_models if n == 'gb'][0]
gb_model.fit(X_train, y_train)

# Create SHAP explainer and compute values (disable additivity check to avoid error)
explainer = shap.Explainer(gb_model, X_train)
shap_values = explainer(X_test, check_additivity=False)

# Define feature names (UCI Heart Disease dataset)
feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

# ==============================
# Save SHAP Summary Plot (Dot Plot)
# ==============================
plt.figure()
shap.summary_plot(shap_values, features=X_test, feature_names=feature_names, show=False)
summary_path = os.path.join(folder_path, "shap_summary_plot.png")
plt.savefig(summary_path, bbox_inches="tight", dpi=300)
plt.close()
print(f" SHAP summary plot saved at: {summary_path}")

# ==============================
# Save SHAP Bar Plot (Global Importance)
# ==============================
plt.figure()
shap.summary_plot(shap_values, features=X_test, feature_names=feature_names, plot_type="bar", show=False)
bar_path = os.path.join(folder_path, "shap_bar_plot.png")
plt.savefig(bar_path, bbox_inches="tight", dpi=300)
plt.close()
print(f" SHAP bar plot saved at: {bar_path}")

# ==============================
# Save Top 10 Features as Text
# ==============================
# Get mean absolute SHAP values per feature
shap_mean = np.abs(shap_values.values).mean(axis=0)
feature_importance = sorted(zip(feature_names, shap_mean), key=lambda x: x[1], reverse=True)

# Save top 10 features to file
txt_path = os.path.join(folder_path, "shap_feature_importance.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("Top 10 Most Important Features (Based on SHAP values)\n")
    f.write("============================================\n")
    for i, (feat, val) in enumerate(feature_importance[:10], start=1):
        f.write(f"{i}. {feat} — {val:.4f}\n")

print(f" SHAP feature importance text file saved at: {txt_path}")
print("\nSHAP feature analysis completed successfully!")
