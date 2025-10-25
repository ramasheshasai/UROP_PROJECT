import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
import matplotlib.pyplot as plt
import os

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

os.makedirs("results", exist_ok=True)
with open("results/power_ensemble_report.txt", "w") as f:
    f.write(f"Power Boost Ensemble Accuracy: {acc}\n\n")
    f.write(report)

# ==============================
# 6️⃣ Shapley Value Interpretability (Updated)
# ==============================
# Use the new SHAP masker API
masker = shap.maskers.Independent(train_meta)

# Create explainer using current SHAP version syntax
explainer = shap.Explainer(meta_model, masker)
shap_values = explainer(test_meta)

# Plot summary (importance of base models in meta-model decision)
plt.figure()
shap.summary_plot(shap_values.values, test_meta, feature_names=[name for name, _ in base_models], show=False)
plt.savefig("results/shap_summary_meta_model.png", bbox_inches='tight')
plt.close()

# Optional: visualize for a single prediction
# ---------- Create folder for centralized SHAP results ----------
save_path = r"C:\Users\InduS\OneDrive\Desktop\UROP Project\Project-Test-1\HeartDisease\results\Centralised_Shapely"
os.makedirs(save_path, exist_ok=True)

# ---------- Save summary SHAP plot ----------
plt.figure()
shap.summary_plot(
    shap_values.values, 
    test_meta, 
    feature_names=[name for name, _ in base_models], 
    show=False
)
plt.savefig(os.path.join(save_path, "shap_summary_meta_model.png"), bbox_inches='tight')
plt.close()

# ---------- Save single prediction SHAP force plot ----------
sample_index = 5
plt.figure()
shap.plots.force(
    explainer.expected_value,
    shap_values.values[sample_index, :],
    feature_names=[name for name, _ in base_models],
    matplotlib=True,
    show=False
)
plt.savefig(os.path.join(save_path, "shap_single_prediction.png"), bbox_inches='tight')
plt.close()

print(f"\nSHAP interpretability plots saved in '{save_path}' folder.")
