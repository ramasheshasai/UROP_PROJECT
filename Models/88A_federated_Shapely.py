import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import RidgeClassifier
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

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==============================
# 2️⃣ Simulate federated clients
# ==============================
n_clients = 3
client_data = np.array_split(X_train_full, n_clients)
client_labels = np.array_split(y_train_full, n_clients)

# ==============================
# 3️⃣ Define base models
# ==============================
base_models = [
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=300, max_depth=6, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42))
]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_meta_full = np.zeros((X_train_full.shape[0], len(base_models)))
test_meta = np.zeros((X_test.shape[0], len(base_models)))

# ==============================
# 4️⃣ Train base models federated way
# ==============================
for i, (name, model) in enumerate(base_models):
    test_fold_preds = np.zeros((X_test.shape[0], kf.n_splits))
    fold_indices = list(kf.split(X_train_full))
    
    for fold, (tr_idx, val_idx) in enumerate(fold_indices):
        # Aggregate client data for this fold
        X_tr_fold = np.vstack([client_data[c][np.isin(range(len(client_data[c])), tr_idx)] for c in range(n_clients)])
        y_tr_fold = np.hstack([client_labels[c][np.isin(range(len(client_labels[c])), tr_idx)] for c in range(n_clients)])
        X_val_fold = X_train_full[val_idx]
        y_val_fold = y_train_full[val_idx]

        model.fit(X_tr_fold, y_tr_fold)
        train_meta_full[val_idx, i] = model.predict(X_val_fold)
        test_fold_preds[:, fold] = model.predict(X_test)
    
    test_meta[:, i] = test_fold_preds.mean(axis=1)

# ==============================
# 5️⃣ Meta-model (centralized)
# ==============================
meta_model = RidgeClassifier()
meta_model.fit(train_meta_full, y_train_full)
stack_pred = meta_model.predict(test_meta)

acc = accuracy_score(y_test, stack_pred)
report = classification_report(y_test, stack_pred)

print("Federated Power Boost Ensemble Accuracy:", acc)
print(report)

# ==============================
# 6️⃣ Federated SHAP computation (corrected)
# ==============================
# Compute SHAP on **full train_meta_full**, no averaging mismatch
masker = shap.maskers.Independent(train_meta_full)
explainer = shap.Explainer(meta_model, masker)
shap_values = explainer(train_meta_full)

# ==============================
# 7️⃣ Save SHAP plots to absolute path
# ==============================
save_path = r"C:\Users\InduS\OneDrive\Desktop\UROP Project\Project-Test-1\HeartDisease\results\federated_shapley"
os.makedirs(save_path, exist_ok=True)

plt.figure()
shap.summary_plot(shap_values.values, train_meta_full, feature_names=[name for name, _ in base_models], show=False)
plt.savefig(os.path.join(save_path, "shap_summary_meta_model.png"), bbox_inches='tight')
plt.close()

print(f"Federated SHAP plot saved successfully at: {save_path}")
