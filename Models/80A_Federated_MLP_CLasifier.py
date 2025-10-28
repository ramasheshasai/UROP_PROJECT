import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
import shap
import matplotlib.pyplot as plt

# ==============================
# 1️⃣ Setup
# ==============================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# 2️⃣ Load & preprocess datasets
# ==============================
paths = {
    "cleveland": "Dataset/processed.cleveland.data",
    "hungarian": "Dataset/processed.hungarian.data",
    "switzerland": "Dataset/processed.switzerland.data",
    "va": "Dataset/processed.va.data"
}

clients_data = {}
for name, path in paths.items():
    df = pd.read_csv(path, header=None)
    df.replace('?', np.nan, inplace=True)
    
    # ✅ Median imputation
    imputer = SimpleImputer(strategy="median")
    df = pd.DataFrame(imputer.fit_transform(df))
    
    if len(df) < 5:
        print(f"Skipping {name}: only {len(df)} usable rows.")
        continue
    
    X = df.iloc[:, :-1].values
    y = (df.iloc[:, -1] > 0).astype(int).values
    clients_data[name] = (X, y)
    print(f"Loaded {name} dataset with {len(y)} samples.")

if len(clients_data) < 2:
    raise ValueError("Not enough valid datasets for federated training!")

# ==============================
# 3️⃣ Scaling and Global Test Set
# ==============================
for key in clients_data:
    X, y = clients_data[key]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    clients_data[key] = (X, y)

X_train_clients, y_train_clients = [], []
X_test_global, y_test_global = [], []

for key, (X, y) in clients_data.items():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train_clients.append(X_train)
    y_train_clients.append(y_train)
    X_test_global.append(X_test)
    y_test_global.append(y_test)

X_test_global = np.concatenate(X_test_global)
y_test_global = np.concatenate(y_test_global)

# ==============================
# 4️⃣ Model Definition
# ==============================
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

class MLP(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

# ==============================
# 5️⃣ Federated Functions
# ==============================
def get_model_params(model):
    return {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}

def set_model_params(model, params):
    model.load_state_dict(params)

def average_models(state_dicts, weights=None):
    avg = {}
    n = len(state_dicts)
    if weights is None:
        weights = [1.0 / n] * n
    for k in state_dicts[0].keys():
        avg[k] = sum([state_dicts[i][k].float() * weights[i] for i in range(n)])
    return avg

def local_train(model, dataloader, epochs=3, lr=1e-3):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
    return model

# ==============================
# 6️⃣ Federated Training (50 rounds)
# ==============================
input_dim = next(iter(clients_data.values()))[0].shape[1]
global_model = MLP(input_dim).to(DEVICE)

rounds = 50
local_epochs = 3
batch_size = 16

client_loaders = []
client_sizes = []
for key, (X_train, y_train) in clients_data.items():
    ds = SimpleDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    client_loaders.append(loader)
    client_sizes.append(len(ds))

print("\nStarting Federated Training (50 Rounds)...\n")

for r in range(1, rounds + 1):
    local_states = []
    for i, loader in enumerate(client_loaders):
        local_model = MLP(input_dim).to(DEVICE)
        set_model_params(local_model, get_model_params(global_model))
        local_model = local_train(local_model, loader, epochs=local_epochs, lr=1e-3)
        local_states.append(get_model_params(local_model))

    weights = [s / sum(client_sizes) for s in client_sizes]
    avg_state = average_models(local_states, weights=weights)
    set_model_params(global_model, avg_state)

    if r % 5 == 0 or r == 1:
        global_model.eval()
        with torch.no_grad():
            Xte = torch.from_numpy(X_test_global).float().to(DEVICE)
            logits = global_model(Xte)
            preds = logits.argmax(dim=1).cpu().numpy()
            acc = accuracy_score(y_test_global, preds)
            print(f"Round {r:02d} - Global Test Accuracy: {acc:.4f}")

# ==============================
# 7️⃣ Save Federated Results
# ==============================
save_path = r"C:\Users\InduS\OneDrive\Desktop\UROP Project\Project-Test-1\HeartDisease\results\federated"
os.makedirs(save_path, exist_ok=True)
output_file = os.path.join(save_path, "federated_global_50rounds_report.txt")

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Final Global Model Accuracy: {acc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test_global, preds))

print(f"\n Results saved successfully at: {output_file}")



# ==============================
# 8️⃣ SHAP Feature Importance (Federated Global Model)
# ==============================
import shap
import matplotlib.pyplot as plt
import numpy as np

print("\nStarting SHAP feature importance analysis for Federated Global Model...")

# Move global model to CPU for SHAP
global_model_cpu = MLP(input_dim)
set_model_params(global_model_cpu, get_model_params(global_model))
global_model_cpu.eval()

# Convert test data
X_sample = torch.tensor(X_test_global).float()

# Use a smaller subset to speed up SHAP computation
sample_size = min(100, len(X_sample))
indices = np.random.choice(len(X_sample), size=sample_size, replace=False)
X_explain = X_sample[indices]

# Initialize SHAP DeepExplainer
explainer = shap.DeepExplainer(global_model_cpu, X_explain)
shap_values = explainer.shap_values(X_explain)

# If SHAP returns list for each class, select class 1
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Define feature names (UCI heart dataset)
feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

# ==============================
# SHAP Summary Plot
# ==============================
plt.figure()
shap.summary_plot(shap_values, X_explain.detach().numpy(), feature_names=feature_names, show=False)
summary_path = os.path.join(save_path, "federated_shap_summary_plot.png")
plt.savefig(summary_path, bbox_inches="tight", dpi=300)
plt.close()
print(f"SHAP summary plot saved at: {summary_path}")

# ==============================
# SHAP Bar Plot
# ==============================
plt.figure()
shap.summary_plot(shap_values, X_explain.detach().numpy(), feature_names=feature_names, plot_type="bar", show=False)
bar_path = os.path.join(save_path, "federated_shap_bar_plot.png")
plt.savefig(bar_path, bbox_inches="tight", dpi=300)
plt.close()
print(f"SHAP bar plot saved at: {bar_path}")

# # ==============================
# # Save Top 10 Features as Text (Fixed)
# # ==============================
# mean_abs_shap = np.abs(shap_values).mean(axis=0)

# # Flatten to ensure no nested list issues
# mean_abs_shap = np.array(mean_abs_shap).flatten()

# # Zip with feature names and sort descending
# feature_importance = sorted(
#     zip(feature_names, mean_abs_shap),
#     key=lambda x: float(x[1]),
#     reverse=True
# )

# txt_path = os.path.join(save_path, "federated_shap_feature_importance.txt")
# with open(txt_path, "w", encoding="utf-8") as f:
#     f.write("Top 10 Most Important Features (Federated Model SHAP)\n")
#     f.write("============================================\n")
#     for i, (feat, val) in enumerate(feature_importance[:10], start=1):
#         f.write(f"{i}. {feat} — {float(val):.4f}\n")

# print(f"SHAP feature importance text file saved at: {txt_path}")

# print("\nTop 10 Most Important Features (Federated Model):")
# for i, (feat, val) in enumerate(feature_importance[:10], start=1):
#     print(f"{i}. {feat} — {float(val):.4f}")
