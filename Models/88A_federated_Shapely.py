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

    # ✅ Impute missing values with median
    imputer = SimpleImputer(strategy="median")
    df = pd.DataFrame(imputer.fit_transform(df))

    if len(df) < 5:
        print(f"Skipping {name}: only {len(df)} usable rows.")
        continue

    X = df.iloc[:, :-1].values
    y = (df.iloc[:, -1] > 0).astype(int).values
    clients_data[name] = (X, y)
    print(f"Loaded {name} dataset with {len(y)} samples.")

# Ensure at least 2 datasets exist
if len(clients_data) < 2:
    raise ValueError("Not enough valid datasets for federated training!")

# ==============================
# 3️⃣ Scale data + create global test set
# ==============================
for key in clients_data:
    X, y = clients_data[key]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    clients_data[key] = (X, y)

X_train_clients, y_train_clients = [], []
X_test_global, y_test_global = [], []

for key, (X, y) in clients_data.items():
    if len(y) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = np.empty((0, X.shape[1])), np.empty((0,))
    X_train_clients.append(X_train)
    y_train_clients.append(y_train)
    X_test_global.append(X_test)
    y_test_global.append(y_test)

# Combine all global test data
X_test_global = np.concatenate(X_test_global) if any(len(x) for x in X_test_global) else np.empty((0,))
y_test_global = np.concatenate(y_test_global) if any(len(y) for y in y_test_global) else np.empty((0,))

# ==============================
# 4️⃣ Dataset + Model Definitions
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
# 5️⃣ Federated Utils
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
# 6️⃣ Federated Training
# ==============================
input_dim = next(iter(clients_data.values()))[0].shape[1]
global_model = MLP(input_dim).to(DEVICE)

rounds = 40
local_epochs = 3
batch_size = 16

client_loaders = []
client_sizes = []
for key, (X_train, y_train) in clients_data.items():
    ds = SimpleDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    client_loaders.append(loader)
    client_sizes.append(len(ds))

print("\nStarting Federated Training...\n")

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
            if len(X_test_global) > 0:
                Xte = torch.from_numpy(X_test_global).float().to(DEVICE)
                logits = global_model(Xte)
                preds = logits.argmax(dim=1).cpu().numpy()
                acc = accuracy_score(y_test_global, preds)
                print(f"Round {r:02d} - Global Test Accuracy: {acc:.4f}")
            else:
                print(f"Round {r:02d} - (No global test samples available)")

# ==============================
# 7️⃣ SHAP Analysis (Explainability)
# ==============================
if len(X_test_global) > 0:
    print("\nRunning SHAP explainability analysis...")

    Xte_tensor = torch.from_numpy(X_test_global).float().to(DEVICE)
    background = X_test_global[np.random.choice(X_test_global.shape[0], min(100, X_test_global.shape[0]), replace=False)]
    background_tensor = torch.from_numpy(background).float().to(DEVICE)

    explainer = shap.DeepExplainer(global_model, background_tensor)
    shap_values = explainer.shap_values(Xte_tensor)
    shap_values_np = np.array(shap_values)[1] if isinstance(shap_values, list) else shap_values

    feature_names = [f"Feature_{i+1}" for i in range(X_test_global.shape[1])]

    shap.summary_plot(shap_values_np, X_test_global, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot - Heart Disease (Federated Global Model)")
    plt.tight_layout()
    shap_summary_path = os.path.join(r"C:\Users\InduS\OneDrive\Desktop\UROP Project\Project-Test-1\HeartDisease\results\federated_Shap", "shap_summary_plot.png")
    plt.savefig(shap_summary_path, dpi=300)
    plt.close()

    shap.summary_plot(shap_values_np, X_test_global, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Bar Plot)")
    plt.tight_layout()
    shap_bar_path = os.path.join(r"C:\Users\InduS\OneDrive\Desktop\UROP Project\Project-Test-1\HeartDisease\results\federated_Shap", "shap_bar_plot.png")
    plt.savefig(shap_bar_path, dpi=300)
    plt.close()

    print(f"✅ SHAP analysis completed.\nSaved plots:\n - {shap_summary_path}\n - {shap_bar_path}")
else:
    print("\nSkipping SHAP analysis (no test data available).")

# ==============================
# 8️⃣ Save Results
# ==============================
save_path = r"C:\Users\InduS\OneDrive\Desktop\UROP Project\Project-Test-1\HeartDisease\results\federated_Shap"
os.makedirs(save_path, exist_ok=True)
output_file = os.path.join(save_path, "federated_global_imputed_all_datasets_report.txt")

with open(output_file, "w", encoding="utf-8") as f:
    if len(X_test_global) > 0:
        f.write(f"Final Global Model Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test_global, preds))
    else:
        f.write("No global test data available for evaluation.\n")

print(f"\n Results saved successfully at: {output_file}")
