
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load & preprocess (adapt path if needed) ----------
df = pd.read_csv("Dataset/processed.cleveland.data", header=None)
df.replace('?', pd.NA, inplace=True)
df = df.dropna().apply(pd.to_numeric)
X = df.iloc[:, :-1].values
y = (df.iloc[:, -1] > 0).astype(int).values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# train/test global split (held-out test to evaluate federated global model)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# ---------- Partition into 4 clients using Dirichlet (non-iid) ----------
def dirichlet_partition(y, n_clients=4, alpha=0.5):
    classes = np.unique(y)
    client_idx = [[] for _ in range(n_clients)]
    for c in classes:
        idx_c = np.where(y == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, n_clients))
        # scale proportions to number of items
        counts = (proportions * len(idx_c)).astype(int)
        # fix rounding issues
        while counts.sum() < len(idx_c):
            counts[np.argmax(proportions)] += 1
        ptr = 0
        for i in range(n_clients):
            ccount = counts[i]
            if ccount > 0:
                client_idx[i].extend(idx_c[ptr:ptr+ccount].tolist())
                ptr += ccount
    return client_idx

n_clients = 4
client_indices = dirichlet_partition(y_trainval, n_clients=n_clients, alpha=0.3)

# ---------- PyTorch dataset helper ----------
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, idxs):
        self.X = torch.from_numpy(X[idxs]).float()
        self.y = torch.from_numpy(y[idxs]).long()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

# ---------- Simple MLP model ----------
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

# ---------- Federated training utils ----------
def get_model_params(model):
    return {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}

def set_model_params(model, params):
    model.load_state_dict(params)

def average_models(state_dicts, weights=None):
    avg = {}
    n = len(state_dicts)
    if weights is None:
        
        weights = [1.0/n]*n
    for k in state_dicts[0].keys():
        avg[k] = sum([state_dicts[i][k].float()*weights[i] for i in range(n)])
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

# ---------- Federated loop ----------
input_dim = X_trainval.shape[1]
global_model = MLP(input_dim).to(DEVICE)
rounds = 40
local_epochs = 3
batch_size = 16

# Pre-create dataloaders for clients
client_loaders = []
client_sizes = []
for idxs in client_indices:
    ds = SimpleDataset(X_trainval, y_trainval, idxs)
    client_sizes.append(len(ds))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    client_loaders.append(loader)

for r in range(1, rounds+1):
    local_states = []
    # each client trains locally
    for i in range(n_clients):
        client_model = MLP(input_dim).to(DEVICE)
        set_model_params(client_model, get_model_params(global_model))
        client_model = local_train(client_model, client_loaders[i], epochs=local_epochs, lr=1e-3)
        local_states.append(get_model_params(client_model))
    # FedAvg weighted by number of samples
    weights = [s/ sum(client_sizes) for s in client_sizes]
    avg_state = average_models(local_states, weights=weights)
    set_model_params(global_model, avg_state)
    if r % 5 == 0 or r == 1:
        global_model.eval()
        with torch.no_grad():
            Xte = torch.from_numpy(X_test).float().to(DEVICE)
            logits = global_model(Xte)
            preds = logits.argmax(dim=1).cpu().numpy()
            acc = accuracy_score(y_test, preds)
            print(f"Round {r:02d}  Global test accuracy: {acc:.4f}")

# ---------- Save results to absolute path ----------
save_path = r"C:\Users\InduS\OneDrive\Desktop\UROP Project\Project-Test-1\HeartDisease\results\federated"
os.makedirs(save_path, exist_ok=True)

output_file = os.path.join(save_path, "federated_global_model_report.txt")

with open(output_file, "w") as f:
    f.write(f"Final Global Model Accuracy: {acc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, preds))

print(f"\nResults saved successfully at: {output_file}")
