# File: 7_ensemble_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
import torch
import torch.nn as nn

# ==============================
# 1️⃣ Load & preprocess dataset
# ==============================
df = pd.read_csv("Dataset/processed.cleveland.data", header=None)
df.replace('?', pd.NA, inplace=True)
df = df.dropna().apply(pd.to_numeric)

X = df.iloc[:, :-1].values
y = (df.iloc[:, -1] > 0).astype(int).values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==============================
# 2️⃣ Train MLP
# ==============================
mlp_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_dim=X_train.shape[1], activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

mlp_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

mlp_model.fit(X_train, y_train, epochs=150, batch_size=16, validation_split=0.1, verbose=0)
mlp_pred = (mlp_model.predict(X_test) > 0.5).astype(int).flatten()

# ==============================
# 3️⃣ Train TabTransformer
# ==============================
class TabTransformer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = x.unsqueeze(1)  # seq dimension
        x, _ = self.attn(x, x, x)
        x = x.squeeze(1)
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.out(x))

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

tt_model = TabTransformer(input_dim=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(tt_model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    tt_model.train()
    optimizer.zero_grad()
    output = tt_model(X_train_t)
    loss = criterion(output, y_train_t)
    loss.backward()
    optimizer.step()

# Predict
tt_model.eval()
with torch.no_grad():
    tt_pred = (tt_model(X_test_t) > 0.5).float().numpy().flatten()

# ==============================
# 4️⃣ Ensemble by Averaging
# ==============================
all_preds = np.vstack([mlp_pred, tt_pred])
ensemble_pred = np.round(np.mean(all_preds, axis=0)).astype(int)

# ==============================
# 5️⃣ Evaluate & Save
# ==============================
acc = accuracy_score(y_test, ensemble_pred)
report = classification_report(y_test, ensemble_pred)

print("Ensemble Accuracy:", acc)
print("\nClassification Report:\n", report)

with open("results/ensemble_report.txt", "w") as f:
    f.write(f"Ensemble Accuracy: {acc}\n\n")
    f.write(report)
