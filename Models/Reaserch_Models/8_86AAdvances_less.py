# File: 8_advanced_ensemble.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
    tf.keras.layers.BatchNormalization(),
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
        self.fc1 = nn.Linear(input_dim, 128)
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = x.unsqueeze(1)  # seq dim
        x, _ = self.attn(x, x, x)
        x = x.squeeze(1)
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.out(x))

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

tt_model = TabTransformer(input_dim=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(tt_model.parameters(), lr=0.001)

for epoch in range(150):
    tt_model.train()
    optimizer.zero_grad()
    output = tt_model(X_train_t)
    loss = criterion(output, y_train_t)
    loss.backward()
    optimizer.step()

tt_model.eval()
with torch.no_grad():
    tt_pred = (tt_model(X_test_t) > 0.5).float().numpy().flatten()

# ==============================
# 4️⃣ Combine predictions into features for stacking
# ==============================
stack_train_features = np.column_stack([
    (mlp_model.predict(X_train) > 0.5).astype(int).flatten(),
    (tt_model(X_train_t).detach().numpy() > 0.5).astype(int).flatten()
])

stack_test_features = np.column_stack([mlp_pred, tt_pred])

# ==============================
# 5️⃣ Train meta-model (Logistic Regression)
# ==============================
meta_model = LogisticRegression()
meta_model.fit(stack_train_features, y_train)

stack_pred = meta_model.predict(stack_test_features)

# ==============================
# 6️⃣ Weighted averaging ensemble (optional)
# ==============================
weights = [0.55, 0.45]  # give slightly higher weight to MLP
weighted_pred = np.round(np.average(stack_test_features, axis=1, weights=weights)).astype(int)

# ==============================
# 7️⃣ Evaluate
# ==============================
final_pred = stack_pred  # choose stack_pred or weighted_pred

acc = accuracy_score(y_test, final_pred)
report = classification_report(y_test, final_pred)

print("Advanced Ensemble Accuracy:", acc)
print("\nClassification Report:\n", report)

with open("results/advanced_ensemble_report.txt", "w") as f:
    f.write(f"Advanced Ensemble Accuracy: {acc}\n\n")
    f.write(report)
