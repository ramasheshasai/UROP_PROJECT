import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("Dataset/processed.cleveland.data", header=None)

# Replace missing values
df.replace('?', pd.NA, inplace=True)
df = df.dropna()

# Convert all columns to numeric
df = df.apply(pd.to_numeric)

# Split data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
y = (y > 0).astype(int)  # 0 = No Disease, 1 = Disease

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train baseline model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
