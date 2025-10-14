import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load dataset
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

path = r"C:\Users\InduS\OneDrive\Desktop\UROP Project\Project-Test-1\HeartDisease\Dataset\processed.cleveland.data"
data = pd.read_csv(path, names=columns)

# Step 2: Clean missing values
data = data.replace('?', np.nan)
data = data.dropna()
data = data.astype(float)

# Step 3: Split features and target
X = data.drop('target', axis=1)
y = data['target']
y = np.where(y > 0, 1, 0)  # Convert multi-class to binary (0 = no disease, 1 = disease)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train a baseline model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
