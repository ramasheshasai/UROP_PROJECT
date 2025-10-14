import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("Dataset/processed.cleveland.data", header=None)
df.replace('?', pd.NA, inplace=True)
df = df.dropna().apply(pd.to_numeric)

X = df.iloc[:, :-1]
y = (df.iloc[:, -1] > 0).astype(int)
columns = [f"Feature_{i}" for i in range(X.shape[1])]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(learning_rate=0.05, n_estimators=300, max_depth=5, random_state=42)
model.fit(X_train, y_train)

explainer = shap.Explainer(model)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, feature_names=columns)
plt.savefig("results/shap_summary.png")
