import pandas as pd
import numpy as np

df = pd.read_csv("diabetes.csv")

invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df = df.astype(float)
for col in invalid_cols:
    df[col] = df[col].replace(0, np.nan)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
