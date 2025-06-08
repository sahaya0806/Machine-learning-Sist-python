import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = load_breast_cancer()
X = data.data
y = data.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(random_state=42)
log_reg.fit(X_train_scaled, y_train)
tree_clf.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test_scaled)
y_pred_tree = tree_clf.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix", confusion_matrix(y_test, y_pred_log))
print("Classification Report", classification_report(y_test, y_pred_log))
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))
print("Classification Report", classification_report(y_test, y_pred_tree))
