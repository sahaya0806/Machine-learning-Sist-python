from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Confusion Matrix\n", confusion_matrix(y_test, y_pred))
print("Accuracy", round(accuracy_score(y_test, y_pred), 4))
print("Precision ", round(precision_score(y_test, y_pred, average='macro'), 4))
print("Recall", round(recall_score(y_test, y_pred, average='macro'), 4))
print("F1 Score ", round(f1_score(y_test, y_pred, average='macro'), 4))
y_test_bin = label_binarize(y_test, classes=range(10))
y_score = model.predict_proba(X_test)
roc_auc = roc_auc_score(y_test_bin, y_score, average='macro', multi_class='ovr')
print("ROC-AUC (One-vs-Rest)", round(roc_auc, 4))
