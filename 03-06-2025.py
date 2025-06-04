import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv("load-train.csv")

imputer = SimpleImputer(strategy="mean")
dataset[["LoanAmount", "Loan_Amount_Term", "Credit_History"]] = imputer.fit_transform(
    dataset[["LoanAmount", "Loan_Amount_Term", "Credit_History"]]
)

label_columns = ["Gender", "Married", "Education", "Self_Employed"]
le = LabelEncoder()
for col in label_columns:
    dataset[col] = le.fit_transform(dataset[col].astype(str))

dataset["Loan_Status"] = LabelEncoder().fit_transform(dataset["Loan_Status"].astype(str))

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), ["Property_Area"])], remainder="passthrough")
X = ct.fit_transform(dataset.drop("Loan_Status", axis=1))
y = dataset["Loan_Status"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

sc = StandardScaler()
X_train[:, -4:] = sc.fit_transform(X_train[:, -4:])
X_test[:, -4:] = sc.transform(X_test[:, -4:])

kclas = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric="minkowski", p=2)
kclas.fit(X_train, y_train)

y_pred = kclas.predict(X_test)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix\n", cm)
print("Accuracy", accuracy_score(y_test, y_pred))
