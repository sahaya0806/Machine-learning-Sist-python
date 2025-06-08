import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

bc = load_breast_cancer()
X = bc.data
y = bc.target

df = pd.DataFrame(X, columns=bc.feature_names)
df['target'] = y
print(df.head(5))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scc = StandardScaler()
X_train = scc.fit_transform(X_train)
X_test = scc.transform(X_test)

params = {
    'n_estimators': [10, 50, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid = GridSearchCV(RandomForestClassifier(random_state=0), param_grid=params, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Best Parameters", grid.best_params_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("Accuracy", accuracy_score(y_test, y_pred))
print(np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1)), axis=1))
