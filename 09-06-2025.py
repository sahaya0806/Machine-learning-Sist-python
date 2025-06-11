import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def rank_models(models, X_test, y_test, metric='accuracy'):
    metric_funcs = {
        'accuracy': accuracy_score,
        'f1': f1_score,
        'roc_auc': roc_auc_score
    }

    if metric not in metric_funcs:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from: {list(metric_funcs.keys())}")

    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        if metric == 'roc_auc':
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)
            else:
                raise ValueError(f"Model {name} does not support probability estimates for ROC AUC.")
            score = roc_auc_score(y_test, y_prob)
        else:
            average = 'binary' if len(set(y_test)) == 2 else 'macro'
            score = metric_funcs[metric](y_test, y_pred) if metric != 'f1' else f1_score(y_test, y_pred, average=average)

        results.append({'Model': name, metric.capitalize(): round(score, 4)})

    df = pd.DataFrame(results).sort_values(by=metric.capitalize(), ascending=False).reset_index(drop=True)
    return df
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier()
}
for model in models.values():
    model.fit(X_train, y_train)
print(rank_models(models, X_test, y_test, metric='accuracy'))
print(rank_models(models, X_test, y_test, metric='f1'))
print(rank_models(models, X_test, y_test, metric='roc_auc'))
