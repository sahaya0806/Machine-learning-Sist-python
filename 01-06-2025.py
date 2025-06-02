import pandas as pd
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
models = {}
results = {}
from sklearn.ensemble import RandomForestClassifier
models['Random Forest'] = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
from sklearn.neighbors import KNeighborsClassifier
models['KNN'] = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
from sklearn.tree import DecisionTreeClassifier
models['Decision Tree'] = DecisionTreeClassifier(criterion='entropy', random_state=0)
from sklearn.linear_model import LogisticRegression
models['Logistic Regression'] = LogisticRegression(random_state=0)
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)  
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }
for name, metrics in results.items():
    print(f"\n{name} Results:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}" if metric_name != 'Confusion Matrix' else f"{metric_name}:\n{value}")
