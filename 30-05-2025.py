import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dataset = pd.read_csv("Iris.csv")
X = dataset.iloc[ :  , : -1].values
y = dataset.iloc[ :  , -1].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train , X_test,y_train , y_test = train_test_split(X,y,test_size =0.2,random_state=0 )
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
kclasii = KNeighborsClassifier(n_neighbors=5,weights="uniform",metric='minkowski',p=2)
kclasii.fit(X_train, y_train)
y_pred = kclasii.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)